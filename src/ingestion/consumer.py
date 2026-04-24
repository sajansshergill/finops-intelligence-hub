"""
consumer.py
-----------
Consumes billing events from Kafka, writes raw Parquet files (GCS equivalent),
and loads into DuckDB (BigQuery equivalent) for downstream analytics.

GCP Equivalent: Pub/Sub → Dataflow → BigQuery

Usage:
    python src/ingestion/consumer.py
    python src/ingestion/consumer.py --batch-size 1000 --flush-interval 60
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from schema import BillingEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("finops.consumer")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KAFKA_TOPIC = "billing-events"
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "finops-consumer-group")

DATA_DIR = Path(os.environ.get("FINOPS_DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"          # Parquet landing zone (GCS equivalent)
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "finops.duckdb"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DuckDB schema setup
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS billing_events (
    event_id        VARCHAR PRIMARY KEY,
    timestamp       TIMESTAMPTZ,
    project_id      VARCHAR,
    department      VARCHAR,
    service_sku     VARCHAR,
    region          VARCHAR,
    cost_usd        DOUBLE,
    usage_amount    DOUBLE,
    usage_unit      VARCHAR,
    label_team      VARCHAR,
    label_env       VARCHAR,
    is_anomaly      BOOLEAN,
    ingested_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS anomaly_scores (
    event_id            VARCHAR,
    scored_at           TIMESTAMPTZ,
    z_score             DOUBLE,
    isolation_score     DOUBLE,
    anomaly_confidence  DOUBLE,
    is_flagged          BOOLEAN,
    PRIMARY KEY (event_id, scored_at)
);

CREATE TABLE IF NOT EXISTS forecasts (
    project_id      VARCHAR,
    service_sku     VARCHAR,
    forecast_date   DATE,
    predicted_cost  DOUBLE,
    lower_80        DOUBLE,
    upper_80        DOUBLE,
    lower_95        DOUBLE,
    upper_95        DOUBLE,
    model_run_id    VARCHAR,
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (project_id, service_sku, forecast_date, model_run_id)
);

CREATE TABLE IF NOT EXISTS alert_history (
    alert_id        VARCHAR PRIMARY KEY,
    triggered_at    TIMESTAMPTZ,
    project_id      VARCHAR,
    service_sku     VARCHAR,
    alert_type      VARCHAR,
    severity        VARCHAR,
    cost_usd        DOUBLE,
    threshold_usd   DOUBLE,
    message         VARCHAR,
    resolved        BOOLEAN DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ
);
"""


def init_db(conn: duckdb.DuckDBPyConnection, *, log: bool = True) -> None:
    """Create all tables if they don't exist."""
    conn.execute(DDL)
    if log:
        logger.info(f"DuckDB initialized at {DB_PATH}")


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def write_parquet(batch: list[dict], partition_dt: datetime) -> Path:
    """
    Write a batch of events to a date-partitioned Parquet file.
    Mirrors the GCS billing export directory structure.
    """
    partition = partition_dt.strftime("%Y/%m/%d")
    out_dir = RAW_DIR / partition
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"billing_{partition_dt.strftime('%H%M%S')}_{len(batch)}.parquet"
    out_path = out_dir / filename

    df = pd.DataFrame(batch)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.to_parquet(out_path, index=False, compression="snappy")

    logger.info(f"Wrote {len(batch):,} events → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# DuckDB loader
# ---------------------------------------------------------------------------

def load_to_duckdb(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: Path,
) -> int:
    """
    Load a Parquet file into DuckDB billing_events table.
    Uses INSERT OR IGNORE semantics to handle duplicates (idempotent).
    Returns number of rows inserted.
    """
    before = conn.execute("SELECT COUNT(*) FROM billing_events").fetchone()[0]

    conn.execute(f"""
        INSERT OR IGNORE INTO billing_events
        SELECT
            event_id,
            timestamp::TIMESTAMPTZ,
            project_id,
            department,
            service_sku,
            region,
            cost_usd,
            usage_amount,
            usage_unit,
            label_team,
            label_env,
            is_anomaly,
            now() AS ingested_at
        FROM read_parquet('{parquet_path}')
    """)

    after = conn.execute("SELECT COUNT(*) FROM billing_events").fetchone()[0]
    inserted = after - before
    logger.info(f"Loaded {inserted:,} new rows into DuckDB (total: {after:,})")
    return inserted


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------

class BillingConsumer:
    def __init__(
        self,
        batch_size: int = 1000,
        flush_interval: int = 60,
        max_messages: int | None = None,
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_messages = max_messages
        self.buffer: list[dict] = []
        self.last_flush = time.time()
        self.total_consumed = 0
        self.running = True
        self._duckdb_init_logged = False

        # Graceful shutdown on SIGINT / SIGTERM
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame) -> None:
        logger.info("Shutdown signal received — flushing buffer...")
        self.running = False
        if self.buffer:
            self._flush()
        logger.info(f"Consumer stopped. Total events consumed: {self.total_consumed:,}")
        sys.exit(0)

    def _flush(self) -> None:
        """Write buffer to Parquet and load into DuckDB."""
        if not self.buffer:
            return

        parquet_path = write_parquet(self.buffer, datetime.utcnow())
        conn = duckdb.connect(str(DB_PATH))
        try:
            init_db(conn, log=not self._duckdb_init_logged)
            self._duckdb_init_logged = True
            load_to_duckdb(conn, parquet_path)
        finally:
            conn.close()

        self.buffer.clear()
        self.last_flush = time.time()

    def _should_flush(self) -> bool:
        return (
            len(self.buffer) >= self.batch_size
            or (time.time() - self.last_flush) >= self.flush_interval
        )

    def run(self, retries: int = 5) -> None:
        """Start consuming from Kafka."""
        consumer = self._build_consumer(retries)

        logger.info(
            f"Consuming from topic '{KAFKA_TOPIC}' "
            f"(batch_size={self.batch_size}, flush_interval={self.flush_interval}s)"
        )

        try:
            for message in consumer:
                if not self.running:
                    break

                event_dict = message.value
                self.buffer.append(event_dict)
                self.total_consumed += 1

                if self.total_consumed % 500 == 0:
                    logger.info(
                        f"Consumed {self.total_consumed:,} events | "
                        f"buffer={len(self.buffer)} | "
                        f"offset={message.offset}"
                    )

                if self._should_flush():
                    self._flush()

                if self.max_messages is not None and self.total_consumed >= self.max_messages:
                    logger.info(f"Stopping after {self.max_messages:,} messages (--max-messages)")
                    break

        except Exception as e:
            logger.error(f"Consumer error: {e}")
            raise
        finally:
            if self.buffer:
                self._flush()
            consumer.close()
            logger.info(f"Consumer closed. Total events: {self.total_consumed:,}")

    def _build_consumer(self, retries: int) -> KafkaConsumer:
        import json

        for attempt in range(1, retries + 1):
            try:
                consumer = KafkaConsumer(
                    KAFKA_TOPIC,
                    bootstrap_servers=KAFKA_BOOTSTRAP,
                    group_id=KAFKA_GROUP_ID,
                    auto_offset_reset="earliest",
                    enable_auto_commit=True,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    max_poll_records=500,
                    session_timeout_ms=30000,
                    heartbeat_interval_ms=10000,
                )
                logger.info(f"Connected to Kafka at {KAFKA_BOOTSTRAP}")
                return consumer
            except NoBrokersAvailable:
                logger.warning(
                    f"Kafka not available (attempt {attempt}/{retries}). Retrying in 5s..."
                )
                time.sleep(5)

        raise RuntimeError(
            f"Could not connect to Kafka after {retries} attempts. "
            "Is Docker running? Try: docker-compose up -d"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FinOps Billing Event Consumer")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Flush to DuckDB every N events (default: 1000)",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=60,
        help="Flush to DuckDB every N seconds regardless of batch size (default: 60)",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        metavar="N",
        help="Exit after consuming N messages (for batch / Airflow runs; default: run forever)",
    )
    args = parser.parse_args()

    consumer = BillingConsumer(
        batch_size=args.batch_size,
        flush_interval=args.flush_interval,
        max_messages=args.max_messages,
    )
    consumer.run()


if __name__ == "__main__":
    main()
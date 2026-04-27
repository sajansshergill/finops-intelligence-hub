"""
producer.py
-----------
Generates synthetic GCP billing events using Faker and publishes
them to a Kafka topic. Simulates 18 months of realistic cloud spend
for a financial services firm with injected anomalies.

GCP Equivalent: Cloud Billing Export → Pub/Sub topic

Usage:
    python src/ingestion/producer.py
    python src/ingestion/producer.py --months 18 --anomaly-rate 0.03
    python src/ingestion/producer.py --count 500000
    python src/ingestion/producer.py --mode live   # continuous stream
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Generator

import numpy as np
from faker import Faker
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from schema import (
    BillingEvent,
    DEPARTMENTS,
    PROJECTS,
    REGIONS,
    SERVICE_SKUS,
    USAGE_UNITS,
    ENVIRONMENTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("finops.producer")

KAFKA_TOPIC = "billing-events"
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

fake = Faker()
rng = np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# Cost generation helpers
# ---------------------------------------------------------------------------


def base_cost(sku: str, date: datetime) -> float:
    """
    Generate a realistic base cost for a SKU on a given date.
    Applies day-of-week and month-end seasonality patterns.
    """
    _, price_per_unit, max_units = USAGE_UNITS[sku]

    # Draw from a log-normal distribution for realistic spend variance
    usage = rng.lognormal(mean=np.log(max_units * 0.3), sigma=0.6)
    usage = float(np.clip(usage, 1, max_units))

    cost = usage * price_per_unit

    # Weekday multiplier — lower spend on weekends
    if date.weekday() >= 5:
        cost *= rng.uniform(0.3, 0.6)

    # Month-end spike — batch jobs, reporting runs
    if date.day >= 28:
        cost *= rng.uniform(1.8, 3.5)

    # Quarter-close amplifier
    if date.month in [3, 6, 9, 12] and date.day >= 25:
        cost *= rng.uniform(1.5, 2.5)

    return round(float(cost), 4)


def inject_anomaly(cost: float, sku: str) -> tuple[float, float]:
    """
    Inject one of three anomaly types:
      - Spike: sudden 8-15x cost surge
      - Drop:  near-zero spend (misconfiguration / outage)
      - Drift: sustained 3-5x elevation
    Returns (anomalous_cost, usage_amount).
    """
    _, price_per_unit, _ = USAGE_UNITS[sku]
    anomaly_type = rng.choice(["spike", "drop", "drift"], p=[0.6, 0.2, 0.2])

    if anomaly_type == "spike":
        multiplier = rng.uniform(8, 15)
        anomalous_cost = cost * multiplier
    elif anomaly_type == "drop":
        anomalous_cost = cost * rng.uniform(0.01, 0.05)
    else:  # drift
        anomalous_cost = cost * rng.uniform(3, 5)

    usage = anomalous_cost / price_per_unit if price_per_unit > 0 else 0
    return round(float(anomalous_cost), 4), round(float(usage), 4)


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------


def generate_events(
    months: int = 18,
    anomaly_rate: float = 0.03,
    max_events: int | None = None,
) -> Generator[BillingEvent, None, None]:
    """
    Yield BillingEvent objects covering `months` of historical data.
    Generates ~5-15 events per day per project/SKU combination.
    If ``max_events`` is set, stops after that many events (extends the
    simulated calendar span as needed so enough days exist).
    """
    end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    days_span = months * 30
    if max_events is not None:
        # ~200–800 events/day; keep enough days to reach max_events comfortably
        days_span = max(days_span, max_events // 200 + 365)
    start_date = end_date - timedelta(days=days_span)

    # Build a stable project → department mapping
    project_dept = {proj: random.choice(DEPARTMENTS) for proj in PROJECTS}

    current = start_date
    total_events = 0

    while current <= end_date:
        # Pick a random subset of project/SKU combos active on this day
        active_combos = [
            (proj, sku)
            for proj in PROJECTS
            for sku in random.sample(SERVICE_SKUS, k=random.randint(8, 15))
        ]

        for project_id, sku in active_combos:
            # Number of billing line items for this project/SKU today
            n_events = int(rng.integers(3, 8))

            for _ in range(n_events):
                # Random time within the day
                event_time = current + timedelta(
                    hours=int(rng.integers(0, 24)),
                    minutes=int(rng.integers(0, 60)),
                )

                cost = base_cost(sku, event_time)
                _, price_per_unit, _ = USAGE_UNITS[sku]
                usage = cost / price_per_unit if price_per_unit > 0 else 0
                is_anomaly = rng.random() < anomaly_rate

                if is_anomaly:
                    cost, usage = inject_anomaly(cost, sku)

                event = BillingEvent.create(
                    project_id=project_id,
                    department=project_dept[project_id],
                    service_sku=sku,
                    region=random.choice(REGIONS),
                    cost_usd=cost,
                    usage_amount=usage,
                    label_env=random.choice(ENVIRONMENTS),
                    timestamp=event_time,
                    is_anomaly=is_anomaly,
                )

                total_events += 1
                yield event
                if max_events is not None and total_events >= max_events:
                    logger.info(
                        f"Generated {total_events:,} billing events "
                        f"(stopped at --count limit)"
                    )
                    return

        current += timedelta(days=1)

    if max_events is None:
        logger.info(f"Generated {total_events:,} billing events over {months} months")
    elif total_events < max_events:
        logger.warning(
            f"Only generated {total_events:,} events (requested {max_events:,}); "
            "increase --months or widen span in generate_events"
        )


def generate_live_events(
    interval_seconds: float = 2.0,
    anomaly_rate: float = 0.03,
) -> Generator[BillingEvent, None, None]:
    """
    Continuously yield real-time billing events (simulates live Pub/Sub ingestion).
    """
    project_dept = {proj: random.choice(DEPARTMENTS) for proj in PROJECTS}

    logger.info(f"Starting live event stream (interval={interval_seconds}s)")
    while True:
        project_id = random.choice(PROJECTS)
        sku = random.choice(SERVICE_SKUS)
        now = datetime.utcnow()

        cost = base_cost(sku, now)
        _, price_per_unit, _ = USAGE_UNITS[sku]
        usage = cost / price_per_unit if price_per_unit > 0 else 0
        is_anomaly = rng.random() < anomaly_rate

        if is_anomaly:
            cost, usage = inject_anomaly(cost, sku)

        event = BillingEvent.create(
            project_id=project_id,
            department=project_dept[project_id],
            service_sku=sku,
            region=random.choice(REGIONS),
            cost_usd=cost,
            usage_amount=usage,
            label_env=random.choice(ENVIRONMENTS),
            timestamp=now,
            is_anomaly=is_anomaly,
        )

        yield event
        time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Kafka publishing
# ---------------------------------------------------------------------------


def build_producer(retries: int = 5) -> KafkaProducer:
    """Build a KafkaProducer with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                compression_type="gzip",
                batch_size=16384,
                linger_ms=10,
            )
            logger.info(f"Connected to Kafka at {KAFKA_BOOTSTRAP}")
            return producer
        except NoBrokersAvailable:
            logger.warning(
                f"Kafka not available (attempt {attempt}/{retries}). "
                f"Retrying in 5s..."
            )
            time.sleep(5)

    raise RuntimeError(
        f"Could not connect to Kafka at {KAFKA_BOOTSTRAP} after {retries} attempts. "
        "Is Docker running? Try: docker-compose up -d"
    )


def publish(
    producer: KafkaProducer,
    event: BillingEvent,
    topic: str = KAFKA_TOPIC,
) -> None:
    """Publish a single BillingEvent to Kafka."""
    producer.send(
        topic=topic,
        key=event.project_id,  # partition by project for ordering
        value=event.to_dict(),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FinOps Billing Event Producer")
    parser.add_argument(
        "--mode",
        choices=["historical", "live"],
        default="historical",
        help="historical: replay 18 months of data | live: continuous stream",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=18,
        help="Number of months of historical data to generate (default: 18)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="N",
        help="Historical mode: publish exactly N events then stop (default: use full months span)",
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.03,
        help="Fraction of events that are anomalies (default: 0.03 = 3%%)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between live events (default: 2.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Flush Kafka producer every N events (default: 500)",
    )
    args = parser.parse_args()

    producer = build_producer()

    try:
        if args.mode == "historical":
            if args.count is not None:
                logger.info(
                    f"Publishing {args.count:,} historical billing events "
                    f"(anomaly_rate={args.anomaly_rate:.1%})"
                )
            else:
                logger.info(
                    f"Publishing {args.months} months of historical billing data "
                    f"(anomaly_rate={args.anomaly_rate:.1%})"
                )
            count = 0
            for event in generate_events(
                months=args.months,
                anomaly_rate=args.anomaly_rate,
                max_events=args.count,
            ):
                publish(producer, event)
                count += 1

                if count % args.batch_size == 0:
                    producer.flush()
                    logger.info(f"Published {count:,} events...")

            producer.flush()
            logger.info(f"Done. Published {count:,} events to topic '{KAFKA_TOPIC}'")

        else:  # live
            logger.info("Starting live billing event stream. Press Ctrl+C to stop.")
            count = 0
            for event in generate_live_events(
                interval_seconds=args.interval,
                anomaly_rate=args.anomaly_rate,
            ):
                publish(producer, event)
                count += 1
                logger.info(
                    f"[{count}] {event.project_id} | {event.service_sku} | "
                    f"${event.cost_usd:.2f} | anomaly={event.is_anomaly}"
                )

    except KeyboardInterrupt:
        logger.info("Producer stopped by user.")
    finally:
        producer.close()
        logger.info("Kafka producer closed.")


if __name__ == "__main__":
    main()

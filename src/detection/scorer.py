"""
scorer.py
---------
Reads anomaly_scored_features from DuckDB, applies business-level
thresholds, and produces a ranked list of anomalies ready for
alerting and dashboard display.

Threshold tiers:
  - CRITICAL : anomaly_confidence >= 0.85 AND cost_usd > $5,000
  - HIGH     : anomaly_confidence >= 0.70 AND cost_usd > $1,000
  - MEDIUM   : anomaly_confidence >= 0.50
  - LOW      : everything else flagged

GCP Equivalent: Cloud Functions threshold evaluation before
                dispatching to Cloud Monitoring / PagerDuty

Usage:
    python src/detection/scorer.py
    python src/detection/scorer.py --top 20
    python src/detection/scorer.py --severity CRITICAL
"""

import argparse
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("finops.scorer")

DB_PATH = Path("data/finops.duckdb")

# Cost thresholds per severity tier
THRESHOLDS = {
    "CRITICAL": {"min_confidence": 0.85, "min_cost": 5_000},
    "HIGH":     {"min_confidence": 0.70, "min_cost": 1_000},
    "MEDIUM":   {"min_confidence": 0.50, "min_cost": 0},
    "LOW":      {"min_confidence": 0.00, "min_cost": 0},
}


def assign_severity(row: pd.Series) -> str:
    """Assign severity tier based on confidence score and cost."""
    conf = row["anomaly_confidence"]
    cost = row["cost_usd"]

    if conf >= 0.85 and cost > 5_000:
        return "CRITICAL"
    elif conf >= 0.70 and cost > 1_000:
        return "HIGH"
    elif conf >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def generate_alert_message(row: pd.Series) -> str:
    """Generate a human-readable alert message for each anomaly."""
    direction = "spike" if row["pct_change_1d"] > 0 else "drop"
    pct = abs(row["pct_change_1d"]) * 100

    return (
        f"[{row['severity']}] {row['project_id']} | {row['service_sku']} | "
        f"{row['event_date']} — "
        f"${row['cost_usd']:,.2f} spend detected "
        f"({direction} of {pct:.1f}% vs prior day). "
        f"Confidence: {row['anomaly_confidence']:.0%}."
    )


def generate_alert_id(row: pd.Series) -> str:
    """Build a stable alert ID so repeated scorer runs are idempotent."""
    key = "|".join(
        [
            str(row["project_id"]),
            str(row["service_sku"]),
            str(row["event_date"]),
            str(row["severity"]),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def score_and_rank(
    conn: duckdb.DuckDBPyConnection,
    severity_filter: str | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Pull flagged anomalies from DuckDB, assign severity tiers,
    rank by confidence × cost, and return scored DataFrame.
    """
    df = conn.execute("""
        SELECT
            project_id,
            service_sku,
            department,
            event_date,
            cost_usd,
            rolling_avg_30d,
            pct_change_1d,
            pct_change_7d,
            z_score_7d,
            z_score_30d,
            isolation_score,
            anomaly_confidence,
            is_flagged,
            has_injected_anomaly
        FROM anomaly_scored_features
        WHERE is_flagged = true
        ORDER BY anomaly_confidence DESC, cost_usd DESC
    """).fetchdf()

    if df.empty:
        logger.warning("No flagged anomalies found. Run anomaly_engine.py first.")
        return df

    # Assign severity
    df["severity"] = df.apply(assign_severity, axis=1)

    # Composite risk score: confidence weighted by normalized cost
    cost_max = df["cost_usd"].max()
    df["risk_score"] = (
        0.6 * df["anomaly_confidence"] +
        0.4 * (df["cost_usd"] / cost_max if cost_max > 0 else 0)
    ).round(4)

    # Cost delta vs 30d average
    df["cost_delta_vs_avg"] = (df["cost_usd"] - df["rolling_avg_30d"]).round(2)
    df["cost_delta_pct"] = (
        (df["cost_usd"] - df["rolling_avg_30d"]) /
        df["rolling_avg_30d"].replace(0, 1) * 100
    ).round(1)

    # Alert message
    df["alert_message"] = df.apply(generate_alert_message, axis=1)

    # Severity filter
    if severity_filter:
        df = df[df["severity"] == severity_filter.upper()]

    # Rank
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based rank

    if top_n:
        df = df.head(top_n)

    logger.info(
        f"Scored {len(df):,} flagged anomalies | "
        f"CRITICAL: {(df['severity'] == 'CRITICAL').sum()} | "
        f"HIGH: {(df['severity'] == 'HIGH').sum()} | "
        f"MEDIUM: {(df['severity'] == 'MEDIUM').sum()}"
    )

    return df


def write_alerts_to_db(
    conn: duckdb.DuckDBPyConnection,
    scored_df: pd.DataFrame,
) -> int:
    """
    Write HIGH and CRITICAL anomalies to alert_history table.
    Only writes alerts not already present (idempotent).
    Returns number of new alerts written.
    """
    alert_df = scored_df[scored_df["severity"].isin(["CRITICAL", "HIGH"])].copy()

    if alert_df.empty:
        logger.info("No HIGH/CRITICAL alerts to write.")
        return 0

    alert_df["alert_id"] = alert_df.apply(generate_alert_id, axis=1)
    alert_df["triggered_at"] = datetime.now(UTC)
    alert_df["threshold_usd"] = alert_df["severity"].map({
        "CRITICAL": 5_000,
        "HIGH": 1_000,
    })
    alert_df["resolved"] = False
    alert_df["resolved_at"] = None

    for col in (
        "alert_id",
        "project_id",
        "service_sku",
        "severity",
        "alert_message",
    ):
        alert_df[col] = alert_df[col].astype("string").astype(object)
    alert_df["resolved"] = alert_df["resolved"].astype(bool)

    before = conn.execute("SELECT COUNT(*) FROM alert_history").fetchone()[0]
    conn.register("alerts_temp", alert_df[[
        "alert_id", "triggered_at", "project_id", "service_sku",
        "severity", "cost_usd", "threshold_usd", "alert_message",
        "resolved", "resolved_at",
    ]])

    conn.execute("""
        INSERT OR IGNORE INTO alert_history (
            alert_id, triggered_at, project_id, service_sku,
            alert_type, severity, cost_usd, threshold_usd,
            message, resolved, resolved_at
        )
        SELECT
            alert_id,
            triggered_at,
            project_id,
            service_sku,
            'ANOMALY_DETECTION'     AS alert_type,
            severity,
            cost_usd,
            threshold_usd,
            alert_message           AS message,
            resolved,
            resolved_at
        FROM alerts_temp
    """)

    after = conn.execute("SELECT COUNT(*) FROM alert_history").fetchone()[0]
    inserted = after - before
    logger.info(f"Written {inserted} new alerts to alert_history table.")
    return inserted


def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    if df.empty:
        print("No anomalies to display.")
        return

    cols = [
        "project_id", "service_sku", "event_date",
        "cost_usd", "cost_delta_pct", "anomaly_confidence",
        "severity", "risk_score",
    ]

    print("\n── Top Anomalies ──────────────────────────────────────────────────────")
    print(df[cols].to_string(index=True))
    print("───────────────────────────────────────────────────────────────────────\n")

    print("── Severity Breakdown ─────────────────────────────────────────────────")
    print(df["severity"].value_counts().to_string())
    print(f"\nTotal flagged : {len(df):,}")
    print(f"Total at risk : ${df['cost_usd'].sum():,.2f}")
    print("───────────────────────────────────────────────────────────────────────\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinOps Anomaly Scorer")
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Show top N anomalies by risk score (default: 50)",
    )
    parser.add_argument(
        "--severity",
        type=str,
        default=None,
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        help="Filter by severity tier",
    )
    parser.add_argument(
        "--write-alerts",
        action="store_true",
        default=True,
        help="Write HIGH/CRITICAL to alert_history table (default: True)",
    )
    args = parser.parse_args()

    conn = duckdb.connect(str(DB_PATH))

    try:
        scored_df = score_and_rank(
            conn,
            severity_filter=args.severity,
            top_n=args.top,
        )

        if not scored_df.empty:
            print_summary(scored_df)

            if args.write_alerts:
                n_alerts = write_alerts_to_db(conn, scored_df)
                logger.info(f"Alert pipeline complete. {n_alerts} alerts queued.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
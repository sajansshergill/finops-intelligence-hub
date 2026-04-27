"""
anomaly_engine.py
-----------------
Dual-method anomaly detection engine for cloud billing data.

Method 1 — Z-Score (SQL-based, interpretable):
    Fast, per-series statistical detection using 7d and 30d rolling windows.
    Direct equivalent of BigQuery ML's ARIMA_PLUS anomaly detection.

Method 2 — Isolation Forest (ML-based, multi-feature):
    sklearn IsolationForest on 16 engineered features per (project, SKU) series.
    Equivalent of BigQuery ML BOOSTED_TREE_CLASSIFIER for anomaly classification.

Both scores are combined into a single anomaly_confidence score (0–1).
Results are written to the anomaly_scores table in DuckDB.

GCP Equivalent:
    BQML: CREATE MODEL ... OPTIONS(model_type='ISOLATION_FOREST')
          SELECT * FROM ML.DETECT_ANOMALIES(MODEL `project.dataset.model`, ...)

Usage:
    python src/detection/anomaly_engine.py
    python src/detection/anomaly_engine.py --date 2025-06-01
    python src/detection/anomaly_engine.py --full-history
"""

import argparse
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from feature_engineering import build_feature_matrix, get_ml_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("finops.anomaly_engine")

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "finops.duckdb"

# Tunable thresholds
Z_SCORE_THRESHOLD = 2.0  # Flag if |z_score| exceeds this
IF_CONTAMINATION = 0.05  # Expected anomaly rate
CONFIDENCE_THRESHOLD = 0.58  # Tuned via threshold sweep: F1=0.398, Recall=0.509
# Business rationale: recall prioritized over precision
# a missed $50K spike > a false alarm in FinOps


# ---------------------------------------------------------------------------
# Z-Score detection
# ---------------------------------------------------------------------------


def compute_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Z-scores using pre-computed rolling stats from feature matrix.
    Uses the more sensitive 7d window as primary, 30d as secondary signal.

    Returns df with added columns:
        z_score_7d, z_score_30d, z_score_combined
    """
    # Combined Z-score: weight 7d more heavily (more sensitive to spikes)
    df = df.copy()
    df["z_score_combined"] = (
        0.6 * df["z_score_7d"].abs() + 0.4 * df["z_score_30d"].abs()
    )
    return df


# ---------------------------------------------------------------------------
# Isolation Forest detection
# ---------------------------------------------------------------------------


def run_isolation_forest(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Train and score Isolation Forest on the feature matrix.
    Trains a separate model per (project_id, service_sku) series for
    better precision — mirrors how BQML would partition models.

    Returns array of anomaly scores in [0, 1] where 1 = most anomalous.
    """
    scores = np.zeros(len(df))
    scaler = (
        RobustScaler()
    )  # RobustScaler handles cost outliers better than StandardScaler

    groups = df.groupby(["project_id", "service_sku"])
    n_groups = len(groups)
    logger.info(f"Training Isolation Forest on {n_groups} (project, SKU) series...")

    for i, ((project_id, sku), group) in enumerate(groups):
        if len(group) < 14:
            # Not enough data for meaningful detection — skip
            continue

        X = group[feature_cols].fillna(0).values
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=100,
            contamination=IF_CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled)

        # decision_function returns negative scores for anomalies
        # Convert to [0, 1] where 1 = most anomalous
        raw_scores = model.decision_function(X_scaled)
        normalized = 1 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-10
        )

        scores[group.index] = normalized

        if (i + 1) % 20 == 0:
            logger.info(f"  Scored {i + 1}/{n_groups} series...")

    logger.info("Isolation Forest scoring complete.")
    return scores


# ---------------------------------------------------------------------------
# Score combination
# ---------------------------------------------------------------------------


def combine_scores(
    df: pd.DataFrame,
    if_scores: np.ndarray,
) -> pd.DataFrame:
    """
    Combine Z-score and Isolation Forest scores into a single
    anomaly_confidence value in [0, 1].

    Weighting:
        - Z-score contributes 40% (interpretable, fast signal)
        - Isolation Forest contributes 60% (multi-feature ML signal)

    An event is flagged if anomaly_confidence >= CONFIDENCE_THRESHOLD.
    """
    df = df.copy()
    df["isolation_score"] = if_scores

    # Normalize Z-score using sigmoid — avoids global max compression
    # sigmoid(z) maps z=2 → 0.88, z=3 → 0.95, z=1 → 0.73
    df["z_score_normalized"] = (1 / (1 + np.exp(-0.8 * df["z_score_combined"]))).clip(
        0, 1
    )

    # Weighted combination — Z-score weighted higher for spike/drop detection
    df["anomaly_confidence"] = (
        0.6 * df["z_score_normalized"] + 0.4 * df["isolation_score"]
    ).clip(0, 1)

    df["is_flagged"] = df["anomaly_confidence"] >= CONFIDENCE_THRESHOLD

    return df


# ---------------------------------------------------------------------------
# Write results to DuckDB
# ---------------------------------------------------------------------------


def write_scores(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    scored_at: datetime,
) -> int:
    """
    Write anomaly scores to the anomaly_scores table.
    Maps back to individual billing events via (project_id, service_sku, date).
    Returns number of rows written.
    """
    scores_df = df.copy()
    score_rows = scores_df[
        [
            "project_id",
            "service_sku",
            "event_date",
            "z_score_7d",
            "z_score_30d",
            "z_score_combined",
            "isolation_score",
            "anomaly_confidence",
            "is_flagged",
            "has_injected_anomaly",
        ]
    ].copy()

    scores_df["scored_at"] = scored_at
    scores_df["z_score"] = scores_df["z_score_combined"]
    score_rows["scored_at"] = scored_at
    score_rows["z_score"] = score_rows["z_score_combined"]

    # DuckDB cannot register pandas columns with dtype "str" (pandas StringDtype).
    # Coerce identifiers to plain Python strings (object) and booleans to bool.
    for frame in (scores_df, score_rows):
        for col in ("project_id", "service_sku", "department"):
            if col in frame.columns:
                frame[col] = frame[col].astype("string").astype(object)
        for col in ("is_flagged", "has_injected_anomaly"):
            if col in frame.columns:
                frame[col] = frame[col].astype(bool)

    # Register temp tables for the compact score table and full dashboard features.
    conn.register("scores_temp", score_rows)
    conn.register("scored_features_temp", scores_df)

    conn.execute(
        """
        INSERT OR IGNORE INTO anomaly_scores (
            event_id,
            scored_at,
            z_score,
            isolation_score,
            anomaly_confidence,
            is_flagged
        )
        SELECT
            gen_random_uuid()::VARCHAR  AS event_id,
            s.scored_at                 AS scored_at,
            s.z_score                   AS z_score,
            s.isolation_score           AS isolation_score,
            s.anomaly_confidence        AS anomaly_confidence,
            s.is_flagged                AS is_flagged
        FROM scores_temp s
    """
    )

    # Also store full scored feature table for dashboard use
    conn.execute("DROP TABLE IF EXISTS anomaly_scored_features")
    conn.execute(
        """
        CREATE TABLE anomaly_scored_features AS
        SELECT * FROM scored_features_temp
    """
    )

    count = conn.execute("SELECT COUNT(*) FROM anomaly_scores").fetchone()[0]
    flagged = conn.execute(
        "SELECT COUNT(*) FROM anomaly_scored_features WHERE is_flagged = true"
    ).fetchone()[0]

    logger.info(f"Wrote scores to DuckDB | flagged={flagged:,} anomalies")
    return count


# ---------------------------------------------------------------------------
# Evaluation: precision / recall vs ground truth
# ---------------------------------------------------------------------------


def evaluate(df: pd.DataFrame) -> dict:
    """
    Compare detected anomalies vs injected ground truth labels.
    Returns precision, recall, F1.
    """
    y_true = df["has_injected_anomaly"].astype(int)
    y_pred = df["is_flagged"].astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "total_flagged": int(y_pred.sum()),
        "total_ground_truth": int(y_true.sum()),
    }

    logger.info(
        f"Evaluation → Precision: {precision:.2%} | "
        f"Recall: {recall:.2%} | F1: {f1:.4f} | "
        f"Flagged: {y_pred.sum():,} / GT: {y_true.sum():,}"
    )

    return metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_detection(
    conn: duckdb.DuckDBPyConnection,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Full anomaly detection pipeline:
      1. Build feature matrix
      2. Z-Score computation
      3. Isolation Forest scoring
      4. Score combination
      5. Write to DuckDB
      6. Evaluate vs ground truth

    Returns evaluation metrics dict.
    """
    scored_at = datetime.now(UTC)

    # Step 1: Features
    df = build_feature_matrix(conn, start_date=start_date, end_date=end_date)

    # Step 2: Z-Scores (already in feature matrix, just combine)
    df = compute_z_scores(df)

    # Step 3: Isolation Forest
    X, feature_cols = get_ml_features(df)
    if_scores = run_isolation_forest(df, feature_cols)

    # Step 4: Combine
    df = combine_scores(df, if_scores)

    # Step 5: Write
    write_scores(conn, df, scored_at)

    # Step 6: Evaluate
    metrics = evaluate(df)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FinOps Anomaly Detection Engine")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Score a specific date (YYYY-MM-DD). Default: yesterday.",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Score entire billing history (slow for large datasets).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Number of days to score (default: 90).",
    )
    args = parser.parse_args()

    conn = duckdb.connect(str(DB_PATH))

    try:
        if args.full_history:
            logger.info("Running full-history anomaly detection...")
            metrics = run_detection(conn)
        elif args.date:
            logger.info(f"Running anomaly detection for date: {args.date}")
            metrics = run_detection(
                conn,
                start_date=args.date,
                end_date=args.date,
            )
        else:
            end = datetime.now(UTC).date()
            start = end - timedelta(days=args.lookback_days)
            logger.info(
                f"Running anomaly detection: {start} → {end} "
                f"({args.lookback_days} days)"
            )
            metrics = run_detection(
                conn,
                start_date=str(start),
                end_date=str(end),
            )

        print("\n── Anomaly Detection Results ──────────────────")
        for k, v in metrics.items():
            print(f"  {k:<25} {v}")
        print("───────────────────────────────────────────────\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

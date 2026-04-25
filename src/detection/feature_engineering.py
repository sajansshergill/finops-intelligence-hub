"""
feature_engineering.py
-----------------------
Builds feature matrix for anomaly detection from raw billing_events in DuckDB.
Mirrors the feature transformations you would do inside BigQuery ML before
calling ML.DETECT_ANOMALIES().

Features produced per (project_id, service_sku, date):
  - cost_usd              : raw daily aggregated spend
  - rolling_avg_7d        : 7-day rolling average
  - rolling_avg_30d       : 30-day rolling average
  - rolling_std_7d        : 7-day rolling std dev
  - rolling_std_30d       : 30-day rolling std dev
  - lag_1d / lag_7d       : spend 1 and 7 days prior
  - pct_change_1d         : day-over-day % change
  - pct_change_7d         : week-over-week % change
  - z_score_7d            : Z-score vs 7-day window
  - z_score_30d           : Z-score vs 30-day window
  - is_month_end          : 1 if day >= 28
  - is_quarter_end        : 1 if month in [3,6,9,12] and day >= 25
  - day_of_week           : 0=Mon … 6=Sun
  - is_weekend            : 1 if Sat/Sun
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger("finops.features")

DB_PATH = Path("data/finops.duckdb")


# ---------------------------------------------------------------------------
# SQL-based feature extraction (DuckDB / BigQuery equivalent)
# ---------------------------------------------------------------------------

FEATURE_SQL = """
WITH daily AS (
    -- Aggregate to daily grain per project/SKU
    SELECT
        project_id,
        service_sku,
        department,
        CAST(timestamp AS DATE)                 AS event_date,
        SUM(cost_usd)                           AS cost_usd,
        SUM(usage_amount)                       AS usage_amount,
        COUNT(*)                                AS event_count,
        BOOL_OR(is_anomaly)                     AS has_injected_anomaly
    FROM billing_events
    GROUP BY 1, 2, 3, 4
),

windowed AS (
    SELECT
        *,
        -- Rolling averages
        AVG(cost_usd) OVER w7                   AS rolling_avg_7d,
        AVG(cost_usd) OVER w30                  AS rolling_avg_30d,

        -- Rolling std devs
        STDDEV_SAMP(cost_usd) OVER w7           AS rolling_std_7d,
        STDDEV_SAMP(cost_usd) OVER w30          AS rolling_std_30d,

        -- Lag features
        LAG(cost_usd, 1)  OVER wpart            AS lag_1d,
        LAG(cost_usd, 7)  OVER wpart            AS lag_7d,

        -- Rolling min/max for range context
        MIN(cost_usd) OVER w30                  AS rolling_min_30d,
        MAX(cost_usd) OVER w30                  AS rolling_max_30d

    FROM daily
    WINDOW
        wpart AS (
            PARTITION BY project_id, service_sku
            ORDER BY event_date
        ),
        w7 AS (
            PARTITION BY project_id, service_sku
            ORDER BY event_date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ),
        w30 AS (
            PARTITION BY project_id, service_sku
            ORDER BY event_date
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        )
),

features AS (
    SELECT
        project_id,
        service_sku,
        department,
        event_date,
        cost_usd,
        usage_amount,
        event_count,
        has_injected_anomaly,

        -- Rolling stats
        COALESCE(rolling_avg_7d, cost_usd)      AS rolling_avg_7d,
        COALESCE(rolling_avg_30d, cost_usd)     AS rolling_avg_30d,
        COALESCE(rolling_std_7d, 0)             AS rolling_std_7d,
        COALESCE(rolling_std_30d, 0)            AS rolling_std_30d,
        COALESCE(rolling_min_30d, cost_usd)     AS rolling_min_30d,
        COALESCE(rolling_max_30d, cost_usd)     AS rolling_max_30d,

        -- Lag features
        COALESCE(lag_1d, cost_usd)              AS lag_1d,
        COALESCE(lag_7d, cost_usd)              AS lag_7d,

        -- % change features
        CASE
            WHEN COALESCE(lag_1d, 0) = 0 THEN 0
            ELSE (cost_usd - lag_1d) / lag_1d
        END                                     AS pct_change_1d,

        CASE
            WHEN COALESCE(lag_7d, 0) = 0 THEN 0
            ELSE (cost_usd - lag_7d) / lag_7d
        END                                     AS pct_change_7d,

        -- Z-scores (GCP BQML ARIMA anomaly detection equivalent)
        CASE
            WHEN COALESCE(rolling_std_7d, 0) = 0 THEN 0
            ELSE (cost_usd - rolling_avg_7d) / rolling_std_7d
        END                                     AS z_score_7d,

        CASE
            WHEN COALESCE(rolling_std_30d, 0) = 0 THEN 0
            ELSE (cost_usd - rolling_avg_30d) / rolling_std_30d
        END                                     AS z_score_30d,

        -- Calendar features
        CASE WHEN DAY(event_date) >= 28 THEN 1 ELSE 0 END          AS is_month_end,
        CASE WHEN MONTH(event_date) IN (3,6,9,12)
              AND DAY(event_date) >= 25 THEN 1 ELSE 0 END           AS is_quarter_end,
        DAYOFWEEK(event_date) - 1                                   AS day_of_week,
        CASE WHEN DAYOFWEEK(event_date) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend

    FROM windowed
)

SELECT * FROM features
ORDER BY project_id, service_sku, event_date
"""


def build_feature_matrix(
    conn: duckdb.DuckDBPyConnection,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix from billing_events.

    Args:
        conn: Active DuckDB connection
        start_date: Optional ISO date string to filter from (e.g. '2024-01-01')
        end_date:   Optional ISO date string to filter to   (e.g. '2024-12-31')

    Returns:
        DataFrame with one row per (project_id, service_sku, event_date)
    """
    sql = FEATURE_SQL

    # Inject date filters if provided
    if start_date or end_date:
        filters = []
        if start_date:
            filters.append(f"CAST(timestamp AS DATE) >= '{start_date}'")
        if end_date:
            filters.append(f"CAST(timestamp AS DATE) <= '{end_date}'")
        where_clause = " AND ".join(filters)
        # Insert WHERE before GROUP BY in the daily CTE
        sql = sql.replace(
            "GROUP BY 1, 2, 3, 4",
            f"WHERE {where_clause}\n    GROUP BY 1, 2, 3, 4",
        )

    logger.info("Building feature matrix from DuckDB...")
    df = conn.execute(sql).fetchdf()

    # Drop rows with insufficient history (first 7 days per series)
    df = df.dropna(subset=["rolling_avg_7d", "lag_1d"])

    logger.info(
        f"Feature matrix: {len(df):,} rows | "
        f"{df['project_id'].nunique()} projects | "
        f"{df['service_sku'].nunique()} SKUs | "
        f"date range: {df['event_date'].min()} → {df['event_date'].max()}"
    )

    return df


def get_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract the numeric feature columns used by Isolation Forest.
    Returns (feature_df, feature_names).
    """
    feature_cols = [
        "cost_usd",
        "rolling_avg_7d",
        "rolling_avg_30d",
        "rolling_std_7d",
        "rolling_std_30d",
        "lag_1d",
        "lag_7d",
        "pct_change_1d",
        "pct_change_7d",
        "z_score_7d",
        "z_score_30d",
        "is_month_end",
        "is_quarter_end",
        "day_of_week",
        "is_weekend",
        "event_count",
    ]

    X = df[feature_cols].copy()

    # Clip extreme pct_change values to reduce noise
    X["pct_change_1d"] = X["pct_change_1d"].clip(-10, 10)
    X["pct_change_7d"] = X["pct_change_7d"].clip(-10, 10)

    # Fill any remaining NaNs
    X = X.fillna(0)

    return X, feature_cols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conn = duckdb.connect(str(DB_PATH))
    df = build_feature_matrix(conn)
    print(df.describe())
    print(f"\nFeature columns: {df.columns.tolist()}")
    conn.close()
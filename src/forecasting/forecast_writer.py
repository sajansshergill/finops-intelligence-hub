"""
forecast_writer.py
------------------
Writes ARIMA forecast results to the forecasts table in DuckDB.
Handles upserts cleanly so re-runs don't duplicate data.

GCP Equivalent: Writing BQML forecast results back to BigQuery
                via CREATE OR REPLACE TABLE ... AS SELECT * FROM ML.FORECAST(...)

Usage:
    Called automatically by arima_model.py
    Can also be imported and called directly.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger("finops.forecast_writer")

DB_PATH = Path("data/finops.duckdb")


def write_forecasts(
    conn: duckdb.DuckDBPyConnection,
    results: list[dict],
) -> int:
    """
    Write forecast results to DuckDB forecasts table.

    Args:
        conn:    Active DuckDB connection
        results: List of result dicts from arima_model.run_forecasting()

    Returns:
        Total number of forecast rows written
    """
    if not results:
        logger.warning("No forecast results to write.")
        return 0

    # Combine all forecast DataFrames
    all_forecasts = []

    for r in results:
        df = r["forecast_df"].copy()
        df["model_run_id"] = r["run_id"]
        df["mape"] = r["metrics"]["mape"]
        df["rmse"] = r["metrics"]["rmse"]
        df["created_at"] = datetime.now(UTC)
        all_forecasts.append(df)

    combined = pd.concat(all_forecasts, ignore_index=True)

    # Ensure correct dtypes
    combined["forecast_date"] = pd.to_datetime(combined["forecast_date"]).dt.date
    combined["predicted_cost"] = combined["predicted_cost"].round(4)
    combined["lower_80"] = combined["lower_80"].round(4)
    combined["upper_80"] = combined["upper_80"].round(4)
    combined["lower_95"] = combined["lower_95"].round(4)
    combined["upper_95"] = combined["upper_95"].round(4)
    for col in ("project_id", "service_sku", "model_run_id"):
        combined[col] = combined[col].astype("string").astype(object)

    # Register and insert
    conn.register("forecasts_temp", combined)

    conn.execute(
        """
        INSERT OR IGNORE INTO forecasts (
            project_id,
            service_sku,
            forecast_date,
            predicted_cost,
            lower_80,
            upper_80,
            lower_95,
            upper_95,
            model_run_id,
            created_at
        )
        SELECT
            project_id,
            service_sku,
            forecast_date,
            predicted_cost,
            lower_80,
            upper_80,
            lower_95,
            upper_95,
            model_run_id,
            created_at
        FROM forecasts_temp
    """
    )

    total = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    written = len(combined)

    logger.info(
        f"Wrote {written:,} forecast rows to DuckDB " f"(total in table: {total:,})"
    )

    # Also save to Parquet for audit trail
    out_path = (
        Path("data/forecasts")
        / f"forecast_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info(f"Forecast snapshot saved to {out_path}")

    return written


def load_latest_forecasts(
    conn: duckdb.DuckDBPyConnection,
    project_id: str | None = None,
    service_sku: str | None = None,
) -> pd.DataFrame:
    """
    Load the most recent forecast per (project_id, service_sku, forecast_date).
    Used by the Streamlit dashboard.
    """
    filters = []
    if project_id:
        filters.append(f"project_id = '{project_id}'")
    if service_sku:
        filters.append(f"service_sku = '{service_sku}'")

    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    df = conn.execute(
        f"""
        WITH ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY project_id, service_sku, forecast_date
                    ORDER BY created_at DESC
                ) AS rn
            FROM forecasts
            {where}
        )
        SELECT
            project_id,
            service_sku,
            forecast_date,
            predicted_cost,
            lower_80,
            upper_80,
            lower_95,
            upper_95,
            model_run_id,
            created_at
        FROM ranked
        WHERE rn = 1
        ORDER BY project_id, service_sku, forecast_date
    """
    ).fetchdf()

    logger.info(
        f"Loaded {len(df):,} forecast rows | "
        f"{df['project_id'].nunique() if not df.empty else 0} projects | "
        f"{df['service_sku'].nunique() if not df.empty else 0} SKUs"
    )

    return df


def forecast_summary(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Aggregated 30-day forecast summary per project.
    Used for dashboard KPI cards.
    """
    return conn.execute(
        """
        WITH latest AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY project_id, service_sku, forecast_date
                    ORDER BY created_at DESC
                ) AS rn
            FROM forecasts
        )
        SELECT
            project_id,
            SUM(predicted_cost)     AS total_forecast_30d,
            SUM(lower_95)           AS total_lower_95,
            SUM(upper_95)           AS total_upper_95,
            COUNT(DISTINCT service_sku) AS n_skus,
            MIN(forecast_date)      AS forecast_from,
            MAX(forecast_date)      AS forecast_to,
            MAX(created_at)         AS last_updated
        FROM latest
        WHERE rn = 1
        GROUP BY project_id
        ORDER BY total_forecast_30d DESC
    """
    ).fetchdf()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    conn = duckdb.connect(str(DB_PATH))
    try:
        summary = forecast_summary(conn)
        if summary.empty:
            print("No forecasts found. Run arima_model.py first.")
        else:
            print("\n── 30-Day Forecast Summary by Project ─────────────────────────")
            print(summary.to_string(index=False))
            print("────────────────────────────────────────────────────────────────\n")
    finally:
        conn.close()

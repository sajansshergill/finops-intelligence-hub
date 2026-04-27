"""
model_registry.py
-----------------
MLflow experiment management utilities for the FinOps forecasting pipeline.

GCP Equivalent: Vertex AI Experiments + Vertex AI Model Registry

Provides:
  - List all forecast runs with metrics
  - Compare runs across experiments
  - Promote best model per (project, SKU)
  - Clean up stale runs

Usage:
    python src/forecasting/model_registry.py
    python src/forecasting/model_registry.py --compare
    python src/forecasting/model_registry.py --best
"""

import argparse
import logging

import mlflow
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("finops.model_registry")

MLFLOW_URI = "models/mlruns"
EXPERIMENT = "finops-spend-forecast"


def setup() -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.MlflowClient()


def list_runs(client: mlflow.MlflowClient) -> pd.DataFrame:
    """Return all runs for the forecast experiment as a DataFrame."""
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if not experiment:
        print(f"Experiment '{EXPERIMENT}' not found. Run arima_model.py first.")
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mape ASC"],
    )

    rows = []
    for run in runs:
        rows.append(
            {
                "run_id": run.info.run_id[:8],
                "run_name": run.info.run_name,
                "status": run.info.status,
                "mape": run.data.metrics.get("mape"),
                "rmse": run.data.metrics.get("rmse"),
                "aic": run.data.params.get("aic"),
                "p": run.data.params.get("arima_p"),
                "d": run.data.params.get("arima_d"),
                "q": run.data.params.get("arima_q"),
                "project_id": run.data.params.get("project_id"),
                "service_sku": run.data.params.get("service_sku"),
                "n_obs": run.data.params.get("n_obs"),
                "30d_forecast": run.data.metrics.get("total_forecast_spend"),
            }
        )

    return pd.DataFrame(rows)


def best_runs(df: pd.DataFrame, metric: str = "mape") -> pd.DataFrame:
    """Return the best run per (project_id, service_sku) by MAPE."""
    if df.empty:
        return df
    return (
        df.sort_values(metric)
        .groupby(["project_id", "service_sku"])
        .first()
        .reset_index()
    )


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

    print("\n── All Forecast Runs ──────────────────────────────────────────────────")
    cols = ["run_name", "mape", "rmse", "p", "d", "q", "30d_forecast", "status"]
    print(df[cols].to_string(index=False))

    print("\n── Aggregate Metrics ──────────────────────────────────────────────────")
    print(f"  Total runs        : {len(df)}")
    print(f"  Successful runs   : {(df['status'] == 'FINISHED').sum()}")
    print(f"  Avg MAPE          : {df['mape'].mean():.2f}%")
    print(f"  Median MAPE       : {df['mape'].median():.2f}%")
    print(f"  Best MAPE         : {df['mape'].min():.2f}%")
    print(f"  Total 30d forecast: ${df['30d_forecast'].sum():,.2f}")
    print("───────────────────────────────────────────────────────────────────────\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinOps MLflow Model Registry")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show all runs with metrics comparison",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Show best run per (project, SKU) by MAPE",
    )
    args = parser.parse_args()

    client = setup()
    df = list_runs(client)

    if df.empty:
        return

    if args.best:
        best = best_runs(df)
        print("\n── Best Model Per Series (lowest MAPE) ────────────────────────────")
        print(
            best[
                ["project_id", "service_sku", "mape", "rmse", "p", "d", "q"]
            ].to_string(index=False)
        )
        print("───────────────────────────────────────────────────────────────────\n")
    else:
        print_summary(df)


if __name__ == "__main__":
    main()

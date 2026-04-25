"""
arima_model.py
--------------
30-day ahead spend forecasting per (project_id, service_sku) series.
Uses auto-ARIMA for order selection and statsmodels for fitting.

GCP Equivalent:
    BigQuery ML: CREATE MODEL ... OPTIONS(model_type='ARIMA_PLUS')
    Vertex AI Experiments: MLflow tracking → Vertex AI Experiments

Each series gets its own model. Results include:
  - Point forecast (30 days)
  - 80% confidence interval
  - 95% confidence interval
  - Per-run MLflow tracking (MAPE, RMSE, model params)

Usage:
    python src/forecasting/arima_model.py
    python src/forecasting/arima_model.py --project risk-analytics-prod
    python src/forecasting/arima_model.py --top-skus 5
"""

import argparse
import logging
import warnings
from datetime import timedelta
from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

try:
    from pmdarima import auto_arima
except Exception:  # pragma: no cover - depends on local environment
    auto_arima = None

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
except Exception as exc:  # pragma: no cover - depends on local environment
    ARIMA = None
    adfuller = None
    STATSMODELS_IMPORT_ERROR = exc
else:
    STATSMODELS_IMPORT_ERROR = None

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("finops.forecasting")

DB_PATH      = Path("data/finops.duckdb")
MLFLOW_URI   = "models/mlruns"
EXPERIMENT   = "finops-spend-forecast"
HORIZON      = 30      # forecast days ahead
MIN_HISTORY  = 60      # minimum days of history required per series


class SimpleForecastResult:
    """Small adapter matching the statsmodels forecast result methods we use."""

    def __init__(self, predicted_mean: pd.Series, residual_std: float):
        self.predicted_mean = predicted_mean
        self.residual_std = residual_std

    def conf_int(self, alpha: float) -> pd.DataFrame:
        z_score = 1.2816 if alpha == 0.20 else 1.96
        delta = z_score * self.residual_std
        return pd.DataFrame({
            "lower": self.predicted_mean - delta,
            "upper": self.predicted_mean + delta,
        })


class SimpleForecastModel:
    """Fallback forecaster used when ARIMA dependencies are unavailable."""

    order = (0, 0, 0)

    def __init__(self, series_log: pd.Series):
        self.series_log = series_log
        self.level = float(series_log.tail(7).mean())
        residuals = series_log.diff().dropna()
        self.residual_std = float(residuals.std()) if not residuals.empty else 0.0

    def get_forecast(self, steps: int) -> SimpleForecastResult:
        index = pd.RangeIndex(steps)
        predicted = pd.Series([self.level] * steps, index=index)
        return SimpleForecastResult(predicted, self.residual_std)

    @staticmethod
    def forecast_from(series_log: pd.Series, steps: int, index: pd.Index) -> pd.Series:
        level = float(series_log.tail(7).mean())
        return pd.Series([level] * steps, index=index)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_series(
    conn: duckdb.DuckDBPyConnection,
    project_id: str | None = None,
    top_skus: int | None = None,
) -> dict[tuple, pd.Series]:
    """
    Load daily aggregated spend per (project_id, service_sku) from DuckDB.
    Returns dict of {(project_id, sku): pd.Series indexed by date}.
    """
    where = ""
    if project_id:
        where = f"WHERE project_id = '{project_id}'"

    df = conn.execute(f"""
        SELECT
            project_id,
            service_sku,
            CAST(timestamp AS DATE) AS event_date,
            SUM(cost_usd)       AS cost_usd
        FROM billing_events
        {where}
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """).fetchdf()

    # Optionally limit to highest-spend SKUs
    if top_skus:
        top = (
            df.groupby("service_sku")["cost_usd"]
            .sum()
            .nlargest(top_skus)
            .index.tolist()
        )
        df = df[df["service_sku"].isin(top)]

    series_dict = {}
    for (proj, sku), group in df.groupby(["project_id", "service_sku"]):
        s = group.set_index("event_date")["cost_usd"].sort_index()
        s.index = pd.to_datetime(s.index)
        s = s.asfreq("D").ffill().fillna(0)
        if len(s) >= MIN_HISTORY:
            series_dict[(proj, sku)] = s

    logger.info(
        f"Loaded {len(series_dict)} series with >= {MIN_HISTORY} days of history"
    )
    return series_dict


# ---------------------------------------------------------------------------
# Stationarity check
# ---------------------------------------------------------------------------

def is_stationary(series: pd.Series, significance: float = 0.05) -> bool:
    """Augmented Dickey-Fuller test for stationarity."""
    if adfuller is None:
        return False

    try:
        result = adfuller(series.dropna(), autolag="AIC")
        return result[1] < significance
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ARIMA fitting
# ---------------------------------------------------------------------------

def fit_arima(
    series: pd.Series,
    project_id: str,
    sku: str,
) -> tuple[object, pd.Series, dict]:
    """
    Fit an ARIMA model to a single spend series.

    Uses auto_arima for order selection (equivalent to BigQuery ML
    ARIMA_PLUS auto-order detection), then fits with statsmodels
    for richer confidence interval support.

    Returns (fitted_model, params_dict).
    """
    # Log-transform to stabilize variance (common for cost data)
    series_log = np.log1p(series)

    order, aic = select_arima_order(series_log)
    params = {
        "project_id": project_id,
        "service_sku": sku,
        "arima_p": order[0],
        "arima_d": order[1],
        "arima_q": order[2],
        "aic": round(aic, 4),
        "n_obs": len(series),
        "log_transformed": True,
        "forecast_method": "arima" if ARIMA is not None else "simple_baseline",
    }

    if ARIMA is None:
        logger.warning(
            "statsmodels ARIMA unavailable; using simple baseline forecast. "
            f"Import error: {STATSMODELS_IMPORT_ERROR}"
        )
        return SimpleForecastModel(series_log), series_log, params

    # Refit with statsmodels for confidence interval support
    model = ARIMA(series_log, order=order)
    fitted = model.fit()

    return fitted, series_log, params


def select_arima_order(series_log: pd.Series) -> tuple[tuple[int, int, int], float]:
    """Select an ARIMA order, falling back to a small AIC grid if pmdarima is absent."""
    if ARIMA is None:
        return (0, 0, 0), 0.0

    if auto_arima is not None:
        auto = auto_arima(
            series_log,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,                  # auto-detect differencing
            seasonal=False,          # daily data, no strong weekly seasonality in log space
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        return auto.order, float(auto.aic())

    d_values = [0] if is_stationary(series_log) else [1]
    candidates = [
        (p, d, q)
        for d in d_values
        for p in range(3)
        for q in range(3)
        if (p, d, q) != (0, 0, 0)
    ]

    best_order: tuple[int, int, int] | None = None
    best_aic = float("inf")
    for order in candidates:
        try:
            result = ARIMA(series_log, order=order).fit()
        except Exception:
            continue
        if result.aic < best_aic:
            best_order = order
            best_aic = float(result.aic)

    if best_order is None:
        best_order = (1, 1, 1)
    return best_order, best_aic


# ---------------------------------------------------------------------------
# Forecast generation
# ---------------------------------------------------------------------------

def generate_forecast(
    fitted_model,
    series_log: pd.Series,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Generate point forecast + confidence intervals.
    Returns DataFrame with columns:
        forecast_date, predicted_cost,
        lower_80, upper_80, lower_95, upper_95
    """
    forecast_result = fitted_model.get_forecast(steps=horizon)
    forecast_mean   = forecast_result.predicted_mean
    conf_80         = forecast_result.conf_int(alpha=0.20)
    conf_95         = forecast_result.conf_int(alpha=0.05)

    last_date = series_log.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    # Inverse log transform
    df = pd.DataFrame({
        "forecast_date":  forecast_dates,
        "predicted_cost": np.expm1(forecast_mean.values),
        "lower_80":       np.expm1(conf_80.iloc[:, 0].values),
        "upper_80":       np.expm1(conf_80.iloc[:, 1].values),
        "lower_95":       np.expm1(conf_95.iloc[:, 0].values),
        "upper_95":       np.expm1(conf_95.iloc[:, 1].values),
    })

    # Clip negative lower bounds (cost can't be negative)
    df["lower_80"] = df["lower_80"].clip(lower=0)
    df["lower_95"] = df["lower_95"].clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# Evaluation metrics (in-sample)
# ---------------------------------------------------------------------------

def compute_metrics(
    fitted_model,
    series_log: pd.Series,
    series_orig: pd.Series,
) -> dict:
    """
    Compute in-sample MAPE and RMSE on original scale.
    Uses last 30 days as holdout for evaluation.
    """
    holdout_n = min(30, len(series_orig) // 5)

    train_log  = series_log.iloc[:-holdout_n]
    actual     = series_orig.iloc[-holdout_n:]

    if ARIMA is None or isinstance(fitted_model, SimpleForecastModel):
        preds_log = SimpleForecastModel.forecast_from(
            train_log,
            holdout_n,
            actual.index,
        )
    else:
        # Refit on train only
        model = ARIMA(train_log, order=fitted_model.model.order)
        fitted_train = model.fit()
        preds_log = fitted_train.forecast(steps=holdout_n)
    preds     = np.expm1(preds_log)

    # MAPE
    mask  = actual > 0
    mape  = np.mean(np.abs((actual[mask] - preds[mask]) / actual[mask])) * 100

    # RMSE
    rmse  = np.sqrt(np.mean((actual.values - preds.values) ** 2))

    return {
        "mape":       round(float(mape), 4),
        "rmse":       round(float(rmse), 4),
        "holdout_n":  holdout_n,
    }


# ---------------------------------------------------------------------------
# MLflow tracking
# ---------------------------------------------------------------------------

def setup_mlflow() -> str:
    """Initialize MLflow experiment. Returns run-level experiment ID."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    return EXPERIMENT


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_forecasting(
    conn: duckdb.DuckDBPyConnection,
    project_id: str | None = None,
    top_skus: int | None = None,
    horizon: int = HORIZON,
) -> list[dict]:
    """
    Full forecasting pipeline:
      1. Load daily spend series from DuckDB
      2. Fit ARIMA per series
      3. Generate 30-day forecast with CIs
      4. Evaluate on holdout
      5. Track with MLflow
      6. Return list of forecast DataFrames for writing

    Returns list of result dicts with keys:
        project_id, service_sku, forecast_df, params, metrics, run_id
    """
    setup_mlflow()
    series_dict = load_series(conn, project_id=project_id, top_skus=top_skus)

    results = []
    n = len(series_dict)

    for i, ((proj, sku), series) in enumerate(series_dict.items()):
        logger.info(f"[{i+1}/{n}] Fitting ARIMA: {proj} | {sku}")

        try:
            with mlflow.start_run(run_name=f"{proj}__{sku.replace(' ', '_')}"):
                # Fit
                fitted, series_log, params = fit_arima(series, proj, sku)

                # Forecast
                forecast_df = generate_forecast(fitted, series_log, horizon=horizon)
                forecast_df["project_id"] = proj
                forecast_df["service_sku"] = sku

                # Evaluate
                metrics = compute_metrics(fitted, series_log, series)

                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_metric("forecast_horizon_days", horizon)
                mlflow.log_metric(
                    "total_forecast_spend",
                    round(float(forecast_df["predicted_cost"].sum()), 2),
                )

                run_id = mlflow.active_run().info.run_id

                results.append({
                    "project_id":  proj,
                    "service_sku": sku,
                    "forecast_df": forecast_df,
                    "params":      params,
                    "metrics":     metrics,
                    "run_id":      run_id,
                })

                logger.info(
                    f"  ARIMA{params['arima_p'],params['arima_d'],params['arima_q']} | "
                    f"MAPE={metrics['mape']:.2f}% | RMSE={metrics['rmse']:,.2f} | "
                    f"30d forecast=${forecast_df['predicted_cost'].sum():,.2f}"
                )

        except Exception as e:
            logger.warning(f"  Failed {proj} | {sku}: {e}")
            continue

    logger.info(
        f"Forecasting complete. {len(results)}/{n} series fitted successfully."
    )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FinOps ARIMA Forecasting Engine")
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Filter to a specific project_id (default: all projects)",
    )
    parser.add_argument(
        "--top-skus",
        type=int,
        default=None,
        help="Limit to top N SKUs by total spend (default: all SKUs)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=HORIZON,
        help=f"Forecast horizon in days (default: {HORIZON})",
    )
    args = parser.parse_args()

    conn = duckdb.connect(str(DB_PATH))

    try:
        results = run_forecasting(
            conn,
            project_id=args.project,
            top_skus=args.top_skus,
            horizon=args.horizon,
        )

        if results:
            # Quick summary
            print("\n── Forecast Summary ───────────────────────────────────────────")
            for r in results[:10]:  # show first 10
                fc = r["forecast_df"]["predicted_cost"].sum()
                print(
                    f"  {r['project_id']:<30} {r['service_sku']:<35} "
                    f"30d=${fc:>12,.2f}  MAPE={r['metrics']['mape']:.1f}%"
                )
            if len(results) > 10:
                print(f"  ... and {len(results)-10} more series")
            print("───────────────────────────────────────────────────────────────\n")

            # Write forecasts to DuckDB
            from forecast_writer import write_forecasts
            write_forecasts(conn, results)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
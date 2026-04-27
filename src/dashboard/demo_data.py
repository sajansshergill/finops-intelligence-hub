"""Deterministic demo data for Streamlit Cloud deployments.

The production dashboard reads from DuckDB when `data/finops.duckdb` exists.
Streamlit Community Cloud does not run the local ingestion pipeline, so these
helpers keep every page populated with representative FinOps data.
"""

from __future__ import annotations

import math

import pandas as pd

PROJECTS = [
    ("risk-analytics-prod", "Quantitative Research", "production"),
    ("payments-platform-prod", "Payments", "production"),
    ("customer-360-prod", "Data Platform", "production"),
    ("fraud-detection-prod", "Security", "production"),
]

SKUS = [
    ("BigQuery: Slot Usage", 4200),
    ("Dataflow: Streaming Workers", 2600),
    ("Cloud Storage: Standard", 1300),
    ("Vertex AI: Training", 3300),
]

REGIONS = ["us-east1", "us-central1", "us-west1", "northamerica-northeast1"]


def build_daily_spend() -> pd.DataFrame:
    start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=179)
    rows = []

    for day_index in range(180):
        event_date = start_date + pd.Timedelta(days=day_index)
        is_month_end = event_date.is_month_end
        weekday_factor = 0.86 if event_date.weekday() >= 5 else 1.0
        growth_factor = 1 + day_index / 900

        for project_index, (project_id, department, env) in enumerate(PROJECTS):
            for sku_index, (service_sku, base_cost) in enumerate(SKUS):
                seasonal = 1 + 0.18 * math.sin((day_index + project_index * 7) / 9)
                month_end_factor = 1.45 if is_month_end and sku_index in {0, 1} else 1.0
                anomaly_factor = (
                    2.25
                    if (day_index + project_index * 11 + sku_index * 5) % 47 == 0
                    else 1.0
                )
                cost = (
                    base_cost
                    * (0.74 + project_index * 0.08 + sku_index * 0.05)
                    * seasonal
                    * weekday_factor
                    * month_end_factor
                    * anomaly_factor
                    * growth_factor
                )

                rows.append(
                    {
                        "event_date": event_date,
                        "project_id": project_id,
                        "service_sku": service_sku,
                        "department": department,
                        "region": REGIONS[(project_index + sku_index) % len(REGIONS)],
                        "label_env": env,
                        "cost_usd": round(cost, 2),
                        "event_count": int(80 + cost / 115 + day_index % 9),
                    }
                )

    return pd.DataFrame(rows)


def build_anomalies() -> pd.DataFrame:
    df = build_daily_spend().sort_values(["project_id", "service_sku", "event_date"])
    grouped = df.groupby(["project_id", "service_sku"])["cost_usd"]

    rolling_avg = grouped.transform(
        lambda series: series.shift(1).rolling(30, min_periods=7).mean()
    )
    rolling_std = grouped.transform(
        lambda series: series.shift(1).rolling(30, min_periods=7).std()
    )
    pct_change = grouped.pct_change().fillna(0)

    df["rolling_avg_30d"] = rolling_avg.fillna(grouped.transform("mean"))
    rolling_std = rolling_std.fillna(grouped.transform("std")).replace(0, 1)
    df["pct_change_1d"] = pct_change
    df["z_score_7d"] = (df["cost_usd"] - df["rolling_avg_30d"]) / rolling_std
    df["z_score_30d"] = df["z_score_7d"] * 0.82
    df["isolation_score"] = (df["z_score_7d"].abs() / 6).clip(0, 1)
    df["anomaly_confidence"] = (
        0.32 + df["isolation_score"] * 0.58 + (df["pct_change_1d"].abs() * 0.18)
    ).clip(0, 0.98)
    df["has_injected_anomaly"] = df.index % 47 == 0
    df.loc[df["has_injected_anomaly"], "anomaly_confidence"] = df.loc[
        df["has_injected_anomaly"], "anomaly_confidence"
    ].clip(lower=0.72)
    df["is_flagged"] = df["anomaly_confidence"] >= 0.58

    return df[
        [
            "project_id",
            "service_sku",
            "department",
            "event_date",
            "cost_usd",
            "rolling_avg_30d",
            "pct_change_1d",
            "z_score_7d",
            "z_score_30d",
            "isolation_score",
            "anomaly_confidence",
            "is_flagged",
            "has_injected_anomaly",
        ]
    ].sort_values("anomaly_confidence", ascending=False)


def build_forecast_actuals() -> pd.DataFrame:
    daily = build_daily_spend()
    return (
        daily.groupby(["event_date", "project_id", "service_sku"], as_index=False)[
            "cost_usd"
        ]
        .sum()
        .sort_values("event_date")
    )


def build_forecasts() -> pd.DataFrame:
    actuals = build_forecast_actuals()
    last_date = actuals["event_date"].max()
    created_at = pd.Timestamp.now().floor("s")
    rows = []

    for (project_id, service_sku), group in actuals.groupby(
        ["project_id", "service_sku"]
    ):
        baseline = group.tail(30)["cost_usd"].mean()
        trend = (
            group.tail(7)["cost_usd"].mean() - group.tail(30)["cost_usd"].mean()
        ) / 30

        for horizon in range(1, 31):
            forecast_date = last_date + pd.Timedelta(days=horizon)
            weekly = 1 + 0.08 * math.sin(horizon / 3.5)
            predicted = max(0, baseline + trend * horizon) * weekly
            rows.append(
                {
                    "project_id": project_id,
                    "service_sku": service_sku,
                    "forecast_date": forecast_date,
                    "predicted_cost": round(predicted, 2),
                    "lower_80": round(predicted * 0.88, 2),
                    "upper_80": round(predicted * 1.12, 2),
                    "lower_95": round(predicted * 0.78, 2),
                    "upper_95": round(predicted * 1.22, 2),
                    "model_run_id": f"demo-{project_id[:4]}-{abs(hash(service_sku)) % 10000}",
                    "created_at": created_at,
                }
            )

    return pd.DataFrame(rows)


def build_alerts() -> pd.DataFrame:
    anomalies = build_anomalies()
    flagged = anomalies[anomalies["is_flagged"]].head(48).copy()
    rows = []

    for index, row in flagged.reset_index(drop=True).iterrows():
        severity = (
            "CRITICAL"
            if row["anomaly_confidence"] >= 0.85 and row["cost_usd"] > 5000
            else "HIGH"
        )
        threshold = max(row["rolling_avg_30d"] * 1.45, row["cost_usd"] * 0.72)
        rows.append(
            {
                "alert_id": f"demo-alert-{index + 1:03d}",
                "triggered_at": row["event_date"] + pd.Timedelta(hours=8 + index % 9),
                "project_id": row["project_id"],
                "service_sku": row["service_sku"],
                "alert_type": "cost_anomaly",
                "severity": severity,
                "cost_usd": row["cost_usd"],
                "threshold_usd": round(threshold, 2),
                "message": (
                    f"{severity} spend anomaly detected for {row['service_sku']} "
                    f"in {row['project_id']}."
                ),
                "resolved": index % 3 == 0,
                "resolved_at": (
                    row["event_date"] + pd.Timedelta(days=1, hours=2)
                    if index % 3 == 0
                    else pd.NaT
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("triggered_at", ascending=False)


def build_dashboard_kpis() -> tuple[int, float, int, int]:
    daily = build_daily_spend()
    anomalies = build_anomalies()
    forecasts = build_forecasts()
    return (
        int(daily["event_count"].sum()),
        float(round(daily["cost_usd"].sum(), 0)),
        int(anomalies["is_flagged"].sum()),
        int(forecasts[["project_id", "service_sku"]].drop_duplicates().shape[0]),
    )

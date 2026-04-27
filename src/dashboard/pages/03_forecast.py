"""
03_forecast.py
--------------
30-day spend forecast per (project, SKU) with confidence bands.
MLflow run summary.
"""

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from src.dashboard.demo_data import build_forecast_actuals, build_forecasts

DB_PATH = Path("data/finops.duckdb")

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#888", size=11),
)

st.set_page_config(page_title="Forecast · FinOps Hub", layout="wide")
st.markdown("# 📈 30-Day Spend Forecast")
st.divider()


# ── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_actuals():
    if not DB_PATH.exists():
        return build_forecast_actuals()

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return conn.execute(
            """
            SELECT
                DATE(timestamp)  AS event_date,
                project_id,
                service_sku,
                SUM(cost_usd)    AS cost_usd
            FROM billing_events
            GROUP BY 1,2,3
            ORDER BY 1
        """
        ).fetchdf()
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_forecasts():
    if not DB_PATH.exists():
        return build_forecasts()

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
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
                project_id, service_sku, forecast_date,
                predicted_cost, lower_80, upper_80, lower_95, upper_95,
                model_run_id, created_at
            FROM latest WHERE rn = 1
            ORDER BY project_id, service_sku, forecast_date
        """
        ).fetchdf()
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


actuals = load_actuals()
forecasts = load_forecasts()

if actuals.empty and forecasts.empty:
    st.stop()

actuals["event_date"] = pd.to_datetime(actuals["event_date"])
if not forecasts.empty:
    forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])

# ── Sidebar selectors ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Select series")

    if forecasts.empty:
        st.warning("No forecasts found. Run arima_model.py first.")
        st.stop()

    projects = sorted(forecasts["project_id"].unique())
    sel_project = st.selectbox("Project", projects)

    skus = sorted(
        forecasts[forecasts["project_id"] == sel_project]["service_sku"].unique()
    )
    sel_sku = st.selectbox("Service SKU", skus)

    lookback = st.slider("Actuals lookback (days)", 30, 365, 90)
    show_95 = st.checkbox("Show 95% CI", value=False)

# ── Filter series ─────────────────────────────────────────────────────────────

fc = forecasts[
    (forecasts["project_id"] == sel_project) & (forecasts["service_sku"] == sel_sku)
].copy()

act = actuals[
    (actuals["project_id"] == sel_project) & (actuals["service_sku"] == sel_sku)
].copy()

cutoff = act["event_date"].max() - pd.Timedelta(days=lookback)
act = act[act["event_date"] >= cutoff]

# ── KPI cards ─────────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("30d forecast total", f"${fc['predicted_cost'].sum():,.0f}")
with col2:
    st.metric("Avg daily forecast", f"${fc['predicted_cost'].mean():,.0f}")
with col3:
    st.metric("Forecast upper 95", f"${fc['upper_95'].sum():,.0f}")
with col4:
    last_actual = (
        act["cost_usd"].iloc[-7:].mean() if len(act) >= 7 else act["cost_usd"].mean()
    )
    st.metric("Avg daily (last 7d)", f"${last_actual:,.0f}")

st.divider()

# ── Main forecast chart ───────────────────────────────────────────────────────

fig = go.Figure()

# Actuals
fig.add_trace(
    go.Scatter(
        x=act["event_date"],
        y=act["cost_usd"],
        name="Actual spend",
        line=dict(color="#3b82f6", width=1.5),
        mode="lines",
    )
)

# 95% CI band
if show_95 and not fc.empty:
    fig.add_trace(
        go.Scatter(
            x=pd.concat([fc["forecast_date"], fc["forecast_date"][::-1]]),
            y=pd.concat([fc["upper_95"], fc["lower_95"][::-1]]),
            fill="toself",
            fillcolor="rgba(239,68,68,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
            showlegend=True,
        )
    )

# 80% CI band
if not fc.empty:
    fig.add_trace(
        go.Scatter(
            x=pd.concat([fc["forecast_date"], fc["forecast_date"][::-1]]),
            y=pd.concat([fc["upper_80"], fc["lower_80"][::-1]]),
            fill="toself",
            fillcolor="rgba(251,146,60,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% CI",
            showlegend=True,
        )
    )

# Point forecast
if not fc.empty:
    fig.add_trace(
        go.Scatter(
            x=fc["forecast_date"],
            y=fc["predicted_cost"],
            name="Forecast",
            line=dict(color="#f97316", width=2, dash="dash"),
            mode="lines",
        )
    )

# Cutoff line
if not act.empty:
    fig.add_vline(
        x=act["event_date"].max(),
        line_dash="dot",
        line_color="#444",
        annotation_text="forecast start",
        annotation_font_size=10,
        annotation_font_color="#555",
    )

fig.update_layout(
    **PLOTLY_THEME,
    title=dict(
        text=f"{sel_project}  ·  {sel_sku}",
        font=dict(size=13, color="#888"),
    ),
    height=380,
    margin=dict(t=50, b=20, l=0, r=0),
    legend=dict(orientation="h", y=1.08, x=0),
    xaxis=dict(showgrid=False, tickfont=dict(size=10)),
    yaxis=dict(
        showgrid=True,
        gridcolor="#1a1a1a",
        tickprefix="$",
        tickfont=dict(size=10),
    ),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── Forecast table ────────────────────────────────────────────────────────────

st.markdown("### Forecast values — next 30 days")
if not fc.empty:
    fc_display = fc[
        [
            "forecast_date",
            "predicted_cost",
            "lower_80",
            "upper_80",
            "lower_95",
            "upper_95",
        ]
    ].copy()
    for col in ["predicted_cost", "lower_80", "upper_80", "lower_95", "upper_95"]:
        fc_display[col] = fc_display[col].map("${:,.2f}".format)
    fc_display.columns = [
        "Date",
        "Forecast",
        "Lower 80%",
        "Upper 80%",
        "Lower 95%",
        "Upper 95%",
    ]
    st.dataframe(fc_display, use_container_width=True, height=300)

# ── All-project forecast summary ──────────────────────────────────────────────

st.divider()
st.markdown("### All-project 30-day forecast summary")

if not forecasts.empty:
    summary = (
        forecasts.groupby("project_id")
        .agg(
            total_forecast=("predicted_cost", "sum"),
            upper_95_total=("upper_95", "sum"),
            n_skus=("service_sku", "nunique"),
        )
        .sort_values("total_forecast", ascending=False)
        .reset_index()
    )

    summary["total_forecast"] = summary["total_forecast"].map("${:,.0f}".format)
    summary["upper_95_total"] = summary["upper_95_total"].map("${:,.0f}".format)
    summary.columns = ["Project", "30d Forecast", "Upper 95% Total", "SKUs"]
    st.dataframe(summary, use_container_width=True, height=280)

# ── MLflow run info ───────────────────────────────────────────────────────────

if not fc.empty:
    with st.expander("MLflow run details"):
        run_id = fc["model_run_id"].iloc[0]
        created = fc["created_at"].iloc[0]
        st.markdown(
            f"""
        ```
        Run ID    : {run_id}
        Created   : {created}
        Project   : {sel_project}
        SKU       : {sel_sku}
        Horizon   : 30 days
        GCP equiv : BigQuery ML ARIMA_PLUS
        Tracking  : Vertex AI Experiments
        ```
        """
        )

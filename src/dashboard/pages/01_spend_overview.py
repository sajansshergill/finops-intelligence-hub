"""
01_spend_overview.py
--------------------
Spend Overview — total spend by department, project, SKU, region.
WoW / MoM delta KPI cards. Daily spend trend line.
"""

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

DB_PATH = Path("data/finops.duckdb")

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#888", size=11),
)

st.set_page_config(page_title="Spend Overview · FinOps Hub", layout="wide")
st.markdown("# 💰 Spend Overview")
st.divider()


# ── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_daily_spend():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return conn.execute(
            """
            SELECT
                DATE(timestamp)         AS event_date,
                project_id,
                service_sku,
                department,
                region,
                label_env,
                SUM(cost_usd)           AS cost_usd,
                COUNT(*)                AS event_count
            FROM billing_events
            GROUP BY 1,2,3,4,5,6
            ORDER BY 1
        """
        ).fetchdf()
    finally:
        conn.close()


def delta_pct(current, prior):
    if prior == 0:
        return 0
    return round((current - prior) / prior * 100, 1)


# ── Sidebar filters ───────────────────────────────────────────────────────────

df_raw = load_daily_spend()
df_raw["event_date"] = pd.to_datetime(df_raw["event_date"])

with st.sidebar:
    st.markdown("### Filters")

    date_min = df_raw["event_date"].min().date()
    date_max = df_raw["event_date"].max().date()

    date_range = st.date_input(
        "Date range",
        value=(date_max - pd.Timedelta(days=90), date_max),
        min_value=date_min,
        max_value=date_max,
    )

    projects = st.multiselect(
        "Project",
        options=sorted(df_raw["project_id"].unique()),
        default=[],
        placeholder="All projects",
    )

    skus = st.multiselect(
        "Service SKU",
        options=sorted(df_raw["service_sku"].unique()),
        default=[],
        placeholder="All SKUs",
    )

    envs = st.multiselect(
        "Environment",
        options=sorted(df_raw["label_env"].unique()),
        default=[],
        placeholder="All environments",
    )

# ── Filter data ───────────────────────────────────────────────────────────────

df = df_raw.copy()
if len(date_range) == 2:
    df = df[
        (df["event_date"] >= pd.Timestamp(date_range[0]))
        & (df["event_date"] <= pd.Timestamp(date_range[1]))
    ]
if projects:
    df = df[df["project_id"].isin(projects)]
if skus:
    df = df[df["service_sku"].isin(skus)]
if envs:
    df = df[df["label_env"].isin(envs)]

# ── KPI cards ─────────────────────────────────────────────────────────────────

total_spend = df["cost_usd"].sum()
total_events = df["event_count"].sum()

# WoW / MoM comparisons
end_date = df["event_date"].max()
cur_week = df[df["event_date"] >= end_date - pd.Timedelta(days=7)]["cost_usd"].sum()
prev_week = df[
    (df["event_date"] >= end_date - pd.Timedelta(days=14))
    & (df["event_date"] < end_date - pd.Timedelta(days=7))
]["cost_usd"].sum()
cur_month = df[df["event_date"] >= end_date - pd.Timedelta(days=30)]["cost_usd"].sum()
prev_month = df[
    (df["event_date"] >= end_date - pd.Timedelta(days=60))
    & (df["event_date"] < end_date - pd.Timedelta(days=30))
]["cost_usd"].sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total spend", f"${total_spend:,.0f}")
with col2:
    st.metric(
        "This week", f"${cur_week:,.0f}", f"{delta_pct(cur_week, prev_week):+.1f}% WoW"
    )
with col3:
    st.metric(
        "This month",
        f"${cur_month:,.0f}",
        f"{delta_pct(cur_month, prev_month):+.1f}% MoM",
    )
with col4:
    st.metric("Billing events", f"{total_events:,.0f}")

st.divider()

# ── Daily spend trend ─────────────────────────────────────────────────────────

daily = df.groupby("event_date")["cost_usd"].sum().reset_index()
daily_7ma = daily.copy()
daily_7ma["ma7"] = daily_7ma["cost_usd"].rolling(7).mean()

fig_trend = go.Figure()
fig_trend.add_trace(
    go.Bar(
        x=daily["event_date"],
        y=daily["cost_usd"],
        name="Daily spend",
        marker_color="#1e3a5f",
        opacity=0.7,
    )
)
fig_trend.add_trace(
    go.Scatter(
        x=daily_7ma["event_date"],
        y=daily_7ma["ma7"],
        name="7-day avg",
        line=dict(color="#3b82f6", width=2),
        mode="lines",
    )
)
fig_trend.update_layout(
    **PLOTLY_THEME,
    title=dict(text="Daily spend trend", font=dict(size=13, color="#888")),
    height=280,
    margin=dict(t=40, b=20, l=0, r=0),
    legend=dict(orientation="h", y=1.1, x=0),
    xaxis=dict(showgrid=False, tickfont=dict(size=10)),
    yaxis=dict(
        showgrid=True, gridcolor="#1a1a1a", tickprefix="$", tickfont=dict(size=10)
    ),
    bargap=0.1,
)
st.plotly_chart(fig_trend, use_container_width=True)

# ── Breakdown charts ──────────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    by_project = (
        df.groupby("project_id")["cost_usd"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig_proj = px.bar(
        by_project,
        x="cost_usd",
        y="project_id",
        orientation="h",
        labels={"cost_usd": "Total spend ($)", "project_id": ""},
        color_discrete_sequence=["#3b82f6"],
    )
    fig_proj.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Spend by project", font=dict(size=13, color="#888")),
        height=300,
        margin=dict(t=40, b=20, l=0, r=0),
        xaxis=dict(tickprefix="$", showgrid=True, gridcolor="#1a1a1a"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

with col_right:
    by_sku = (
        df.groupby("service_sku")["cost_usd"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_sku = px.bar(
        by_sku,
        x="service_sku",
        y="cost_usd",
        labels={"cost_usd": "Total spend ($)", "service_sku": ""},
        color_discrete_sequence=["#6366f1"],
    )
    fig_sku.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Top 10 SKUs by spend", font=dict(size=13, color="#888")),
        height=300,
        margin=dict(t=40, b=60, l=0, r=0),
        xaxis=dict(tickangle=-35, showgrid=False),
        yaxis=dict(tickprefix="$", showgrid=True, gridcolor="#1a1a1a"),
    )
    st.plotly_chart(fig_sku, use_container_width=True)

# ── Heatmap: dept × month ─────────────────────────────────────────────────────

st.markdown("### Spend heatmap — department × month")
df["month"] = df["event_date"].dt.to_period("M").astype(str)
heatmap_data = (
    df.groupby(["department", "month"])["cost_usd"].sum().unstack(fill_value=0)
)

fig_heat = px.imshow(
    heatmap_data,
    color_continuous_scale="Blues",
    labels=dict(color="Spend ($)"),
    aspect="auto",
)
fig_heat.update_layout(
    **PLOTLY_THEME,
    height=280,
    margin=dict(t=20, b=20, l=0, r=0),
    coloraxis_colorbar=dict(tickprefix="$", thickness=12, len=0.8),
    xaxis=dict(tickfont=dict(size=9)),
    yaxis=dict(tickfont=dict(size=10)),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Raw table ─────────────────────────────────────────────────────────────────

with st.expander("Raw daily spend table"):
    summary = (
        df.groupby(["event_date", "project_id", "service_sku"])["cost_usd"]
        .sum()
        .reset_index()
        .sort_values("event_date", ascending=False)
    )
    summary["cost_usd"] = summary["cost_usd"].map("${:,.2f}".format)
    st.dataframe(summary, use_container_width=True, height=300)

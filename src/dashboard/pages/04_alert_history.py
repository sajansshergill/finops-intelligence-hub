"""
04_alert_history.py
-------------------
Alert history — all HIGH/CRITICAL alerts dispatched by the scorer.
Resolution tracking, timeline, project breakdown.
"""

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

DB_PATH = Path("data/finops.duckdb")

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#888", size=11),
)

SEVERITY_COLORS = {
    "CRITICAL": "#ef4444",
    "HIGH": "#f97316",
}

st.set_page_config(page_title="Alert History · FinOps Hub", layout="wide")
st.markdown("# 🔔 Alert History")
st.divider()


# ── Data loader ───────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)
def load_alerts():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return conn.execute(
            """
            SELECT
                alert_id,
                triggered_at,
                project_id,
                service_sku,
                alert_type,
                severity,
                cost_usd,
                threshold_usd,
                message,
                resolved,
                resolved_at
            FROM alert_history
            ORDER BY triggered_at DESC
        """
        ).fetchdf()
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


df = load_alerts()

if df.empty:
    st.info("No alerts found. Run scorer.py --write-alerts to populate alert history.")
    st.stop()

df["triggered_at"] = pd.to_datetime(df["triggered_at"])
df["date"] = df["triggered_at"].dt.date

# ── Sidebar filters ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Filters")
    sev_filter = st.multiselect(
        "Severity",
        options=["CRITICAL", "HIGH"],
        default=["CRITICAL", "HIGH"],
    )
    proj_filter = st.multiselect(
        "Project",
        options=sorted(df["project_id"].unique()),
        default=[],
        placeholder="All projects",
    )
    resolved_filter = st.radio(
        "Status",
        options=["All", "Open", "Resolved"],
        index=0,
    )

view = df.copy()
if sev_filter:
    view = view[view["severity"].isin(sev_filter)]
if proj_filter:
    view = view[view["project_id"].isin(proj_filter)]
if resolved_filter == "Open":
    view = view[~view["resolved"].astype(bool)]
elif resolved_filter == "Resolved":
    view = view[view["resolved"].astype(bool)]

# ── KPI cards ─────────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total alerts", f"{len(df):,}")
with col2:
    st.metric("Critical", f"{(df['severity']=='CRITICAL').sum():,}")
with col3:
    st.metric("Open", f"{(~df['resolved']).sum():,}")
with col4:
    st.metric("Total spend flagged", f"${df['cost_usd'].sum():,.0f}")

st.divider()

# ── Alert timeline ────────────────────────────────────────────────────────────

daily_alerts = df.groupby(["date", "severity"]).size().reset_index(name="count")
daily_alerts["date"] = pd.to_datetime(daily_alerts["date"])

fig_timeline = px.bar(
    daily_alerts,
    x="date",
    y="count",
    color="severity",
    color_discrete_map=SEVERITY_COLORS,
    labels={"date": "", "count": "Alerts", "severity": ""},
    barmode="stack",
)
fig_timeline.update_layout(
    **PLOTLY_THEME,
    title=dict(text="Alert timeline", font=dict(size=13, color="#888")),
    height=240,
    margin=dict(t=40, b=20, l=0, r=0),
    legend=dict(orientation="h", y=1.1),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#1a1a1a"),
    bargap=0.15,
)
st.plotly_chart(fig_timeline, use_container_width=True)

# ── Project + SKU breakdown ───────────────────────────────────────────────────

col_l, col_r = st.columns(2)

with col_l:
    by_proj = (
        df.groupby("project_id")
        .agg(alerts=("alert_id", "count"), spend=("cost_usd", "sum"))
        .sort_values("spend", ascending=True)
        .reset_index()
    )
    fig_proj = px.bar(
        by_proj,
        x="spend",
        y="project_id",
        orientation="h",
        labels={"spend": "Spend flagged ($)", "project_id": ""},
        color_discrete_sequence=["#ef4444"],
    )
    fig_proj.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Flagged spend by project", font=dict(size=13, color="#888")),
        height=280,
        margin=dict(t=40, b=20, l=0, r=0),
        xaxis=dict(tickprefix="$", showgrid=True, gridcolor="#1a1a1a"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

with col_r:
    by_sku = (
        df.groupby("service_sku")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    fig_sku = px.bar(
        by_sku,
        x="service_sku",
        y="count",
        labels={"service_sku": "", "count": "Alert count"},
        color_discrete_sequence=["#f97316"],
    )
    fig_sku.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Alerts by SKU", font=dict(size=13, color="#888")),
        height=280,
        margin=dict(t=40, b=60, l=0, r=0),
        xaxis=dict(tickangle=-35, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a"),
    )
    st.plotly_chart(fig_sku, use_container_width=True)

# ── Alert table ───────────────────────────────────────────────────────────────

st.markdown(f"### Alert log — {len(view):,} alerts")

display = view[
    [
        "triggered_at",
        "project_id",
        "service_sku",
        "severity",
        "cost_usd",
        "threshold_usd",
        "resolved",
        "message",
    ]
].copy()

display["triggered_at"] = display["triggered_at"].dt.strftime("%Y-%m-%d %H:%M")
display["cost_usd"] = display["cost_usd"].map("${:,.2f}".format)
display["threshold_usd"] = display["threshold_usd"].map("${:,.0f}".format)
display["resolved"] = display["resolved"].map({True: "✓ Resolved", False: "⚠ Open"})

display.columns = [
    "Triggered",
    "Project",
    "SKU",
    "Severity",
    "Cost",
    "Threshold",
    "Status",
    "Message",
]

st.dataframe(display, use_container_width=True, height=420)

# ── Resolution rate ───────────────────────────────────────────────────────────

with st.expander("Resolution stats"):
    res_rate = df["resolved"].mean() * 100
    avg_cost_open = df[~df["resolved"]]["cost_usd"].mean()
    avg_cost_resolved = df[df["resolved"]]["cost_usd"].mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Resolution rate", f"{res_rate:.1f}%")
    with c2:
        st.metric(
            "Avg cost (open)",
            f"${avg_cost_open:,.2f}" if not pd.isna(avg_cost_open) else "N/A",
        )
    with c3:
        st.metric(
            "Avg cost (resolved)",
            f"${avg_cost_resolved:,.2f}" if not pd.isna(avg_cost_resolved) else "N/A",
        )

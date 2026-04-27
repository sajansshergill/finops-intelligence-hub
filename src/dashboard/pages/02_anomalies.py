"""
02_anomalies.py
---------------
Anomaly detection results — flagged events, confidence scores,
Z-score drill-down, severity breakdown.
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

SEVERITY_COLORS = {
    "CRITICAL": "#ef4444",
    "HIGH": "#f97316",
    "MEDIUM": "#eab308",
    "LOW": "#6b7280",
}

st.set_page_config(page_title="Anomalies · FinOps Hub", layout="wide")
st.markdown("# 🚨 Anomaly Detection")
st.divider()


# ── Data loader ───────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_anomalies():
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return conn.execute(
            """
            SELECT
                project_id,
                service_sku,
                department,
                event_date,
                cost_usd,
                rolling_avg_30d,
                pct_change_1d,
                z_score_7d,
                z_score_30d,
                isolation_score,
                anomaly_confidence,
                is_flagged,
                has_injected_anomaly
            FROM anomaly_scored_features
            ORDER BY anomaly_confidence DESC
        """
        ).fetchdf()
    except Exception as e:
        st.error(f"Run anomaly_engine.py first: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def assign_severity(row):
    c, cost = row["anomaly_confidence"], row["cost_usd"]
    if c >= 0.85 and cost > 5000:
        return "CRITICAL"
    elif c >= 0.70 and cost > 1000:
        return "HIGH"
    elif c >= 0.58:
        return "MEDIUM"
    else:
        return "LOW"


df = load_anomalies()
if df.empty:
    st.info(
        "No anomaly data found. Streamlit Cloud does not include `data/finops.duckdb`; "
        "run anomaly_engine.py after loading billing data to populate this page."
    )
    st.stop()

df["event_date"] = pd.to_datetime(df["event_date"])
df["severity"] = df.apply(assign_severity, axis=1)
df_flagged = df[df["is_flagged"]].copy()

# ── Sidebar filters ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Filters")
    severity_filter = st.multiselect(
        "Severity",
        options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH"],
    )
    project_filter = st.multiselect(
        "Project",
        options=sorted(df_flagged["project_id"].unique()),
        default=[],
        placeholder="All projects",
    )
    conf_threshold = st.slider("Min confidence", 0.0, 1.0, 0.58, 0.01)

# ── Apply filters ─────────────────────────────────────────────────────────────

view = df_flagged[df_flagged["anomaly_confidence"] >= conf_threshold]
if severity_filter:
    view = view[view["severity"].isin(severity_filter)]
if project_filter:
    view = view[view["project_id"].isin(project_filter)]

# ── KPI row ───────────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total flagged", f"{len(df_flagged):,}")
with col2:
    n_crit = (df_flagged["severity"] == "CRITICAL").sum()
    st.metric("Critical", f"{n_crit:,}")
with col3:
    n_high = (df_flagged["severity"] == "HIGH").sum()
    st.metric("High", f"{n_high:,}")
with col4:
    precision = (
        (df_flagged["has_injected_anomaly"].sum() / len(df_flagged) * 100)
        if len(df_flagged) > 0
        else 0
    )
    st.metric("Precision", f"{precision:.1f}%")
with col5:
    at_risk = df_flagged["cost_usd"].sum()
    st.metric("Spend at risk", f"${at_risk:,.0f}")

st.divider()

# ── Charts row ────────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    # Confidence score distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(
            x=df["anomaly_confidence"],
            nbinsx=50,
            name="All events",
            marker_color="#1e3a5f",
            opacity=0.7,
        )
    )
    fig_dist.add_trace(
        go.Histogram(
            x=df_flagged["anomaly_confidence"],
            nbinsx=50,
            name="Flagged",
            marker_color="#ef4444",
            opacity=0.8,
        )
    )
    fig_dist.add_vline(
        x=0.58,
        line_dash="dash",
        line_color="#f97316",
        annotation_text="threshold=0.58",
        annotation_font_size=10,
    )
    fig_dist.update_layout(
        **PLOTLY_THEME,
        title=dict(
            text="Confidence score distribution", font=dict(size=13, color="#888")
        ),
        height=280,
        margin=dict(t=40, b=20, l=0, r=0),
        barmode="overlay",
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(title="Anomaly confidence", showgrid=False),
        yaxis=dict(title="Count", showgrid=True, gridcolor="#1a1a1a"),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_right:
    # Severity breakdown donut
    sev_counts = df_flagged["severity"].value_counts().reset_index()
    sev_counts.columns = ["severity", "count"]
    fig_donut = px.pie(
        sev_counts,
        values="count",
        names="severity",
        hole=0.6,
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
    )
    fig_donut.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Severity breakdown", font=dict(size=13, color="#888")),
        height=280,
        margin=dict(t=40, b=20, l=0, r=0),
        legend=dict(orientation="h", y=-0.1),
        showlegend=True,
    )
    fig_donut.update_traces(textinfo="percent+label", textfont_size=11)
    st.plotly_chart(fig_donut, use_container_width=True)

# ── Z-score scatter ───────────────────────────────────────────────────────────

st.markdown("### Z-score vs cost — anomaly map")
sample = df.sample(min(5000, len(df)), random_state=42)
fig_scatter = px.scatter(
    sample,
    x="z_score_7d",
    y="cost_usd",
    color="anomaly_confidence",
    color_continuous_scale="RdYlGn_r",
    hover_data=["project_id", "service_sku", "event_date"],
    labels={
        "z_score_7d": "Z-score (7d)",
        "cost_usd": "Cost ($)",
        "anomaly_confidence": "Confidence",
    },
    opacity=0.6,
    size_max=6,
)
fig_scatter.add_vline(x=2.0, line_dash="dash", line_color="#555", line_width=1)
fig_scatter.add_vline(x=-2.0, line_dash="dash", line_color="#555", line_width=1)
fig_scatter.update_layout(
    **PLOTLY_THEME,
    height=320,
    margin=dict(t=20, b=20, l=0, r=0),
    xaxis=dict(showgrid=True, gridcolor="#1a1a1a"),
    yaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickprefix="$"),
    coloraxis_colorbar=dict(thickness=12, len=0.8, title="Conf."),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Flagged events table ──────────────────────────────────────────────────────

st.markdown(f"### Flagged anomalies — {len(view):,} events")

display = (
    view[
        [
            "event_date",
            "project_id",
            "service_sku",
            "department",
            "cost_usd",
            "rolling_avg_30d",
            "pct_change_1d",
            "z_score_7d",
            "anomaly_confidence",
            "severity",
        ]
    ]
    .copy()
    .sort_values("anomaly_confidence", ascending=False)
)

display["cost_usd"] = display["cost_usd"].map("${:,.2f}".format)
display["rolling_avg_30d"] = display["rolling_avg_30d"].map("${:,.2f}".format)
display["pct_change_1d"] = display["pct_change_1d"].map("{:+.1%}".format)
display["z_score_7d"] = display["z_score_7d"].map("{:.2f}".format)
display["anomaly_confidence"] = display["anomaly_confidence"].map("{:.3f}".format)

st.dataframe(
    display.rename(
        columns={
            "event_date": "Date",
            "project_id": "Project",
            "service_sku": "SKU",
            "department": "Dept",
            "cost_usd": "Cost",
            "rolling_avg_30d": "30d avg",
            "pct_change_1d": "DoD Δ",
            "z_score_7d": "Z-score",
            "anomaly_confidence": "Confidence",
            "severity": "Severity",
        }
    ),
    use_container_width=True,
    height=420,
)

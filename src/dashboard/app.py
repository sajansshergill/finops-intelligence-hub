"""
app.py
------
FinOps Intelligence Hub — Streamlit Dashboard
Main entry point. Configures page layout and shared state.

Run:
    streamlit run src/dashboard/app.py
"""

from pathlib import Path

import duckdb
import streamlit as st
from src.dashboard.demo_data import build_dashboard_kpis

st.set_page_config(
    page_title="FinOps Intelligence Hub",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Font + base */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f0f0f;
    border-right: 1px solid #1e1e1e;
}
section[data-testid="stSidebar"] * {
    color: #c8c8c8 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label {
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #666 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 16px 20px;
}
[data-testid="stMetric"] label {
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #555 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px !important;
    color: #f0f0f0 !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e1e;
    border-radius: 8px;
}

/* Headers */
h1 { font-size: 22px !important; font-weight: 600 !important; letter-spacing: -0.02em; }
h2 { font-size: 16px !important; font-weight: 500 !important; color: #888 !important; }
h3 { font-size: 13px !important; font-weight: 500 !important; letter-spacing: 0.06em; text-transform: uppercase; color: #555 !important; }

/* Tab styling */
[data-testid="stTabs"] button {
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Severity badges */
.badge-critical { background:#3d0f0f; color:#f87171; border:1px solid #7f1d1d; padding:2px 8px; border-radius:4px; font-size:11px; font-family:'IBM Plex Mono',monospace; }
.badge-high     { background:#3d2a0f; color:#fb923c; border:1px solid #7c3600; padding:2px 8px; border-radius:4px; font-size:11px; font-family:'IBM Plex Mono',monospace; }
.badge-medium   { background:#2d2d00; color:#facc15; border:1px solid #713f12; padding:2px 8px; border-radius:4px; font-size:11px; font-family:'IBM Plex Mono',monospace; }
.badge-low      { background:#111; color:#6b7280; border:1px solid #374151; padding:2px 8px; border-radius:4px; font-size:11px; font-family:'IBM Plex Mono',monospace; }

/* Divider */
hr { border-color: #1e1e1e; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-size:18px; font-weight:600; color:#f0f0f0; letter-spacing:-0.02em;'>
            💸 FinOps Hub
        </div>
        <div style='font-size:11px; color:#444; letter-spacing:0.06em; text-transform:uppercase; margin-top:4px;'>
            Intelligence · Detection · Forecast
        </div>
    </div>
    <hr style='border-color:#1e1e1e; margin: 0 0 20px 0;'>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style='font-size:11px; color:#444; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:8px;'>
        Navigate
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.page_link("app.py", label="Overview", icon="📊")
    st.page_link("pages/01_spend_overview.py", label="Spend", icon="💰")
    st.page_link("pages/02_anomalies.py", label="Anomalies", icon="🚨")
    st.page_link("pages/03_forecast.py", label="Forecast", icon="📈")
    st.page_link("pages/04_alert_history.py", label="Alert history", icon="🔔")

    st.markdown(
        "<hr style='border-color:#1e1e1e; margin: 20px 0;'>", unsafe_allow_html=True
    )

    st.markdown(
        """
    <div style='font-size:10px; color:#333; line-height:1.8;'>
        GCP Equivalent Stack<br>
        <span style='color:#555;'>Kafka → Pub/Sub</span><br>
        <span style='color:#555;'>DuckDB → BigQuery</span><br>
        <span style='color:#555;'>Airflow → Cloud Composer</span><br>
        <span style='color:#555;'>ARIMA → BQML ARIMA_PLUS</span><br>
        <span style='color:#555;'>This UI → Looker Studio</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── Home page ─────────────────────────────────────────────────────────────────
st.markdown("# FinOps Intelligence Hub")
st.markdown("#### Cloud cost anomaly detection & 30-day spend forecasting")
st.divider()

col1, col2, col3, col4 = st.columns(4)

DB_PATH = Path("data/finops.duckdb")


@st.cache_data(ttl=300)
def get_kpis():
    if not DB_PATH.exists():
        return build_dashboard_kpis()

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        total = conn.execute("SELECT COUNT(*) FROM billing_events").fetchone()[0]
        spend = conn.execute(
            "SELECT ROUND(SUM(cost_usd),0) FROM billing_events"
        ).fetchone()[0]
        flagged = 0
        try:
            flagged = conn.execute(
                "SELECT COUNT(*) FROM anomaly_scored_features WHERE is_flagged=true"
            ).fetchone()[0]
        except Exception:
            pass
        forecasts = 0
        try:
            forecasts = conn.execute(
                "SELECT COUNT(DISTINCT project_id || service_sku) FROM forecasts"
            ).fetchone()[0]
        except Exception:
            pass
        return total, spend, flagged, forecasts
    finally:
        conn.close()


try:
    kpis = get_kpis()
    total, spend, flagged, forecasts = kpis
    with col1:
        st.metric("Total events", f"{total:,.0f}")
    with col2:
        st.metric("Total spend", f"${spend:,.0f}")
    with col3:
        st.metric("Anomalies flagged", f"{flagged:,.0f}")
    with col4:
        st.metric("Forecast series", f"{forecasts:,.0f}")
except Exception as e:
    st.error(f"Could not connect to DuckDB: {e}")

st.divider()

st.markdown(
    """
<div style='display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:8px;'>
    <div style='background:#111; border:1px solid #1e1e1e; border-radius:8px; padding:20px;'>
        <div style='font-size:11px; letter-spacing:0.08em; text-transform:uppercase; color:#555; margin-bottom:12px;'>Detection engine</div>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:13px; color:#c8c8c8; line-height:2;'>
            Z-Score (7d / 30d rolling)<br>
            Isolation Forest (16 features)<br>
            Threshold · <span style='color:#fb923c;'>0.58</span><br>
            Precision · <span style='color:#f0f0f0;'>32.8%</span><br>
            Recall · <span style='color:#f0f0f0;'>50.9%</span><br>
            F1 · <span style='color:#34d399;'>0.399</span>
        </div>
    </div>
    <div style='background:#111; border:1px solid #1e1e1e; border-radius:8px; padding:20px;'>
        <div style='font-size:11px; letter-spacing:0.08em; text-transform:uppercase; color:#555; margin-bottom:12px;'>Forecast engine</div>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:13px; color:#c8c8c8; line-height:2;'>
            Auto-ARIMA per (project, SKU)<br>
            Log-transform · variance stable<br>
            Horizon · <span style='color:#fb923c;'>30 days</span><br>
            Confidence · <span style='color:#f0f0f0;'>80% / 95% CI</span><br>
            Tracking · <span style='color:#f0f0f0;'>MLflow</span><br>
            GCP eq. · <span style='color:#34d399;'>BQML ARIMA_PLUS</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
<div style='font-size:11px; color:#333; text-align:center; letter-spacing:0.06em;'>
    SAJAN SHERGILL · MS DATA SCIENCE · PACE UNIVERSITY 2026 · 
    <a href='https://linkedin.com/in/sajanshergill' style='color:#444;'>LINKEDIN</a> · 
    <a href='https://sajansshergill.github.io' style='color:#444;'>PORTFOLIO</a>
</div>
""",
    unsafe_allow_html=True,
)

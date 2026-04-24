# FinOps Intelligence Hub 💸
> Cloud Cost Anomaly Detection & Forecasting Engine — Local-First, GCP-Mapped

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Airflow](https://img.shields.io/badge/Airflow-Astro_CLI-017CEE?logo=apacheairflow)](https://astronomer.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10-FFD700?logo=duckdb)](https://duckdb.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Kafka](https://img.shields.io/badge/Kafka-Streaming-231F20?logo=apachekafka)](https://kafka.apache.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)

---

## The Problem

Enterprise customers migrating to the cloud face a trust crisis before they ever get to scale:
**runaway spend they can't explain.**

A BigQuery slot burst on a Monday. A Dataflow job that ran 6x longer than expected. Storage costs
that tripled because no one set a lifecycle policy. Finance escalates. Engineering shrugs.
Leadership questions the entire migration.

This is the #1 adoption blocker a Google Cloud Customer Engineer encounters in the field —
not a technical gap, but a **visibility gap.** Cloud cost anomaly detection, delivered as an MVP
in a customer workshop, closes deals.

This project is that MVP — built locally with open-source tooling, architected 1:1 against the
GCP stack a CE would deploy in production.

---

## Architecture

```
Synthetic Billing Data (Faker)
        │
        ▼
┌──────────────────┐     ┌─────────────────────┐
│  Apache Kafka    │────▶│  Kafka Consumer      │  ← Pub/Sub equivalent
│  (Streaming)     │     │  (Python)            │
└──────────────────┘     └──────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │      DuckDB         │  ← BigQuery equivalent
                          │  (Parquet on GCS)   │
                          └─────────┬──────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
   │  Anomaly Engine  │  │  Forecast Engine  │  │  Alert Engine        │
   │  (Isolation      │  │  (ARIMA / stats   │  │  (Cloud Functions    │
   │   Forest /       │  │   models)         │  │   equivalent →       │
   │   Z-Score)       │  │                   │  │   Slack / Email)     │
   └────────┬─────────┘  └────────┬──────────┘  └──────────┬───────────┘
            └────────────────────┬┘                         │
                                 ▼                          │
                    ┌────────────────────────┐              │
                    │  Apache Airflow        │◀─────────────┘
                    │  (Cloud Composer eq.)  │
                    │  DAG Orchestration     │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Streamlit Dashboard   │  ← Looker Studio equivalent
                    │  • Spend by dept/SKU   │
                    │  • Anomaly flags       │
                    │  • 30-day forecast     │
                    │  • Alert history       │
                    └────────────────────────┘
```

---

## GCP Equivalency Map

Every component maps directly to a production GCP service — making this a pitchable MVP:

| Local Component | GCP Production Equivalent | Notes |
|---|---|---|
| Apache Kafka | **Cloud Pub/Sub** | Message ingestion, real-time billing events |
| DuckDB | **BigQuery** | Analytical queries, BQML anomaly detection |
| Parquet files | **Google Cloud Storage** | Raw billing export landing zone |
| statsmodels ARIMA | **BigQuery ML (ARIMA_PLUS)** | Native SQL-based forecasting |
| Isolation Forest | **BigQuery ML (BOOSTED_TREE_CLASSIFIER)** | Anomaly classification |
| Apache Airflow (Astro CLI) | **Cloud Composer** | Managed Airflow, DAG scheduling |
| Python alert scripts | **Cloud Functions** | Serverless event-driven alerting |
| Streamlit | **Looker Studio / Looker** | Dashboards, embedding, sharing |
| Docker Compose | **Cloud Run / GKE** | Container orchestration |
| MLflow | **Vertex AI Experiments** | Model tracking, lineage |

---

## Project Structure

```
finops-intelligence-hub/
├── dags/                          # Airflow DAGs (Cloud Composer equivalent)
│   ├── ingest_billing_dag.py      # Kafka → DuckDB ingestion
│   ├── anomaly_detection_dag.py   # Daily anomaly scoring
│   ├── forecast_dag.py            # 30-day spend forecast
│   └── alert_dispatch_dag.py      # Threshold breach alerting
│
├── src/
│   ├── ingestion/
│   │   ├── producer.py            # Faker → Kafka billing event generator
│   │   ├── consumer.py            # Kafka → Parquet → DuckDB loader
│   │   └── schema.py              # Billing event schema (Avro-compatible)
│   │
│   ├── detection/
│   │   ├── anomaly_engine.py      # Isolation Forest + Z-Score detection
│   │   ├── engineering.py # Lag features, rolling averages
│   │   └── scorer.py              # Anomaly scoring, threshold logic
│   │
│   ├── forecasting/
│   │   ├── arima_model.py         # ARIMA_PLUS equivalent (statsmodels)
│   │   ├── model_registry.py      # MLflow experiment tracking
│   │   └── forecast_writer.py     # Writes forecasts back to DuckDB
│   │
│   ├── alerting/
│   │   ├── alert_engine.py        # Cloud Functions equivalent
│   │   ├── slack_notifier.py      # Slack webhook integration
│   │   └── email_notifier.py      # SMTP alert dispatch
│   │
│   └── dashboard/
│       ├── app.py                 # Streamlit entry point
│       ├── pages/
│       │   ├── 01_spend_overview.py
│       │   ├── 02_anomalies.py
│       │   ├── 03_forecast.py
│       │   └── 04_alert_history.py
│       └── components/
│           ├── charts.py          # Plotly chart components
│           └── filters.py         # Department / SKU / date filters
│
├── data/
│   ├── raw/                       # Parquet landing zone (GCS equivalent)
│   ├── processed/                 # Cleaned billing data
│   └── forecasts/                 # Model output
│
├── models/                        # MLflow model artifacts
│   └── mlruns/
│
├── tests/
│   ├── test_anomaly_engine.py
│   ├── test_forecast.py
│   └── test_ingestion.py
│
├── docker-compose.yml             # Kafka + Zookeeper + Airflow + Streamlit
├── Dockerfile
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI
└── README.md
```

---

## Synthetic Data Model

Billing events are generated with `Faker` to simulate a financial services firm
(e.g., a regional bank migrating from on-prem Hadoop to GCP):

```python
# Sample billing event schema
{
  "event_id": "uuid4",
  "timestamp": "2024-01-15T09:23:11Z",
  "project_id": "risk-analytics-prod",
  "department": "Quantitative Research",
  "service_sku": "BigQuery: Slot Usage",
  "region": "us-east1",
  "cost_usd": 847.32,
  "usage_amount": 1240.5,
  "usage_unit": "slot-hours",
  "label_team": "derivatives-desk",
  "label_env": "production"
}
```

Events span **18 months** with realistic seasonality — month-end spikes,
quarter-close processing bursts, and injected anomalies (3% of records).

---

## Anomaly Detection Logic

Two complementary methods run in parallel, mimicking what BigQuery ML offers natively:

**1. Statistical Z-Score (fast, interpretable)**

```sql
-- DuckDB equivalent of BQML anomaly detection
SELECT
    project_id,
    service_sku,
    cost_usd,
    AVG(cost_usd) OVER (
        PARTITION BY project_id, service_sku
        ORDER BY DATE(timestamp)
        ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
    ) AS rolling_avg,
    STDDEV(cost_usd) OVER (
        PARTITION BY project_id, service_sku
        ORDER BY DATE(timestamp)
        ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
    ) AS rolling_std,
    (cost_usd - rolling_avg) / NULLIF(rolling_std, 0) AS z_score
FROM billing_events
QUALIFY ABS(z_score) > 3.0
```

**2. Isolation Forest (ML-based, multi-feature)**

Uses `sklearn.ensemble.IsolationForest` on engineered features:
lag spend, rolling 7d/30d averages, day-of-week, is-month-end flag.

Both scores are combined into a single `anomaly_confidence` field written back to DuckDB.

---

## Forecasting

30-day ahead spend forecasting per `(project_id, service_sku)` using `statsmodels` ARIMA:

- **Model**: ARIMA(1,1,1) with seasonal decomposition for month-end effects
- **Output**: Point forecast + 80% / 95% confidence intervals
- **Tracking**: MLflow logs model params, MAPE, RMSE per run
- **GCP Equivalent**: `CREATE MODEL ... OPTIONS(model_type='ARIMA_PLUS')` in BigQuery ML

---

## Airflow DAGs

| DAG | Schedule | Description |
|---|---|---|
| `ingest_billing` | `*/5 * * * *` | Kafka consumer → DuckDB loader |
| `anomaly_detection` | `0 6 * * *` | Daily anomaly scoring on prior day |
| `forecast_refresh` | `0 2 * * 1` | Weekly model retrain + 30-day forecast |
| `alert_dispatch` | `0 7 * * *` | Check thresholds, fire Slack/email alerts |

---

## Streamlit Dashboard

Four pages, dark-themed, Plotly-powered:

- **Spend Overview** — Total spend by department, SKU, region; WoW / MoM delta cards
- **Anomalies** — Flagged events table with confidence score, Z-score drill-down
- **Forecast** — 30-day projection with confidence bands per project/SKU
- **Alert History** — Log of all dispatched alerts with resolution status

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/sajansshergill/finops-intelligence-hub
cd finops-intelligence-hub

# Start infrastructure (Kafka + Airflow)
docker-compose up -d

# Install dependencies
pip install -r requirements.txt

# Generate 18 months of synthetic billing data
python src/ingestion/producer.py --months 18 --anomaly-rate 0.03

# Run anomaly detection manually
python src/detection/anomaly_engine.py

# Run forecast
python src/forecasting/arima_model.py

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## CI/CD

GitHub Actions pipeline on every push to `main`:

```yaml
jobs:
  test:     # pytest suite
  lint:     # ruff + black
  docker:   # docker-compose build validation
```

---

## Business Context: The CE Pitch

This project was built to simulate an **MVP deliverable** in a Google Cloud
Customer Engineer pre-sales engagement with a financial services customer.

**Customer scenario**: Regional bank, 1,200 engineers, mid-migration from on-prem
Hadoop to GCP. Finance team flagged a $340K overage in Month 2. Cloud program
at risk of being paused pending an audit.

**CE approach**:
1. 2-hour discovery call to map spend categories to GCP SKUs
2. 3-day MVP sprint producing this pipeline on sample billing export data
3. Live demo in customer's sandbox environment
4. Outcome: Technical win secured; customer approved Phase 2 migration budget

**GCP services this translates to in production**:

| Layer | GCP Service |
|---|---|
| Ingestion | Pub/Sub + Dataflow |
| Storage & Query | BigQuery + GCS |
| ML | BigQuery ML (ARIMA_PLUS, anomaly detection) |
| Orchestration | Cloud Composer |
| Alerting | Cloud Functions |
| Visualization | Looker Studio |
| Model Tracking | Vertex AI Experiments |

---

## Author

**Sajan Shergill**
MS Data Science, Pace University (May 2026)
[LinkedIn](https://linkedin.com/in/sajanshergill) · [Portfolio](https://sajansshergill.github.io)

> *Built as a portfolio project targeting the Google Cloud Customer Engineer —
> Data Analytics role, demonstrating pre-sales MVP development, GCP data stack
> expertise, and financial services domain knowledge.*

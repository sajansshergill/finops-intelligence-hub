# FinOps Intelligence Hub: Local-First, GCP-Mapped Build
> Cloud Cost Anomaly Detection & Forecasting Engine —— Local-First, GCP-Mapped

---

## The Problem
Enterprise customers migrating to the cloud face trust crisis before they ever get to scale: **runawya spend they can't explain.**

A BigQuery slot burst on a Monday. A Dataflow job that ran 6x longer than expected. Stroage costs that tripled because no one set a lifecycle policy. Finances escalates. Engineering shurgs. Leadership questions the entire migration.

This is the #1 adoption blocker a Google Clous Customer Engineer encounters in the field —— not a technical gap, but a **visibility gap**. Cloud cost anomaly detection, delivered as an MVP in a customer workshop, closes deals.

This project is that MVP —— build locally with open-source tooling, architected 1:1 asgainst the GCP stack a CE would deploy in production.

---

## Architecture
<img width="856" height="1404" alt="image" src="https://github.com/user-attachments/assets/aab61e85-9289-45f2-9445-41825fc3313c" />

---

## GCP Equivalency Map

This project is designed to be pitched to enterprise customers as a proof-of-concept. Every 
component maps directly to a production GCP service:

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
<img width="652" height="1544" alt="image" src="https://github.com/user-attachments/assets/974a4dcf-66a5-4543-96f6-5401a32ab729" />

---

## Synthetic Data Model

Biling events are generate with `Faker` to simulate a financial services firm (e.g., a regional bank migrating from on-prem Hadoop to GCP).

```python
# Sample billing event schema
{
"event_id": "uuid4,
"timestamp:" "2024-01-15T09:23:11Z",
"project_id": "risk-analytics-prod",
"department": "Quantitative Research",
"service_sku": "BigQuery: Slot Usage",
"region": "us-east1",
"cost_usd": 847.32,
"usage_amount": 1240.5,
"usage_unit": "slot-hours",
"label_team": "derivates-desk",
"label_env": "production"
}
```

Events span **18 months** with realistic seasonality – month-end spikes,
quarter-close processing bursts, and injected anomlaies (3% of records).

---

## Anomaly Detection Logic

Two complmentary methods run in parallel mimicking what BigQuery ML offers natively:

**1. Statistical Z-Score (fast, interpreatable)**
```sql
-- DuckDB equivalent of BQML anomaly detection
SELECT
  project_id,
  service_sku,
  cost_usd,
  AVG(cost_usd) OVER (
    PARTITION BY project_idm service_sku
    ORDER BY DATE(timestamp)
    ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
  ) AS rolling_std,
  (cost_usd - rolling_avg / NULLIF(rolling_std, 0) AS z_score
FROM billing_events
QUALIFY ABS(z_score) > 3.0

### 2. Isolation Forest (ML-based, multi-feature)

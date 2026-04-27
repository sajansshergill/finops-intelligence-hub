"""
schema.py
---------
Billing event schema for the FinOps Intelligence Hub.
Mirrors the structure of a GCP Billing Export to BigQuery.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import uuid

# GCP service SKUs to simulate realistic billing data
SERVICE_SKUS = [
    "BigQuery: Slot Usage",
    "BigQuery: Storage",
    "BigQuery: Streaming Inserts",
    "Cloud Storage: Regional Storage",
    "Cloud Storage: Egress",
    "Dataflow: Batch Processing",
    "Dataflow: Streaming Processing",
    "Cloud Composer: Environment Fee",
    "Cloud Functions: Invocations",
    "Cloud Functions: Compute Time",
    "Pub/Sub: Message Delivery",
    "Vertex AI: Training",
    "Vertex AI: Prediction",
    "Compute Engine: N2 Instance",
    "Cloud SQL: PostgreSQL",
]

DEPARTMENTS = [
    "Quantitative Research",
    "Risk Analytics",
    "Data Engineering",
    "Compliance",
    "Trading Operations",
    "Machine Learning Platform",
    "Data Science",
    "Finance & Reporting",
]

PROJECTS = [
    "risk-analytics-prod",
    "risk-analytics-dev",
    "quant-research-prod",
    "ml-platform-prod",
    "ml-platform-staging",
    "data-engineering-prod",
    "compliance-reporting-prod",
    "trading-ops-prod",
]

REGIONS = [
    "us-east1",
    "us-central1",
    "us-west1",
    "europe-west1",
]

ENVIRONMENTS = ["production", "staging", "development"]

USAGE_UNITS = {
    "BigQuery: Slot Usage": ("slot-hours", 0.04, 2000),
    "BigQuery: Storage": ("gibibyte month", 0.02, 50000),
    "BigQuery: Streaming Inserts": ("mebibytes", 0.01, 10000),
    "Cloud Storage: Regional Storage": ("gibibyte month", 0.02, 100000),
    "Cloud Storage: Egress": ("gibibytes", 0.08, 5000),
    "Dataflow: Batch Processing": ("vCPU hour", 0.056, 500),
    "Dataflow: Streaming Processing": ("vCPU hour", 0.069, 300),
    "Cloud Composer: Environment Fee": ("hours", 0.10, 720),
    "Cloud Functions: Invocations": ("invocations", 0.0000004, 5000000),
    "Cloud Functions: Compute Time": ("GB-seconds", 0.0000025, 200000),
    "Pub/Sub: Message Delivery": ("tebibyte", 0.05, 10),
    "Vertex AI: Training": ("vCPU hour", 0.12, 200),
    "Vertex AI: Prediction": ("vCPU hour", 0.08, 150),
    "Compute Engine: N2 Instance": ("hours", 0.085, 720),
    "Cloud SQL: PostgreSQL": ("hours", 0.105, 720),
}


@dataclass
class BillingEvent:
    event_id: str
    timestamp: str
    project_id: str
    department: str
    service_sku: str
    region: str
    cost_usd: float
    usage_amount: float
    usage_unit: str
    label_team: str
    label_env: str
    is_anomaly: bool = False  # ground truth label for evaluation

    @classmethod
    def create(
        cls,
        project_id: str,
        department: str,
        service_sku: str,
        region: str,
        cost_usd: float,
        usage_amount: float,
        label_env: str,
        timestamp: datetime,
        is_anomaly: bool = False,
    ) -> "BillingEvent":
        unit, _, _ = USAGE_UNITS.get(service_sku, ("units", 1.0, 100))
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=timestamp.isoformat() + "Z",
            project_id=project_id,
            department=department,
            service_sku=service_sku,
            region=region,
            cost_usd=round(cost_usd, 4),
            usage_amount=round(usage_amount, 4),
            usage_unit=unit,
            label_team=department.lower().replace(" ", "-"),
            label_env=label_env,
            is_anomaly=is_anomaly,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "BillingEvent":
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "BillingEvent":
        return cls(**data)

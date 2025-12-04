# api/schemas.py
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    project_id: str
    platform: str
    bq_dataset: str
    bq_table_predictions: str
    model_path: str

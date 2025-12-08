# api/main.py
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI

from api.deps import get_cfg, get_model_bundle, log_prediction_to_bq
from api.schemas import HealthResponse, PredictionResponse, TransactionInput
from internalpy.config import Cfg, load_vars
from internalpy.log import get_logger

# Load config and logger once at import time.
# load_vars() is already lru_cached inside internalpy.config.
_VARS = load_vars()
_LOGGER = get_logger(_VARS.Platform)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan handler.

    Replaces the deprecated @app.on_event("startup") mechanism.
    """
    # Startup
    cfg = get_cfg()
    _LOGGER.info(
        'fraud api starting',
        extra={
            'project_id': cfg.Vars.ProjectID,
            'platform': cfg.Vars.Platform,
            'bq_dataset': cfg.Vars.BQDataset,
            'bq_table_predictions': cfg.Vars.BQTablePredictions,
            'model_path': cfg.Vars.ModelPath,
        },
    )

    # Warm up the model bundle so first prediction doesn't pay the load cost
    try:
        get_model_bundle()
    except Exception:
        _LOGGER.exception('Failed to load model bundle during startup')

    try:
        yield
    finally:
        _LOGGER.info(
            'fraud api shutting down',
            extra={
                'project_id': cfg.Vars.ProjectID,
                'platform': cfg.Vars.Platform,
            },
        )


app = FastAPI(
    title='Fraud Detection API',
    version='0.1.0',
    docs_url='/docs',
    redoc_url='/redoc',
    lifespan=lifespan,
)


@app.get('/health', response_model=HealthResponse, tags=['meta'])
def health(cfg: Cfg = Depends(get_cfg)) -> HealthResponse:
    """
    Simple health check that also returns key config values.
    Useful to verify env vars / Secret Manager wiring in each environment.
    """
    _LOGGER.info(
        'health check',
        extra={
            'project_id': cfg.Vars.ProjectID,
            'platform': cfg.Vars.Platform,
        },
        stacklevel=2,
    )
    return HealthResponse(
        status='ok',
        project_id=cfg.Vars.ProjectID,
        platform=cfg.Vars.Platform,
        bq_dataset=cfg.Vars.BQDataset,
        bq_table_predictions=cfg.Vars.BQTablePredictions,
        model_path=cfg.Vars.ModelPath,
    )


@app.post('/predict', response_model=PredictionResponse, tags=['inference'])
def predict(
    tx: TransactionInput,
    cfg: Cfg = Depends(get_cfg),
) -> PredictionResponse:
    """
    Score a single transaction and return:
      - fraud_prob: probability of fraud
      - fraud_flag: bool decision based on cost-optimized threshold
      - model_version, threshold from metadata
    """
    bundle = get_model_bundle()
    pipeline = bundle.pipeline
    metadata = bundle.metadata

    raw_features = metadata.get('features')
    if not isinstance(raw_features, list):
        raise RuntimeError("Metadata is missing 'features' list")
    features: list[str] = [str(f) for f in raw_features]

    threshold_raw = metadata.get('threshold')
    if threshold_raw is None:
        raise RuntimeError("Metadata is missing 'threshold' value")

    if not isinstance(threshold_raw, (int | float)):
        raise RuntimeError(f"Metadata 'threshold' must be numeric, got {type(threshold_raw)!r}")

    threshold = float(threshold_raw)
    model_version = str(metadata.get('model_version', 'unknown'))

    row_dict: dict[str, Any] = tx.model_dump()
    try:
        row_values = [row_dict[f] for f in features]
    except KeyError as e:
        raise RuntimeError(f'Missing feature in input: {e}') from e

    features_index = pd.Index(features)
    df = pd.DataFrame([row_values], columns=features_index)

    # Build a 1-row DataFrame in the correct feature order
    row_dict = tx.model_dump()
    try:
        row_values = [row_dict[f] for f in features]
    except KeyError as e:
        raise RuntimeError(f'Missing feature in input: {e}') from e

    df = pd.DataFrame([row_values], columns=pd.Index(features))

    # Predict probability
    fraud_prob = float(pipeline.predict_proba(df)[:, 1][0])
    fraud_flag = fraud_prob >= threshold

    # Log to app logger
    _LOGGER.info(
        'prediction',
        extra={
            'project_id': cfg.Vars.ProjectID,
            'platform': cfg.Vars.Platform,
            'fraud_prob': fraud_prob,
            'fraud_flag': fraud_flag,
            'model_version': model_version,
        },
        stacklevel=2,
    )

    # Best-effort BigQuery logging (no-op on 'loc')
    log_prediction_to_bq(
        {
            'fraud_prob': fraud_prob,
            'fraud_flag': fraud_flag,
            'model_version': model_version,
            'Time': tx.Time,
            'Amount': tx.Amount,
        }
    )

    return PredictionResponse(
        fraud_prob=fraud_prob,
        fraud_flag=fraud_flag,
        model_version=model_version,
        threshold=threshold,
    )

# api/main.py
from __future__ import annotations

from fastapi import Depends, FastAPI

from api.deps import get_cfg
from api.schemas import HealthResponse
from internalpy.config import Cfg, load_vars
from internalpy.log import get_logger

# Load config and logger once at import time.
# load_vars() is already lru_cached inside internalpy.config.
_VARS = load_vars()
_LOGGER = get_logger(_VARS.Platform)

# Test

app = FastAPI(
    title='Fraud Detection API',
    version='0.1.0',
    docs_url='/docs',
    redoc_url='/redoc',
)


@app.on_event('startup')
def on_startup() -> None:
    _LOGGER.info(
        'fraud api starting',
        extra={
            'project_id': _VARS.ProjectID,
            'platform': _VARS.Platform,
            'bq_dataset': _VARS.BQDataset,
            'bq_table_predictions': _VARS.BQTablePredictions,
            'model_path': _VARS.ModelPath,
        },
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

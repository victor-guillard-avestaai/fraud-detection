# api/deps.py
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.cloud import bigquery

import joblib

from internalpy.config import Cfg, build_cfg

logger = logging.getLogger(__name__)


# ----- Config dependency -----


@lru_cache(maxsize=1)
def _load_cfg() -> Cfg:
    return build_cfg()


def get_cfg() -> Cfg:
    """
    FastAPI dependency to access the global, immutable configuration.
    """
    return _load_cfg()


# ----- Model loading -----


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Any
    metadata: dict[str, Any]


def _resolve_model_and_metadata_paths(cfg: Cfg) -> tuple[Path, Path]:
    """
    Resolve model and metadata paths.

    - If ModelPath is absolute, use it as-is.
    - If it's relative, resolve it from project root (parent of api/).
    """
    raw_model_path = Path(cfg.Vars.ModelPath)

    if raw_model_path.is_absolute():
        model_path = raw_model_path
    else:
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / raw_model_path

    metadata_path = model_path.with_name('metadata.json')
    return model_path, metadata_path


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    """
    Load the trained model pipeline and metadata.json once and cache them.
    """
    cfg = get_cfg()
    model_path, metadata_path = _resolve_model_and_metadata_paths(cfg)

    if not model_path.exists():
        raise RuntimeError(f'Model artifact not found at {model_path}')

    if not metadata_path.exists():
        raise RuntimeError(f'Metadata file not found at {metadata_path}')

    logger.info('Loading model from %s', model_path)
    pipeline = joblib.load(model_path)

    logger.info('Loading metadata from %s', metadata_path)
    with metadata_path.open('r', encoding='utf-8') as f:
        metadata = json.load(f)

    return ModelBundle(pipeline=pipeline, metadata=metadata)


@lru_cache(maxsize=1)
def get_demo_samples() -> list[dict[str, Any]]:
    """
    Load demo samples (exported from the test set) from model/demo_samples.json.

    Returns a list of:
      {
        "id": str,
        "label": str,
        "class": 0 or 1,
        "features": { "Time": ..., "V1": ..., ..., "Amount": ... }
      }
    """
    cfg = get_cfg()
    model_path, _ = _resolve_model_and_metadata_paths(cfg)
    demo_path = model_path.parents[1] / 'demo_samples.json'  # /app/model/demo_samples.json

    if not demo_path.exists():
        logger.warning('No demo_samples.json found at %s', demo_path)
        return []

    with demo_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.warning('demo_samples.json content is not a list')
        return []

    return data


# ----- BigQuery logging -----


@lru_cache(maxsize=1)
def get_bq_client(project_id: str) -> bigquery.Client:
    """
    Lazily create a BigQuery client. Not used in 'loc' platform.
    """
    from google.cloud import bigquery  # runtime import

    return bigquery.Client(project=project_id)


def log_prediction_to_bq(record: dict[str, Any]) -> None:
    """
    Best-effort logging of predictions to BigQuery.

    In 'loc' platform, we skip logging entirely.

    record is expected to contain at least:
      - 'fraud_prob'
      - 'fraud_flag'
      - 'model_version'
      - 'Amount' (optional, used to populate 'amount')
      - optionally 'transaction_id' and 'features'
    """
    cfg = get_cfg()

    if cfg.Vars.Platform == 'loc':
        logger.info('Skipping BigQuery logging on local platform')
        return

    table_id = f'{cfg.Vars.ProjectID}.' f'{cfg.Vars.BQDataset}.' f'{cfg.Vars.BQTablePredictions}'

    client = get_bq_client(cfg.Vars.ProjectID)

    # Map our internal record -> BigQuery schema
    row = {
        # BigQuery table schema
        'timestamp': datetime.now(UTC).isoformat(),
        'transaction_id': record.get('transaction_id') or str(uuid.uuid4()),
        'amount': float(record.get('Amount', 0.0)),
        # You can stick features in here later (e.g. JSON string) if you want.
        'features': record.get('features'),
        'model_version': str(record.get('model_version', '')),
        'fraud_prob': float(record.get('fraud_prob', 0.0)),
        'fraud_flag': bool(record.get('fraud_flag', False)),
    }

    try:
        errors = client.insert_rows_json(table_id, [row])
        if errors:
            # Log actual errors so we can debug future issues
            logger.error('BigQuery insert error: %s', errors)
    except Exception:
        logger.exception('Failed to insert prediction log into BigQuery')

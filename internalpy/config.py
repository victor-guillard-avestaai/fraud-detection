# internalpy/config.py
from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import cast

from pydantic import BaseModel, ConfigDict

__all__ = [
    'Vars',
    'Cfg',
    'build_cfg',
    'load_vars',
]


class Vars(BaseModel):
    """
    Central runtime configuration.

    All sensitive values come from environment variables or GCP Secret Manager.
    Nothing is hard-coded in the repo.
    """

    model_config = ConfigDict(frozen=True)

    Platform: str  # "loc" | "dev" | "prod"
    ProjectID: str

    # Model
    ModelPath: str  # e.g. "/app/model/artifacts/model.joblib"

    # BigQuery logging
    BQDataset: str  # e.g. "fraud"
    BQTablePredictions: str  # e.g. "predictions"

    # Optional internal auth token for protected endpoints
    InternalAuth: str | None = None


class Cfg(BaseModel):
    Vars: Vars
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


def _access_secret(project_id: str, name: str) -> str:
    """
    Load secret from GCP Secret Manager.

    This is only used when the environment variable is missing.
    """
    try:
        sm = importlib.import_module('google.cloud.secretmanager')
    except Exception as e:  # pragma: no cover
        raise RuntimeError('google-cloud-secret-manager not installed') from e

    client = sm.SecretManagerServiceClient()
    path = f'projects/{project_id}/secrets/{name}/versions/latest'
    resp = client.access_secret_version(name=path)
    payload = getattr(resp, 'payload', None)
    if payload is None:
        raise RuntimeError(f'secret has no payload: {name}')
    data = getattr(payload, 'data', None)
    if not isinstance(data, (bytes | bytearray)):
        raise RuntimeError(f'secret payload not bytes: {name}')
    return cast(bytes, data).decode('utf-8')


def _get(name: str, project_id: str, *, required: bool = True) -> str:
    """
    Prefer environment variable; otherwise fall back to Secret Manager.
    """
    val = os.getenv(name)
    if val:
        return val
    if not required:
        return ''
    return _access_secret(project_id, name)


def _detect_platform(project_id: str) -> str:
    """
    Platform resolution:
      1) explicit PLATFORM env in {"loc","dev","prod"}
      2) in Cloud Run (K_SERVICE/CLOUD_RUN_SERVICE present): treat as "prod"
      3) else "loc"
    """
    explicit = (os.getenv('PLATFORM') or '').lower()
    if explicit in {'loc', 'dev', 'prod'}:
        return explicit

    in_cloud_run = bool(
        os.getenv('K_SERVICE') or os.getenv('CLOUD_RUN_SERVICE') or os.getenv('CLOUD_RUN_JOB')
    )
    if in_cloud_run:
        # You can refine this later if you actually separate dev/prod projects
        return 'prod'

    return 'loc'


@lru_cache(maxsize=1)
def load_vars() -> Vars:
    """
    Load configuration once, combining .env, environment variables, and Secret Manager.
    """

    # Allow .env in local dev
    try:
        dotenv = importlib.import_module('dotenv')
        dotenv.load_dotenv()
    except Exception:
        pass

    project_id = os.getenv('PROJECT_ID', '').strip()
    if not project_id:
        raise RuntimeError('PROJECT_ID environment variable is not set')

    platform = _detect_platform(project_id)

    # Core non-secret config (should usually be env vars, not secrets)
    model_path = _get('MODEL_PATH', project_id, required=True)
    bq_dataset = _get('BQ_DATASET', project_id, required=True)
    bq_table_predictions = _get('BQ_TABLE_PREDICTIONS', project_id, required=True)

    # Optional secret: internal auth token
    # In Cloud Run, you probably want this in Secret Manager via INTERNAL_AUTH_TOKEN
    internal_auth = os.getenv('INTERNAL_AUTH_TOKEN') or ''
    if not internal_auth:
        # try Secret Manager only if not set in env
        try:
            internal_auth = _get('INTERNAL_AUTH_TOKEN', project_id, required=False).strip()
        except Exception:
            internal_auth = ''

    return Vars(
        Platform=platform,
        ProjectID=project_id,
        ModelPath=model_path,
        BQDataset=bq_dataset,
        BQTablePredictions=bq_table_predictions,
        InternalAuth=internal_auth or None,
    )


def build_cfg() -> Cfg:
    vars_ = load_vars()
    return Cfg(Vars=vars_)

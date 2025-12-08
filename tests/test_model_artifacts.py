from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _get_project_root() -> Path:
    # tests/ is at the same level as api/, model/, etc.
    return Path(__file__).resolve().parents[1]


def test_model_artifacts_exist_and_load() -> None:
    root = _get_project_root()
    model_path = root / 'model' / 'artifacts' / 'model.joblib'
    metadata_path = root / 'model' / 'artifacts' / 'metadata.json'

    assert model_path.exists(), f'Missing model artifact at {model_path}'
    assert metadata_path.exists(), f'Missing metadata at {metadata_path}'

    model = joblib.load(model_path)
    with metadata_path.open('r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Basic shape checks
    assert 'features' in metadata, "Metadata must contain 'features' key"
    assert isinstance(metadata['features'], list)
    n_features = len(metadata['features'])
    assert n_features > 0, 'Metadata features list is empty'

    # Threshold sanity check
    threshold = metadata.get('threshold')
    assert isinstance(threshold, (float | int)), 'threshold must be numeric'
    assert 0.0 < float(threshold) < 1.0, 'threshold must be in (0, 1)'

    # Metrics structure sanity check
    metrics = metadata.get('metrics', {})
    for split in ('validation', 'test'):
        assert split in metrics, f"Missing '{split}' metrics in metadata"
        for key in ('pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'expected_cost'):
            assert key in metrics[split], f"Missing metric '{key}' in metrics['{split}']"

    # Predict_proba sanity check
    X_dummy = np.zeros((5, n_features), dtype=float)
    X_dummy_df = pd.DataFrame(X_dummy, columns=metadata['features'])
    proba = model.predict_proba(X_dummy_df)
    assert proba.shape == (5, 2), 'predict_proba should output (n_samples, 2)'


def test_cost_matrix_present_and_valid() -> None:
    root = _get_project_root()
    metadata_path = root / 'model' / 'artifacts' / 'metadata.json'

    with metadata_path.open('r', encoding='utf-8') as f:
        metadata = json.load(f)

    cost_matrix = metadata.get('cost_matrix')
    assert isinstance(cost_matrix, dict), 'cost_matrix must be a dict'

    assert 'FN' in cost_matrix and 'FP' in cost_matrix, 'Expect FN and FP keys'
    fn_cost = float(cost_matrix['FN'])
    fp_cost = float(cost_matrix['FP'])

    assert fn_cost > 0, 'FN cost should be > 0'
    assert fp_cost > 0, 'FP cost should be > 0'
    # Optional: check FN is larger than FP
    assert fn_cost > fp_cost, 'FN cost should be > FP cost'

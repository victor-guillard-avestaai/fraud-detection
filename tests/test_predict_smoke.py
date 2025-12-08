# tests/test_predict_smoke.py
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_predict_smoke() -> None:
    payload = {
        'Time': 0.0,
        'Amount': 100.0,
    }
    for i in range(1, 29):
        payload[f'V{i}'] = 0.0

    resp = client.post('/predict', json=payload)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert 'fraud_prob' in data
    assert 'fraud_flag' in data
    assert 'model_version' in data
    assert 'threshold' in data

    assert 0.0 <= data['fraud_prob'] <= 1.0

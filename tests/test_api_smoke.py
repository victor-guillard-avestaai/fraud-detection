# tests/test_api_smoke.py
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_health_works() -> None:
    client = TestClient(app)
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert data['status'] == 'ok'
    assert 'project_id' in data
    assert 'platform' in data

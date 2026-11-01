from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from cpa.config import Settings, get_settings

client = TestClient(app)


def test_health_returns_ok_when_database_reachable() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "reachable"}


def test_health_returns_503_when_database_unreachable() -> None:
    # Port 59999 on localhost refuses the connection immediately -- fast
    # and deterministic, unlike a real timeout that would flake depending
    # on the environment.
    unreachable = Settings(
        database_url="postgresql://cpa:cpa_local_dev@127.0.0.1:59999/cpa"
    )
    app.dependency_overrides[get_settings] = lambda: unreachable

    try:
        response = client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503

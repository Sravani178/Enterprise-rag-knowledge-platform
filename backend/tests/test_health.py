from fastapi.testclient import TestClient

from app.main import app


def test_liveness() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_exposes_docs_and_health() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"
    assert response.json()["health"] == "/api/v1/health"

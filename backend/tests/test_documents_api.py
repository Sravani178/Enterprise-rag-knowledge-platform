from fastapi.testclient import TestClient

from app.main import app


def test_upload_rejects_non_pdf_before_storage_or_database() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 415

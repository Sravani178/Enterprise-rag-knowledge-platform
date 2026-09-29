"""Exercise the running Docker stack from upload through grounded query."""

from __future__ import annotations

import os
import time
from uuid import UUID

import httpx
import pymupdf

BASE_URL = os.getenv("SMOKE_BASE_URL", "http://localhost:8000").rstrip("/")
ORGANIZATION_ID = os.getenv(
    "SMOKE_ORGANIZATION_ID",
    "00000000-0000-0000-0000-000000000001",
)
READY_TIMEOUT_SECONDS = int(os.getenv("SMOKE_READY_TIMEOUT_SECONDS", "60"))
PROCESSING_TIMEOUT_SECONDS = int(os.getenv("SMOKE_PROCESSING_TIMEOUT_SECONDS", "120"))


def create_fixture_pdf() -> bytes:
    document = pymupdf.open()
    try:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "FY2025 revenue was 1250 crore. The result came from stronger enterprise sales.",
            fontsize=16,
        )
        return document.tobytes()
    finally:
        document.close()


def wait_for_ready(client: httpx.Client) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last_error = "service did not become ready"
    while time.monotonic() < deadline:
        try:
            response = client.get(f"{BASE_URL}/api/v1/health/ready")
            if response.status_code == 200:
                return
            last_error = f"readiness returned HTTP {response.status_code}: {response.text}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(last_error)


def wait_for_completion(client: httpx.Client, document_id: UUID) -> dict[str, object]:
    deadline = time.monotonic() + PROCESSING_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = client.get(
            f"{BASE_URL}/api/v1/documents/{document_id}",
            headers={"X-Organization-ID": ORGANIZATION_ID},
        )
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in {"COMPLETED", "FAILED"}:
            return payload
        time.sleep(2)
    raise RuntimeError(f"document {document_id} did not finish processing")


def main() -> None:
    headers = {"X-Organization-ID": ORGANIZATION_ID}
    idempotency_headers = {
        **headers,
        "Idempotency-Key": "docker-smoke-test-v1",
    }
    with httpx.Client(timeout=10) as client:
        wait_for_ready(client)

        pdf_bytes = create_fixture_pdf()
        upload = client.post(
            f"{BASE_URL}/api/v1/documents",
            headers=idempotency_headers,
            files={"file": ("smoke-test.pdf", pdf_bytes, "application/pdf")},
        )
        upload.raise_for_status()
        document_id = UUID(upload.json()["id"])

        replay = client.post(
            f"{BASE_URL}/api/v1/documents",
            headers=idempotency_headers,
            files={"file": ("smoke-test.pdf", pdf_bytes, "application/pdf")},
        )
        replay.raise_for_status()
        if UUID(replay.json()["id"]) != document_id:
            raise RuntimeError("Idempotency replay created a second document")

        document = wait_for_completion(client, document_id)
        if document["status"] != "COMPLETED":
            raise RuntimeError(f"ingestion failed: {document}")
        if int(document["chunk_count"]) < 1:
            raise RuntimeError(f"ingestion produced no chunks: {document}")

        query = client.post(
            f"{BASE_URL}/api/v1/query",
            headers=headers,
            json={"query": "What was FY2025 revenue?", "top_k": 5},
        )
        query.raise_for_status()
        result = query.json()
        citations = result.get("citations", [])
        if not citations:
            raise RuntimeError(f"query returned no citations: {result}")
        if citations[0]["document_name"] != "smoke-test.pdf":
            raise RuntimeError(f"unexpected citation: {result}")
        if citations[0]["page"] != 1:
            raise RuntimeError(f"unexpected citation page: {result}")
        if "1250 crore" not in result["answer"]:
            raise RuntimeError(f"answer was not grounded in the fixture: {result}")

        deletion = client.delete(
            f"{BASE_URL}/api/v1/documents/{document_id}",
            headers=headers,
        )
        deletion.raise_for_status()
        if deletion.json()["status"] != "DELETED":
            raise RuntimeError(f"document was not marked deleted: {deletion.json()}")

        after_delete = client.get(
            f"{BASE_URL}/api/v1/documents/{document_id}",
            headers=headers,
        )
        if after_delete.status_code != 404:
            raise RuntimeError("deleted document remained readable")

        print(
            "Smoke test passed: upload -> Celery ingestion -> PostgreSQL/Qdrant indexing "
            "-> hybrid query -> citation."
        )


if __name__ == "__main__":
    main()

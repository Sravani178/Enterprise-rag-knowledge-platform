import json
import logging

from fastapi.testclient import TestClient

from app.core.observability import JsonFormatter, metrics, reset_request_id, set_request_id
from app.main import app


def test_http_requests_return_and_log_a_safe_request_id() -> None:
    metrics.reset()

    with TestClient(app) as client:
        response = client.get("/", headers={"X-Request-ID": "trace-123"})
        metric_response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "trace-123"
    assert "http_requests_total" in metric_response.text
    assert 'route="/"' in metric_response.text


def test_invalid_request_id_is_replaced() -> None:
    with TestClient(app) as client:
        response = client.get("/", headers={"X-Request-ID": "bad value"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "bad value"
    assert " " not in response.headers["X-Request-ID"]


def test_json_formatter_includes_correlation_id_without_request_content() -> None:
    token = set_request_id("trace-456")
    try:
        record = logging.LogRecord(
            "app",
            logging.INFO,
            __file__,
            1,
            "Query completed",
            (),
            None,
        )
        record.event = "query"
        record.fields = {"status": 200}
        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_id(token)

    assert payload["request_id"] == "trace-456"
    assert payload["event"] == "query"
    assert payload["status"] == 200
    assert "query text" not in json.dumps(payload)

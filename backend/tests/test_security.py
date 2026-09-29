"""Adversarial security and tenant-isolation checks for the API boundaries."""

from uuid import uuid4

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.deps import get_organization_id
from app.api.routes.auth import LoginRequest, RegisterRequest
from app.api.routes.query import QueryRequest
from app.auth import Principal, TokenError
from app.auth.service import decode_access_token
from app.core.config import Settings
from app.main import _safe_request_id, app
from app.models import MembershipRole


def test_production_cannot_start_without_authentication_or_rate_limiting() -> None:
    with pytest.raises(ValueError, match="AUTH_REQUIRED"):
        Settings(environment="production", auth_required=False, rate_limit_enabled=True)

    with pytest.raises(ValueError, match="RATE_LIMIT_ENABLED"):
        Settings(
            environment="production",
            auth_required=True,
            rate_limit_enabled=False,
            jwt_secret="a" * 64,
        )


def test_auth_enabled_rejects_default_or_unsupported_jwt_configuration() -> None:
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings(auth_required=True, jwt_secret="development-only-change-this-secret")

    with pytest.raises(ValueError, match="JWT_ALGORITHM"):
        Settings(auth_required=True, jwt_secret="a" * 64, jwt_algorithm="none")


def test_jwt_decoder_requires_expiration_and_issued_at_claims() -> None:
    settings = Settings(auth_required=True, jwt_secret="a" * 64)
    token = jwt.encode(
        {
            "sub": str(uuid4()),
            "org": str(uuid4()),
            "role": MembershipRole.MEMBER.value,
            "email": "user@example.com",
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(TokenError):
        from unittest.mock import patch

        with patch("app.auth.service.get_settings", return_value=settings):
            decode_access_token(token)


def test_tenant_header_cannot_override_authenticated_principal() -> None:
    organization_id = uuid4()
    principal = Principal(
        uuid4(), organization_id, MembershipRole.MEMBER, "user@example.com"
    )

    with pytest.raises(HTTPException) as error:
        import asyncio

        asyncio.run(get_organization_id(uuid4(), principal))

    assert error.value.status_code == 403


def test_public_input_boundaries_reject_blank_or_oversized_values() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(
            email="   ",
            password="correct horse battery staple",
            full_name="User",
            organization_name="Acme",
        )
    with pytest.raises(ValidationError):
        LoginRequest(email="   ", password="password")
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 2001)
    with pytest.raises(ValidationError):
        QueryRequest(query="valid", top_k=21)


def test_request_id_accepts_safe_values_and_replaces_header_injection_attempts() -> None:
    assert _safe_request_id("trace-123_abc.4") == "trace-123_abc.4"
    assert _safe_request_id("bad\r\nX-Injected: true") != "bad\r\nX-Injected: true"


def test_security_headers_and_request_id_are_returned_by_api() -> None:
    with TestClient(app) as client:
        response = client.get("/", headers={"X-Request-ID": "security-test"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "security-test"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Permissions-Policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )

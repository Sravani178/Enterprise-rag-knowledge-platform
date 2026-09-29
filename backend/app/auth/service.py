from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app.core.config import get_settings
from app.models import MembershipRole


class TokenError(ValueError):
    """Raised when an access token is invalid or expired."""


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: UUID
    organization_id: UUID
    role: MembershipRole
    email: str


password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, encoded: str) -> bool:
    try:
        return password_hasher.verify(encoded, password)
    except (InvalidHashError, VerifyMismatchError):
        return False


def create_access_token(principal: Principal) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": str(principal.user_id),
        "org": str(principal.organization_id),
        "role": principal.role.value,
        "email": principal.email,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, str]:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={
                "require": [
                    "exp",
                    "iat",
                    "sub",
                    "org",
                    "role",
                    "email",
                ]
            },
        )
    except jwt.PyJWTError as exc:
        raise TokenError("Invalid or expired access token") from exc
    required = {"sub", "org", "role", "email"}
    if not required.issubset(payload):
        raise TokenError("Access token is missing required claims")
    return {key: str(payload[key]) for key in required}

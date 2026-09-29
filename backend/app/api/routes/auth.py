import re
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.rate_limit import enforce_auth_rate_limit
from app.auth import Principal, create_access_token, hash_password, verify_password
from app.core.config import get_settings
from app.db.session import get_db
from app.models import Membership, MembershipRole, Organization, User

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=128)
    full_name: str = Field(min_length=1, max_length=255)
    organization_name: str = Field(min_length=1, max_length=255)

    @field_validator("email", "full_name", "organization_name")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must contain non-whitespace characters")
        return normalized


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)
    organization_id: UUID | None = None

    @field_validator("email")
    @classmethod
    def email_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("email must contain non-whitespace characters")
        return normalized


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    organization_id: UUID
    role: MembershipRole


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:90] or f"organization-{uuid4().hex[:8]}"


def _token_response(principal: Principal) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(principal),
        expires_in=settings.access_token_expire_minutes * 60,
        organization_id=principal.organization_id,
        role=principal.role,
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
    _rate_limit: None = Depends(enforce_auth_rate_limit),
) -> TokenResponse:
    email = request.email.strip().lower()
    existing = await db.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email is already registered")

    organization = Organization(
        name=request.organization_name.strip(),
        slug=f"{_slugify(request.organization_name)}-{uuid4().hex[:8]}",
    )
    user = User(
        email=email,
        password_hash=await run_in_threadpool(hash_password, request.password),
        full_name=request.full_name.strip(),
    )
    db.add(organization)
    db.add(user)
    await db.flush()
    db.add(
        Membership(
            organization_id=organization.id,
            user_id=user.id,
            role=MembershipRole.OWNER,
        )
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Unable to create organization") from exc

    return _token_response(Principal(user.id, organization.id, MembershipRole.OWNER, email))


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
    _rate_limit: None = Depends(enforce_auth_rate_limit),
) -> TokenResponse:
    email = request.email.strip().lower()
    user = await db.scalar(select(User).where(User.email == email, User.is_active.is_(True)))
    password_valid = (
        user is not None
        and await run_in_threadpool(verify_password, request.password, user.password_hash)
    )
    if not password_valid:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    membership_query = select(Membership).where(Membership.user_id == user.id)
    if request.organization_id is not None:
        membership_query = membership_query.where(
            Membership.organization_id == request.organization_id
        )
    membership = (await db.execute(membership_query)).scalars().first()
    if membership is None:
        raise HTTPException(status_code=403, detail="Organization membership is required")

    return _token_response(
        Principal(user.id, membership.organization_id, membership.role, user.email)
    )

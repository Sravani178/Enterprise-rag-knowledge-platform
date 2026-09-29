from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import Principal, TokenError
from app.auth.service import decode_access_token
from app.core.config import get_settings
from app.db.session import get_db
from app.models import Membership, MembershipRole, User

settings = get_settings()


async def get_current_principal(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> Principal | None:
    if not settings.auth_required:
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer access token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = decode_access_token(authorization[7:].strip())
        user_id = UUID(claims["sub"])
        organization_id = UUID(claims["org"])
        token_role = MembershipRole(claims["role"])
    except (TokenError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    result = await db.execute(
        select(User.email, User.is_active, Membership.role)
        .join(Membership, Membership.user_id == User.id)
        .where(
            User.id == user_id,
            User.is_active.is_(True),
            Membership.organization_id == organization_id,
        )
    )
    membership = result.one_or_none()
    if membership is None:
        raise HTTPException(status_code=403, detail="Organization membership is required")
    role = MembershipRole(membership.role)
    if role != token_role:
        raise HTTPException(status_code=401, detail="Access token role is stale")
    return Principal(user_id, organization_id, role, str(membership.email))


async def get_organization_id(
    x_organization_id: UUID | None = Header(default=None),
    principal: Principal | None = Depends(get_current_principal),
) -> UUID:
    if principal is not None:
        if x_organization_id is not None and x_organization_id != principal.organization_id:
            raise HTTPException(status_code=403, detail="Organization header does not match token")
        return principal.organization_id
    """Use a stable demo tenant while local authentication is disabled."""
    return x_organization_id or settings.default_organization_id


def require_roles(*allowed_roles: MembershipRole):
    async def dependency(
        principal: Principal | None = Depends(get_current_principal),
    ) -> Principal | None:
        if principal is None:
            return None
        if principal.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Insufficient organization permissions")
        return principal

    return dependency

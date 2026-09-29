from datetime import UTC, datetime
from functools import lru_cache
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from redis import Redis
from redis.exceptions import RedisError

from app.api.deps import get_organization_id
from app.core.config import get_settings
from app.core.redis import create_redis_client


@lru_cache(maxsize=1)
def get_rate_limit_client() -> Redis:
    settings = get_settings()
    return create_redis_client(
        settings.celery_broker_url,
        decode_responses=True,
    )


def _increment(client: Redis, key: str) -> int:
    count = int(client.incr(key))
    if count == 1:
        client.expire(key, 61)
    return count


async def enforce_rate_limit(
    organization_id: UUID = Depends(get_organization_id),
) -> None:
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return

    bucket = int(datetime.now(UTC).timestamp() // 60)
    key = f"rate-limit:organization:{organization_id}:{bucket}"
    try:
        count = await run_in_threadpool(_increment, get_rate_limit_client(), key)
    except RedisError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate-limit service is unavailable",
        ) from exc
    if count > settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )


async def enforce_auth_rate_limit(request: Request) -> None:
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return

    client_host = request.client.host if request.client else "unknown"
    bucket = int(datetime.now(UTC).timestamp() // 60)
    key = f"rate-limit:auth:{client_host}:{bucket}"
    try:
        count = await run_in_threadpool(_increment, get_rate_limit_client(), key)
    except RedisError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate-limit service is unavailable",
        ) from exc
    if count > settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )

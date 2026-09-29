import asyncio
import time
from collections.abc import AsyncIterator
from functools import lru_cache
from uuid import uuid4

from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool
from redis import Redis
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.core.redis import create_redis_client

settings = get_settings()
_query_slots = asyncio.Semaphore(settings.query_concurrency_limit)
_DISTRIBUTED_KEY = "query:concurrency:active"
_ACQUIRE_SCRIPT = """
local now = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local expiry = tonumber(ARGV[3])
local token = ARGV[4]
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) >= limit then
    return 0
end
redis.call('ZADD', KEYS[1], now + expiry, token)
redis.call('EXPIRE', KEYS[1], math.ceil(expiry / 1000) + 1)
return 1
"""


@lru_cache(maxsize=1)
def get_query_concurrency_client() -> Redis:
    return create_redis_client(
        settings.query_concurrency_redis_url,
        decode_responses=True,
    )


def _acquire_distributed_slot(token: str) -> bool:
    now_ms = int(time.time() * 1000)
    expiry_ms = settings.query_concurrency_lease_seconds * 1000
    result = get_query_concurrency_client().eval(
        _ACQUIRE_SCRIPT,
        1,
        _DISTRIBUTED_KEY,
        now_ms,
        settings.query_concurrency_limit,
        expiry_ms,
        token,
    )
    return bool(result)


def _release_distributed_slot(token: str) -> None:
    get_query_concurrency_client().zrem(_DISTRIBUTED_KEY, token)


async def enforce_query_concurrency() -> AsyncIterator[None]:
    """Bound query work locally and, when available, across API replicas."""

    try:
        await asyncio.wait_for(
            _query_slots.acquire(),
            timeout=settings.query_concurrency_wait_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query capacity is temporarily full",
            headers={"Retry-After": "5"},
        ) from exc

    token = uuid4().hex
    distributed_acquired = False
    try:
        if settings.query_concurrency_distributed_enabled:
            deadline = time.monotonic() + settings.query_concurrency_wait_seconds
            redis_available = True
            while time.monotonic() < deadline:
                try:
                    distributed_acquired = await run_in_threadpool(
                        _acquire_distributed_slot,
                        token,
                    )
                except RedisError:
                    # Redis is not required for query correctness. The local
                    # semaphore still protects each API process during outage.
                    redis_available = False
                    break
                if distributed_acquired:
                    break
                await asyncio.sleep(0.05)
            if redis_available and not distributed_acquired:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Query capacity is temporarily full",
                    headers={"Retry-After": "5"},
                )
        yield
    finally:
        if distributed_acquired:
            try:
                await run_in_threadpool(_release_distributed_slot, token)
            except RedisError:
                pass
        _query_slots.release()

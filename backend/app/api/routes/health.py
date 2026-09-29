from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.redis import create_redis_client
from app.db.session import get_db
from app.storage import S3Storage, StorageError
from app.vectorstore import QdrantVectorStore, VectorStoreError

router = APIRouter(prefix="/health", tags=["health"])
settings = get_settings()


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    database: Literal["ok", "unavailable"]
    redis: Literal["ok", "unavailable"]
    storage: Literal["ok", "unavailable"]
    vector_store: Literal["ok", "unavailable"]


@lru_cache(maxsize=1)
def get_health_redis() -> Redis:
    return create_redis_client(
        settings.celery_broker_url,
    )


@router.get("/live", response_model=dict[str, str])
async def liveness() -> dict[str, str]:
    """Report whether the API process is running."""

    return {"status": "ok", "service": settings.app_name}


@router.get("/ready", response_model=HealthResponse)
async def readiness(db: AsyncSession = Depends(get_db)) -> HealthResponse | JSONResponse:
    """Report whether the API can reach its transactional database."""

    checks = {
        "database": "ok",
        "redis": "ok",
        "storage": "ok",
        "vector_store": "ok",
    }
    try:
        await db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        checks["database"] = "unavailable"

    try:
        await run_in_threadpool(get_health_redis().ping)
    except RedisError:
        checks["redis"] = "unavailable"

    try:
        await run_in_threadpool(S3Storage().check_connection)
    except StorageError:
        checks["storage"] = "unavailable"

    try:
        await run_in_threadpool(QdrantVectorStore().check_connection)
    except VectorStoreError:
        checks["vector_store"] = "unavailable"

    is_healthy = all(value == "ok" for value in checks.values())
    response = HealthResponse(
        status="ok" if is_healthy else "degraded",
        service=settings.app_name,
        **checks,
    )
    if not is_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )
    return response


@router.get("", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse | JSONResponse:
    """Backward-compatible readiness endpoint for simple probes."""

    return await readiness(db)

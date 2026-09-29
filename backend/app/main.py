from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes.auth import router as auth_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.core.observability import (
    configure_logging,
    logger,
    metrics,
    reset_request_id,
    set_request_id,
)
from app.db.session import dispose_engine

settings = get_settings()
configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await dispose_engine()


app = FastAPI(
    title=settings.app_name,
    description="Secure foundation for document-grounded enterprise question answering.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_request_id(value: str | None) -> str:
    candidate = (value or "").strip()
    if 1 <= len(candidate) <= 128 and all(
        character.isalnum() or character in {"-", "_", "."} for character in candidate
    ):
        return candidate
    return uuid4().hex


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = _safe_request_id(request.headers.get("X-Request-ID"))
    token = set_request_id(request_id)
    started_at = perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    except Exception:
        metrics.increment("http_request_errors_total", labels={"method": request.method})
        logger.exception(
            "Unhandled request failure",
            extra={"event": "http_request_error", "fields": {"method": request.method}},
        )
        raise
    finally:
        latency_ms = (perf_counter() - started_at) * 1000
        route = request.scope.get("route")
        route_name = getattr(route, "path", "unmatched")
        status_code = response.status_code if response is not None else 500
        labels = {
            "method": request.method,
            "route": route_name,
            "status": status_code,
        }
        metrics.increment("http_requests_total", labels=labels)
        metrics.observe("http_request_duration_ms", latency_ms, labels=labels)
        logger.info(
            "HTTP request completed",
            extra={
                "event": "http_request",
                "fields": {
                    "method": request.method,
                    "route": route_name,
                    "status": status_code,
                    "latency_ms": round(latency_ms, 2),
                },
            },
        )
        if response is not None:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        reset_request_id(token)


@app.get("/metrics", include_in_schema=False, response_class=PlainTextResponse)
async def metrics_endpoint() -> str:
    return metrics.render()


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(_: Request, __: SQLAlchemyError) -> JSONResponse:
    """Convert connection/transaction failures into a safe dependency error."""

    return JSONResponse(
        status_code=503,
        content={"detail": "Database service is unavailable"},
    )

app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(auth_router, prefix=settings.api_v1_prefix)
app.include_router(documents_router, prefix=settings.api_v1_prefix)
app.include_router(query_router, prefix=settings.api_v1_prefix)


@app.get("/", tags=["system"])
async def root() -> dict[str, Any]:
    return {
        "service": settings.app_name,
        "environment": settings.environment,
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
    }

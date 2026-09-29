from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "enterprise-knowledge",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=settings.celery_soft_time_limit_seconds,
    task_time_limit=settings.celery_time_limit_seconds,
    task_default_queue="default",
    broker_pool_limit=settings.celery_broker_pool_limit,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "max_connections": settings.redis_max_connections,
        "socket_connect_timeout": settings.redis_connect_timeout_seconds,
        "socket_timeout": settings.redis_socket_timeout_seconds,
    },
    redis_max_connections=settings.redis_max_connections,
    redis_backend_health_check_interval=30,
    redis_socket_connect_timeout=settings.redis_connect_timeout_seconds,
    redis_socket_timeout=settings.redis_socket_timeout_seconds,
    result_expires=86400,
    beat_schedule={
        "reconcile-document-indexes": {
            "task": "documents.reconcile",
            "schedule": 300.0,
        },
    },
)

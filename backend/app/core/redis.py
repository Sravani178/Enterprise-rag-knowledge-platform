from redis import Redis
from redis.connection import BlockingConnectionPool

from app.core.config import get_settings


def create_redis_client(url: str, *, decode_responses: bool = False) -> Redis:
    """Create a bounded Redis client whose pool waits for a finite time.

    A blocking pool prevents every API feature from creating its own unbounded
    set of sockets. The finite pool timeout turns exhaustion into a normal
    RedisError that the caller can handle using its existing fail-open or
    service-unavailable policy.
    """

    settings = get_settings()
    pool = BlockingConnectionPool.from_url(
        url,
        decode_responses=decode_responses,
        max_connections=settings.redis_max_connections,
        timeout=settings.redis_pool_timeout_seconds,
        socket_connect_timeout=settings.redis_connect_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
    )
    return Redis(connection_pool=pool)

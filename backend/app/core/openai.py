import time
from typing import Any

import httpx

from app.core.config import get_settings


def post_with_retry(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> httpx.Response:
    """Retry transient OpenAI network/provider failures with bounded backoff."""

    settings = get_settings()
    response: httpx.Response | None = None
    retryable_statuses = {408, 429, 500, 502, 503, 504}
    for attempt in range(settings.openai_max_retries + 1):
        try:
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=settings.llm_timeout_seconds,
            )
        except httpx.RequestError:
            if attempt >= settings.openai_max_retries:
                raise
        else:
            if (
                response.status_code not in retryable_statuses
                or attempt >= settings.openai_max_retries
            ):
                return response
        retry_after = 0.0
        if response is not None:
            try:
                retry_after = max(0.0, float(response.headers.get("Retry-After", "0")))
            except (TypeError, ValueError):
                retry_after = 0.0
        backoff = settings.openai_retry_backoff_seconds * (2**attempt)
        time.sleep(min(max(backoff, retry_after), 30.0))

    if response is None:
        raise httpx.TimeoutException("OpenAI request did not return a response")
    return response

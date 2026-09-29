import json
import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime

request_id_context: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> object:
    return request_id_context.set(request_id)


def reset_request_id(token: object) -> None:
    request_id_context.reset(token)  # type: ignore[arg-type]


class JsonFormatter(logging.Formatter):
    """Emit structured application logs without logging request contents."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_context.get(),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging() -> None:
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)


logger = logging.getLogger("app")


@dataclass(frozen=True)
class MetricKey:
    name: str
    labels: tuple[tuple[str, str], ...]


class MetricsRegistry:
    """Small thread-safe Prometheus text registry for process-local metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[MetricKey, float] = {}
        self._observations: dict[MetricKey, tuple[int, float]] = {}

    @staticmethod
    def _key(name: str, labels: dict[str, object] | None) -> MetricKey:
        normalized = tuple(
            sorted((str(key), str(value)) for key, value in (labels or {}).items())
        )
        return MetricKey(name, normalized)

    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        labels: dict[str, object] | None = None,
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, object] | None = None,
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            count, total = self._observations.get(key, (0, 0.0))
            self._observations[key] = (count + 1, total + value)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._observations.clear()

    @staticmethod
    def _labels(labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""

        def escape(value: str) -> str:
            return value.replace("\\", "\\\\").replace('"', '\\"')

        encoded = ",".join(
            f'{key}="{escape(value)}"'
            for key, value in labels
        )
        return "{" + encoded + "}"

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = sorted(self._counters.items(), key=lambda item: item[0])
            observations = sorted(self._observations.items(), key=lambda item: item[0])
        for key, value in counters:
            lines.append(f"{key.name}{self._labels(key.labels)} {value:g}")
        for key, (count, total) in observations:
            labels = self._labels(key.labels)
            lines.append(f"{key.name}_count{labels} {count}")
            lines.append(f"{key.name}_sum{labels} {total:g}")
        return "\n".join(lines) + ("\n" if lines else "")


metrics = MetricsRegistry()

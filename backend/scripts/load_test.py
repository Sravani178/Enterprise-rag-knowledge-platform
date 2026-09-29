"""Bounded async load test for the API.

Examples:
    python backend/scripts/load_test.py --path /api/v1/health/live
    python backend/scripts/load_test.py --in-process --path /metrics --requests-per-user 2
    python backend/scripts/load_test.py --path /api/v1/query --method POST \
        --payload '{"query":"What was FY2025 revenue?"}' --allow-large-load
"""

import argparse
import asyncio
import json
import logging
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import httpx

DEFAULT_LEVELS = (10, 100, 1000)


@dataclass(frozen=True)
class ScenarioResult:
    users: int
    requests_per_user: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    throughput_requests_per_second: float
    duration_seconds: float
    latency_ms: dict[str, float]
    status_codes: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


async def run_scenario(
    client: httpx.AsyncClient,
    *,
    path: str,
    method: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str],
    users: int,
    requests_per_user: int,
) -> ScenarioResult:
    """Run one synchronized virtual-user scenario against an HTTP client."""

    start_gate = asyncio.Event()
    latencies: list[float] = []
    status_codes: Counter[str] = Counter()
    lock = asyncio.Lock()

    async def request_once() -> None:
        await start_gate.wait()
        started_at = time.perf_counter()
        status = "exception"
        try:
            response = await client.request(
                method,
                path,
                json=payload,
                headers=headers,
            )
            status = str(response.status_code)
        except Exception:
            pass
        latency = (time.perf_counter() - started_at) * 1000
        async with lock:
            latencies.append(latency)
            status_codes[status] += 1

    async def virtual_user() -> None:
        await asyncio.gather(*(request_once() for _ in range(requests_per_user)))

    virtual_users = [asyncio.create_task(virtual_user()) for _ in range(users)]
    started_at = time.perf_counter()
    start_gate.set()
    await asyncio.gather(*virtual_users)
    duration = max(time.perf_counter() - started_at, 0.000001)

    total = len(latencies)
    successful = sum(
        count for code, count in status_codes.items() if code.isdigit() and int(code) < 400
    )
    failed = total - successful
    return ScenarioResult(
        users=users,
        requests_per_user=requests_per_user,
        total_requests=total,
        successful_requests=successful,
        failed_requests=failed,
        error_rate=failed / total if total else 0.0,
        throughput_requests_per_second=total / duration,
        duration_seconds=duration,
        latency_ms={
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies, default=0.0),
        },
        status_codes=dict(sorted(status_codes.items())),
    )


async def run_levels(
    *,
    base_url: str,
    path: str,
    method: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str],
    levels: tuple[int, ...],
    requests_per_user: int,
    timeout_seconds: float,
    in_process: bool,
    suppress_app_logs: bool = False,
) -> list[ScenarioResult]:
    transport = None
    previous_app_log_level: int | None = None
    if in_process:
        from app.main import app

        transport = httpx.ASGITransport(app=app)
        if suppress_app_logs:
            app_logger = logging.getLogger("app")
            previous_app_log_level = app_logger.level
            app_logger.setLevel(logging.CRITICAL)
    limits = httpx.Limits(
        max_connections=max(levels),
        max_keepalive_connections=max(levels),
    )
    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            limits=limits,
            transport=transport,
        ) as client:
            results: list[ScenarioResult] = []
            for users in levels:
                result = await run_scenario(
                    client,
                    path=path,
                    method=method,
                    payload=payload,
                    headers=headers,
                    users=users,
                    requests_per_user=requests_per_user,
                )
                results.append(result)
            return results
    finally:
        if previous_app_log_level is not None:
            logging.getLogger("app").setLevel(previous_app_log_level)


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        name, separator, content = value.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"Invalid header: {value!r}; expected Name=Value")
        headers[name.strip()] = content
    return headers


def _parse_levels(value: str) -> tuple[int, ...]:
    levels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not levels or any(level <= 0 or level > 1000 for level in levels):
        raise ValueError("levels must contain values between 1 and 1000")
    return levels


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded API load scenarios")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--path", default="/api/v1/health/live")
    parser.add_argument("--method", default="GET", choices=("GET", "POST"))
    parser.add_argument("--payload", help="JSON object sent as the request body")
    parser.add_argument("--header", action="append", default=[], help="Request header Name=Value")
    parser.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)))
    parser.add_argument("--requests-per-user", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--allow-large-load", action="store_true")
    parser.add_argument("--json-output", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    levels = _parse_levels(args.levels)
    if args.requests_per_user <= 0 or args.requests_per_user > 100:
        raise SystemExit("requests-per-user must be between 1 and 100")
    if max(levels) > 100 and not args.allow_large_load:
        raise SystemExit("Use --allow-large-load for scenarios above 100 users")
    try:
        payload = json.loads(args.payload) if args.payload else None
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        headers = _parse_headers(args.header)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    results = asyncio.run(
        run_levels(
            base_url=args.base_url,
            path=args.path,
            method=args.method,
            payload=payload,
            headers=headers,
            levels=levels,
            requests_per_user=args.requests_per_user,
            timeout_seconds=args.timeout_seconds,
            in_process=args.in_process,
            suppress_app_logs=args.in_process and args.json_output,
        )
    )
    if args.json_output:
        print(json.dumps([result.as_dict() for result in results], indent=2))
        return
    for result in results:
        print(
            f"users={result.users:<4} requests={result.total_requests:<5} "
            f"errors={result.failed_requests:<5} error_rate={result.error_rate:.2%} "
            f"throughput={result.throughput_requests_per_second:.2f}/s "
            f"p50={result.latency_ms['p50']:.1f}ms "
            f"p95={result.latency_ms['p95']:.1f}ms "
            f"p99={result.latency_ms['p99']:.1f}ms"
        )


if __name__ == "__main__":
    main()

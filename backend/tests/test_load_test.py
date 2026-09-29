import asyncio

import pytest

from scripts.load_test import _parse_levels, percentile, run_scenario


def test_percentile_interpolates_latency_values() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 25.0
    assert percentile([], 0.95) == 0.0


def test_load_scenario_reports_concurrency_and_statuses() -> None:
    class FakeResponse:
        status_code = 200

    class FakeClient:
        calls = 0

        async def request(self, method: str, path: str, **kwargs: object) -> FakeResponse:
            self.calls += 1
            await asyncio.sleep(0)
            return FakeResponse()

    result = asyncio.run(
        run_scenario(
            FakeClient(),
            path="/health",
            method="GET",
            payload=None,
            headers={},
            users=10,
            requests_per_user=2,
        )
    )

    assert result.total_requests == 20
    assert result.successful_requests == 20
    assert result.failed_requests == 0
    assert result.status_codes == {"200": 20}
    assert result.throughput_requests_per_second > 0


def test_load_scenario_counts_exceptions_as_failures() -> None:
    class FailingClient:
        async def request(self, method: str, path: str, **kwargs: object) -> object:
            raise TimeoutError("request timed out")

    result = asyncio.run(
        run_scenario(
            FailingClient(),
            path="/health",
            method="GET",
            payload=None,
            headers={},
            users=2,
            requests_per_user=1,
        )
    )

    assert result.total_requests == 2
    assert result.failed_requests == 2
    assert result.status_codes == {"exception": 2}
    assert result.error_rate == 1.0


def test_load_levels_are_bounded() -> None:
    assert _parse_levels("10, 100, 1000") == (10, 100, 1000)
    with pytest.raises(ValueError):
        _parse_levels("1001")

import asyncio
import time

from databricks_apps_telemetry._event_loop import EventLoopMonitor
from databricks_apps_telemetry._peak import PeakTracker


def test_peak_survives_burst_between_exports(otel) -> None:
    tracker = PeakTracker()
    tracker.register(otel.meter_provider.get_meter("t"), "test.inflight.peak", "peak")
    attrs = {"model": "m"}
    for _ in range(30):
        tracker.enter(attrs)
    for _ in range(30):
        tracker.exit(attrs)  # burst finished before the export
    assert otel.collect()["test.inflight.peak"] == [(30, attrs)]
    # next export: nothing in flight, peak resets to the current value
    assert otel.collect()["test.inflight.peak"] == [(0, attrs)]
    tracker.enter(attrs)
    assert otel.collect()["test.inflight.peak"] == [(1, attrs)]


async def test_event_loop_lag_detects_blocking_code(otel) -> None:
    monitor = EventLoopMonitor(otel.meter_provider.get_meter("t"), interval_seconds=0.05)
    monitor.start()
    await asyncio.sleep(0.2)
    healthy = otel.collect()["app.event_loop.lag.peak"][0][0]
    time.sleep(0.4)  # noqa: ASYNC251 - deliberately block the loop, like a sync generator in an async handler
    await asyncio.sleep(0.1)
    blocked = otel.collect()["app.event_loop.lag.peak"][0][0]
    assert healthy < 0.05
    assert blocked >= 0.3

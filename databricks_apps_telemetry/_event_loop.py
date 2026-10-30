"""asyncio event-loop lag: the earliest sign that something is blocking an async server.

A coroutine sleeps for a fixed interval and measures how late it wakes up. On a healthy
loop the lateness is ~0; when a handler runs synchronous work on the loop (a sync
generator, a blocking HTTP call, CPU-bound code), every other request on that worker
waits, and the lateness grows by exactly that much.

  app.event_loop.lag.peak   gauge (s)  worst lateness since the previous export
"""

import asyncio
import threading
import time
from collections.abc import Iterable

from opentelemetry import metrics
from opentelemetry.metrics import CallbackOptions, Observation


class EventLoopMonitor:
    """Samples one event loop's lag and exports the worst value per export interval."""

    def __init__(self, meter: metrics.Meter, interval_seconds: float = 0.1) -> None:
        self.interval_seconds: float = interval_seconds
        self._lock: threading.Lock = threading.Lock()
        self._peak: float | None = None
        self._task: asyncio.Task[None] | None = None
        meter.create_observable_gauge(
            "app.event_loop.lag.peak",
            [self._observe],
            unit="s",
            description="Worst asyncio event-loop lag since the previous export",
        )

    def start(self) -> None:
        """Start sampling on the running loop. Idempotent; call from inside the loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run(), name="event-loop-lag-monitor")

    async def _run(self) -> None:
        while True:
            started: float = time.perf_counter()
            await asyncio.sleep(self.interval_seconds)
            lag: float = max(time.perf_counter() - started - self.interval_seconds, 0.0)
            with self._lock:
                self._peak = lag if self._peak is None else max(self._peak, lag)

    def _observe(self, options: CallbackOptions) -> Iterable[Observation]:
        with self._lock:
            peak, self._peak = self._peak, None
        return [] if peak is None else [Observation(peak)]

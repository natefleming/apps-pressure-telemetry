"""Concurrency tracking that does not miss short bursts.

A gauge is sampled once per metric export. A burst that starts and ends between two
exports (e.g. 30 LLM calls finishing in 3 s with a 10 s export interval) reads 0 at
both samples. PeakTracker keeps the high-water mark since the previous export instead.
"""

import threading
from collections.abc import Iterable, Mapping

from opentelemetry import metrics
from opentelemetry.metrics import CallbackOptions, Observation


class PeakTracker:
    """Counts in-progress work per attribute set and reports the peak between exports."""

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._current: dict[tuple[tuple[str, str], ...], int] = {}
        self._peak: dict[tuple[tuple[str, str], ...], int] = {}

    @staticmethod
    def _key(attributes: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(attributes.items()))

    def enter(self, attributes: Mapping[str, str]) -> None:
        key = self._key(attributes)
        with self._lock:
            current: int = self._current.get(key, 0) + 1
            self._current[key] = current
            self._peak[key] = max(self._peak.get(key, 0), current)

    def exit(self, attributes: Mapping[str, str]) -> None:
        key = self._key(attributes)
        with self._lock:
            self._current[key] = max(self._current.get(key, 0) - 1, 0)

    def observe_peak(self, options: CallbackOptions) -> Iterable[Observation]:
        """Report each series' peak since the last export, then reset it to the current value."""
        with self._lock:
            observations: list[Observation] = [Observation(peak, dict(key)) for key, peak in self._peak.items()]
            self._peak = dict(self._current)
        return observations

    def register(self, meter: metrics.Meter, name: str, description: str, unit: str = "{request}") -> None:
        meter.create_observable_gauge(name, [self.observe_peak], unit=unit, description=description)

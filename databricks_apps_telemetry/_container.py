"""Container-level CPU / memory / disk metrics read from the process's own cgroup.

`opentelemetry-instrumentation-system-metrics` reports `system.*` for the node the
app runs on and `process.*` for one process. Neither answers "how close is this app
to its limits?". The container's cgroup does: it holds the CPU quota and memory
limit the platform enforces, and usage across every process in the container.

Databricks Apps currently mounts cgroup v1 (read-only) at /sys/fs/cgroup; v2 is
handled too. Outside a container (or on macOS/Windows) the cgroup files are absent and
the instruments simply report nothing.

Exported instruments (all prefixed `container.`):
  cpu.limit             gauge    cores the container may use (quota / period)
  cpu.time              counter  cumulative CPU seconds used by all processes
  cpu.utilization       gauge    CPU used since the last export / limit (0-1)
  cpu.throttled.time    counter  cumulative seconds the container was CPU-throttled
  memory.usage          gauge    bytes charged to the cgroup (includes page cache)
  memory.working_set    gauge    usage minus inactive file cache (what OOM/eviction track)
  memory.limit          gauge    memory limit in bytes
  memory.utilization    gauge    working_set / limit (0-1)
  memory.oom_kills      counter  OOM kills recorded by the cgroup
  filesystem.usage      gauge    bytes used on the root filesystem
  filesystem.capacity   gauge    size of the root filesystem
  filesystem.utilization gauge   usage / capacity (0-1)
"""

import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from opentelemetry import metrics
from opentelemetry.metrics import CallbackOptions, Observation

CGROUP_ROOT: Path = Path("/sys/fs/cgroup")


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class CgroupReader:
    """Reads CPU and memory accounting for the current container from cgroup v1 or v2."""

    def __init__(self, root: Path = CGROUP_ROOT) -> None:
        self.root: Path = root
        self.v2: bool = (root / "cgroup.controllers").exists()

    def _read(self, *candidates: str) -> str | None:
        for rel in candidates:
            try:
                return (self.root / rel).read_text().strip()
            except OSError:
                continue
        return None

    def _stat(self, *candidates: str) -> dict[str, int]:
        text: str | None = self._read(*candidates)
        stats: dict[str, int] = {}
        for line in (text or "").splitlines():
            key, _, value = line.partition(" ")
            if value.strip().lstrip("-").isdigit():
                stats[key] = int(value)
        return stats

    def cpu_limit_cores(self) -> float:
        if self.v2:
            quota, _, period = (self._read("cpu.max") or "max 100000").partition(" ")
            if quota != "max":
                return int(quota) / int(period)
        else:
            quota_us: str | None = self._read("cpu/cpu.cfs_quota_us", "cpu,cpuacct/cpu.cfs_quota_us")
            period_us: str | None = self._read("cpu/cpu.cfs_period_us", "cpu,cpuacct/cpu.cfs_period_us")
            if quota_us and period_us and int(quota_us) > 0:
                return int(quota_us) / int(period_us)
        return float(_available_cpus())

    def cpu_usage_seconds(self) -> float | None:
        if self.v2:
            usec: int | None = self._stat("cpu.stat").get("usage_usec")
            return usec / 1e6 if usec is not None else None
        ns: str | None = self._read("cpuacct/cpuacct.usage", "cpu,cpuacct/cpuacct.usage")
        return int(ns) / 1e9 if ns else None

    def cpu_throttled_seconds(self) -> float | None:
        if self.v2:
            usec: int | None = self._stat("cpu.stat").get("throttled_usec")
            return usec / 1e6 if usec is not None else None
        ns: int | None = self._stat("cpu/cpu.stat", "cpu,cpuacct/cpu.stat").get("throttled_time")
        return ns / 1e9 if ns is not None else None

    def memory(self) -> dict[str, int | None]:
        if self.v2:
            usage: str | None = self._read("memory.current")
            limit: str | None = self._read("memory.max")
            inactive: int = self._stat("memory.stat").get("inactive_file", 0)
            oom: int | None = self._stat("memory.events").get("oom_kill")
        else:
            usage = self._read("memory/memory.usage_in_bytes")
            limit = self._read("memory/memory.limit_in_bytes")
            inactive = self._stat("memory/memory.stat").get("total_inactive_file", 0)
            oom = self._stat("memory/memory.oom_control").get("oom_kill")
        usage_b: int | None = int(usage) if usage and usage.isdigit() else None
        limit_b: int | None = int(limit) if limit and limit.isdigit() else None
        # cgroup v1 reports "no limit" as a huge sentinel (~2^63)
        if limit_b is not None and limit_b >= 1 << 60:
            limit_b = None
        working_set: int | None = max(usage_b - inactive, 0) if usage_b is not None else None
        return {"usage": usage_b, "working_set": working_set, "limit": limit_b, "oom_kills": oom}


class ContainerMetrics:
    """Registers observable instruments that sample the cgroup on every metric export."""

    def __init__(
        self,
        meter: metrics.Meter,
        reader: CgroupReader | None = None,
        path: str = "/",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.reader: CgroupReader = reader or CgroupReader()
        self.path: str = path
        self._clock: Callable[[], float] = clock
        self._lock: threading.Lock = threading.Lock()
        self._last_cpu: tuple[float, float] | None = None  # (monotonic time, cpu seconds)

        meter.create_observable_gauge(
            "container.cpu.limit", [self._cpu_limit], unit="{cpu}", description="CPU cores the container may use"
        )
        meter.create_observable_counter(
            "container.cpu.time",
            [self._cpu_time],
            unit="s",
            description="CPU seconds used by all processes in the container",
        )
        meter.create_observable_gauge(
            "container.cpu.utilization",
            [self._cpu_utilization],
            unit="1",
            description="CPU used since the previous export divided by the CPU limit",
        )
        meter.create_observable_counter(
            "container.cpu.throttled.time",
            [self._cpu_throttled],
            unit="s",
            description="Seconds the container was throttled by its CPU quota",
        )
        meter.create_observable_gauge(
            "container.memory.usage",
            [self._mem("usage")],
            unit="By",
            description="Bytes charged to the container cgroup, including page cache",
        )
        meter.create_observable_gauge(
            "container.memory.working_set",
            [self._mem("working_set")],
            unit="By",
            description="Memory usage minus inactive file cache",
        )
        meter.create_observable_gauge(
            "container.memory.limit", [self._mem("limit")], unit="By", description="Container memory limit"
        )
        meter.create_observable_gauge(
            "container.memory.utilization",
            [self._mem_utilization],
            unit="1",
            description="Working set divided by the memory limit",
        )
        meter.create_observable_counter(
            "container.memory.oom_kills",
            [self._mem("oom_kills")],
            unit="1",
            description="OOM kills recorded by the container cgroup",
        )
        meter.create_observable_gauge(
            "container.filesystem.usage", [self._fs("used")], unit="By", description="Bytes used on the root filesystem"
        )
        meter.create_observable_gauge(
            "container.filesystem.capacity", [self._fs("total")], unit="By", description="Size of the root filesystem"
        )
        meter.create_observable_gauge(
            "container.filesystem.utilization",
            [self._fs_utilization],
            unit="1",
            description="Root filesystem used / capacity",
        )

    def _cpu_limit(self, options: CallbackOptions) -> Iterable[Observation]:
        yield Observation(self.reader.cpu_limit_cores())

    def _cpu_time(self, options: CallbackOptions) -> Iterable[Observation]:
        seconds: float | None = self.reader.cpu_usage_seconds()
        if seconds is not None:
            yield Observation(seconds)

    def _cpu_utilization(self, options: CallbackOptions) -> Iterable[Observation]:
        seconds: float | None = self.reader.cpu_usage_seconds()
        if seconds is None:
            return
        now: float = self._clock()
        with self._lock:
            last, self._last_cpu = self._last_cpu, (now, seconds)
        if last is not None and now > last[0]:
            used_cores: float = (seconds - last[1]) / (now - last[0])
            yield Observation(max(used_cores, 0.0) / self.reader.cpu_limit_cores())

    def _cpu_throttled(self, options: CallbackOptions) -> Iterable[Observation]:
        seconds: float | None = self.reader.cpu_throttled_seconds()
        if seconds is not None:
            yield Observation(seconds)

    def _mem(self, key: str):
        def callback(options: CallbackOptions) -> Iterable[Observation]:
            value: int | None = self.reader.memory()[key]
            if value is not None:
                yield Observation(value)

        return callback

    def _mem_utilization(self, options: CallbackOptions) -> Iterable[Observation]:
        mem: dict[str, int | None] = self.reader.memory()
        if mem["working_set"] is not None and mem["limit"]:
            yield Observation(mem["working_set"] / mem["limit"])

    def _fs(self, field: str):
        def callback(options: CallbackOptions) -> Iterable[Observation]:
            yield Observation(getattr(shutil.disk_usage(self.path), field))

        return callback

    def _fs_utilization(self, options: CallbackOptions) -> Iterable[Observation]:
        usage = shutil.disk_usage(self.path)
        yield Observation(usage.used / usage.total)

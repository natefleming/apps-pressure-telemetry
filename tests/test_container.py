from pathlib import Path

import pytest

from databricks_apps_telemetry._container import CgroupReader, ContainerMetrics

GIB: int = 1024**3


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


@pytest.fixture
def cgroup_v1(tmp_path: Path) -> Path:
    # Shape observed in a MEDIUM Databricks App: 2 vCPU quota, 8 GiB limit
    return _write(
        tmp_path,
        {
            "cpu/cpu.cfs_quota_us": "200000\n",
            "cpu/cpu.cfs_period_us": "100000\n",
            "cpu/cpu.stat": "nr_periods 33268\nnr_throttled 1\nthrottled_time 5716302\n",
            "cpuacct/cpuacct.usage": "587217687224\n",
            "memory/memory.usage_in_bytes": str(2 * GIB),
            "memory/memory.limit_in_bytes": str(8 * GIB),
            "memory/memory.stat": f"total_rss {GIB}\ntotal_inactive_file {GIB // 2}\n",
            "memory/memory.oom_control": "oom_kill_disable 0\nunder_oom 0\noom_kill 3\n",
        },
    )


@pytest.fixture
def cgroup_v2(tmp_path: Path) -> Path:
    return _write(
        tmp_path,
        {
            "cgroup.controllers": "cpu memory io pids",
            "cpu.max": "150000 100000",
            "cpu.stat": "usage_usec 4000000\nuser_usec 3000000\nthrottled_usec 250000\n",
            "memory.current": str(3 * GIB),
            "memory.max": str(4 * GIB),
            "memory.stat": f"anon {GIB}\ninactive_file {GIB}\n",
            "memory.events": "low 0\nhigh 0\nmax 0\noom 1\noom_kill 1\n",
        },
    )


def test_v1_limits_and_usage(cgroup_v1: Path) -> None:
    reader = CgroupReader(cgroup_v1)
    assert not reader.v2
    assert reader.cpu_limit_cores() == 2.0
    assert reader.cpu_usage_seconds() == pytest.approx(587.217687224)
    assert reader.cpu_throttled_seconds() == pytest.approx(0.005716302)
    mem = reader.memory()
    assert mem == {"usage": 2 * GIB, "working_set": 2 * GIB - GIB // 2, "limit": 8 * GIB, "oom_kills": 3}


def test_v2_limits_and_usage(cgroup_v2: Path) -> None:
    reader = CgroupReader(cgroup_v2)
    assert reader.v2
    assert reader.cpu_limit_cores() == 1.5
    assert reader.cpu_usage_seconds() == 4.0
    assert reader.cpu_throttled_seconds() == 0.25
    assert reader.memory() == {"usage": 3 * GIB, "working_set": 2 * GIB, "limit": 4 * GIB, "oom_kills": 1}


def test_unlimited_memory_and_missing_cgroup(tmp_path: Path) -> None:
    _write(tmp_path, {"memory/memory.usage_in_bytes": "100", "memory/memory.limit_in_bytes": str(2**63 - 4096)})
    assert CgroupReader(tmp_path).memory()["limit"] is None
    empty = CgroupReader(tmp_path / "nothing")
    assert empty.cpu_usage_seconds() is None
    assert empty.memory() == {"usage": None, "working_set": None, "limit": None, "oom_kills": None}
    assert empty.cpu_limit_cores() >= 1


def test_metrics_export_fractions_of_limit(otel, cgroup_v1: Path) -> None:
    now = [100.0]  # fake monotonic clock: deterministic, independent of how fast the test host is
    ContainerMetrics(otel.meter_provider.get_meter("t"), reader=CgroupReader(cgroup_v1), clock=lambda: now[0])
    first = otel.collect()
    assert first["container.memory.utilization"][0][0] == pytest.approx(1.5 / 8)
    assert first["container.memory.limit"][0][0] == 8 * GIB
    assert first["container.cpu.limit"][0][0] == 2.0
    assert "container.cpu.utilization" not in first  # needs two samples
    assert 0 <= first["container.filesystem.utilization"][0][0] <= 1

    # 2 s later the container has used 2 more CPU-seconds: one full core of a 2-core limit -> 50%
    now[0] += 2.0
    (cgroup_v1 / "cpuacct/cpuacct.usage").write_text(str(int((587.217687224 + 2.0) * 1e9)))
    assert otel.collect()["container.cpu.utilization"][0][0] == pytest.approx(0.5)
    # next 2 s: 1 more CPU-second -> half a core -> 25%
    now[0] += 2.0
    (cgroup_v1 / "cpuacct/cpuacct.usage").write_text(str(int((587.217687224 + 3.0) * 1e9)))
    assert otel.collect()["container.cpu.utilization"][0][0] == pytest.approx(0.25)

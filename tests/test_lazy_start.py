"""Lazy start and WSGI support, each in a fresh interpreter (OTel global providers are set-once)."""

import json
import subprocess
import sys
import textwrap


def _run(code: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], check=False, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_supervisor_import_does_not_start_telemetry() -> None:
    # What a uvicorn/gunicorn supervisor does: import the module that builds and instruments the app,
    # but never serve. No providers, no exporter threads, nothing patched.
    result = _run("""
        import json, threading
        import openai
        from fastapi import FastAPI
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from databricks_apps_telemetry import instrument

        app = FastAPI()
        t = instrument(app)
        print(json.dumps({
            "started": t.started,
            "sdk_meter_provider": isinstance(metrics.get_meter_provider(), MeterProvider),
            "openai_patched": getattr(openai.resources.chat.completions.Completions.create, "__databricks_apps_telemetry_wrapped__", False),
            "features": t.features,
        }))
    """)
    assert result == {
        "started": False,
        "sdk_meter_provider": False,
        "openai_patched": False,
        "features": ["http(FastAPI)"],
    }


def test_serving_process_starts_on_lifespan_and_custom_meter_forwards() -> None:
    result = _run("""
        import json
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from databricks_apps_telemetry import instrument

        reader = InMemoryMetricReader()
        app = FastAPI()

        @app.get("/hello")
        def hello() -> dict:
            orders.add(1)
            return {"ok": True}

        t = instrument(app, metric_readers=[reader])
        orders = t.meter.create_counter("app.orders")  # created before start: must forward afterwards
        with TestClient(app) as client:                  # runs ASGI lifespan startup, like uvicorn
            started_at_lifespan = t.started
            client.get("/hello"); client.get("/assets/index.js"); client.get("/health")
        names = {m.name: m for rm in reader.get_metrics_data().resource_metrics
                 for sm in rm.scope_metrics for m in sm.metrics}
        ttfb = names["app.http.time_to_first_byte"].data.data_points
        print(json.dumps({
            "started_at_lifespan": started_at_lifespan,
            "orders": names["app.orders"].data.data_points[0].value,
            "ttfb_routes": sorted(p.attributes["http.route"] for p in ttfb),
            "has_container": "container.filesystem.utilization" in names,
        }))
    """)
    assert result["started_at_lifespan"] is True
    assert result["orders"] == 1
    assert result["ttfb_routes"] == ["/hello"]  # /assets/* and /health are not app load
    assert result["has_container"] is True


def test_agent_server_object_is_accepted() -> None:
    # Duck-typed: anything named AgentServer exposing `.app` (avoids importing mlflow in the test suite)
    result = _run("""
        import json
        from fastapi import FastAPI
        from databricks_apps_telemetry import instrument

        class AgentServer:
            def __init__(self):
                self.app = FastAPI()

        server = AgentServer()
        t = instrument(server)
        instrument(server.app)  # idempotent: same app, no second middleware
        ours = [m for m in server.app.user_middleware if m.cls.__name__ == "RequestPressureMiddleware"]
        print(json.dumps({"features": t.features, "middleware": len(ours)}))
    """)
    assert result == {"features": ["http(FastAPI)"], "middleware": 1}


def test_flask_and_dash_style_apps() -> None:
    result = _run("""
        import json
        from flask import Flask, Response
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from databricks_apps_telemetry import instrument

        reader = InMemoryMetricReader()
        flask_app = Flask("demo")

        @flask_app.get("/stream")
        def stream():
            def gen():
                yield b"first"
                yield b"second"
            return Response(gen())

        class Dash:  # Dash exposes its Flask app as .server
            def __init__(self, server):
                self.server = server

        t = instrument(Dash(flask_app), metric_readers=[reader])
        before = t.started
        client = flask_app.test_client()
        for _ in range(3):
            assert client.get("/stream").data == b"firstsecond"
        names = {m.name: m for rm in reader.get_metrics_data().resource_metrics
                 for sm in rm.scope_metrics for m in sm.metrics}
        print(json.dumps({
            "started_before_request": before,
            "started_after": t.started,
            "features": t.features,
            "ttfb_count": names["app.http.time_to_first_byte"].data.data_points[0].count,
            "inflight_peak": max(p.value for p in names["app.http.requests.inflight.peak"].data.data_points),
        }))
    """)
    assert result["started_before_request"] is False and result["started_after"] is True
    assert result["features"][0] == "http(Flask)"
    assert result["ttfb_count"] == 3
    assert result["inflight_peak"] == 1


def test_exclude_metrics_drops_matching_metrics_only() -> None:
    result = _run("""
        import json
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from databricks_apps_telemetry import instrument

        reader = InMemoryMetricReader()
        instrument(metric_readers=[reader], exclude_metrics=["system.*", "process.runtime.*", "container.memory.*"])
        names = {m.name for rm in reader.get_metrics_data().resource_metrics for sm in rm.scope_metrics for m in sm.metrics}
        print(json.dumps(sorted(names)))
    """)
    assert not [n for n in result if n.startswith(("system.", "process.runtime.", "container.memory."))]
    assert "container.filesystem.utilization" in result and "container.cpu.limit" in result
    # process.* kept, only process.runtime.* dropped (metric names vary across instrumentation versions)
    assert [n for n in result if n.startswith("process.") and not n.startswith("process.runtime.")]

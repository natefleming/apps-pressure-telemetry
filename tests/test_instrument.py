"""End-to-end: instrument() on a FastAPI app, the way an app (or AgentServer) would call it.

instrument() configures process-global providers once, so everything lives in one module.
"""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from databricks_apps_telemetry import instrument

READER = InMemoryMetricReader()
SPANS = InMemorySpanExporter()

app = FastAPI()


@app.get("/stream")
async def stream() -> StreamingResponse:
    async def gen():
        await asyncio.sleep(0.15)  # e.g. waiting for the first LLM token
        yield b"data: first\n\n"
        await asyncio.sleep(0.05)
        yield b"data: second\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


TELEMETRY = instrument(app, service_name="test-app", metric_readers=[READER], span_exporters=[SPANS])


def _metrics() -> dict[str, list]:
    out: dict[str, list] = {}
    for rm in READER.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                out.setdefault(metric.name, []).extend(metric.data.data_points)
    return out


async def test_instrument_wires_every_layer() -> None:
    assert TELEMETRY.features == ["http(FastAPI)"] and not TELEMETRY.started  # lazy until it serves
    assert instrument(app) is TELEMETRY  # idempotent per process

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        responses = await asyncio.gather(*(client.get("/stream") for _ in range(5)))
        await client.get("/health")
    assert all(r.status_code == 200 for r in responses)
    assert TELEMETRY.started  # started by the first request (httpx's ASGITransport sends no lifespan)
    features = " ".join(TELEMETRY.features)
    for feature in ("container", "system", "openai", "event_loop", "http"):
        assert feature in features

    m = _metrics()
    ttfb = [p for p in m["app.http.time_to_first_byte"] if p.attributes.get("http.route") == "/stream"]
    assert ttfb[0].count == 5 and ttfb[0].sum / 5 >= 0.14
    assert max(p.value for p in m["app.http.requests.inflight.peak"]) == 5
    assert "app.event_loop.lag.peak" in m
    assert "http.server.duration" in m or "http.server.request.duration" in m

    TELEMETRY.tracer_provider.force_flush()
    server_spans = [
        s for s in SPANS.get_finished_spans() if s.attributes.get("http.route") == "/stream" and s.kind.name == "SERVER"
    ]
    assert len(server_spans) == 5
    assert server_spans[0].attributes["app.http.time_to_first_byte_ms"] >= 140
    assert not [s for s in SPANS.get_finished_spans() if s.attributes.get("http.route") == "/health"]
    assert server_spans[0].resource.attributes["service.name"] == "test-app"


def test_container_metrics_are_safe_without_cgroups() -> None:
    # The test host (macOS / CI) may have no cgroup files: instruments must just stay quiet
    m = _metrics()
    assert "container.filesystem.utilization" in m
    pytest.importorskip("opentelemetry.instrumentation.system_metrics")

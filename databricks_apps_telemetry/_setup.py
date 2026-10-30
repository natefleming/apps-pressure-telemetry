"""`instrument()`: one call that turns on every pressure signal for a Databricks App.

With App telemetry enabled, Databricks Apps injects OTEL_EXPORTER_OTLP_ENDPOINT (a local collector
that writes to the app's Unity Catalog OTEL tables), OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES,
and captures stdout/stderr into otel_logs itself - so this module only sets up metrics and traces.
Without an OTLP endpoint (local runs, tests) nothing is exported unless readers/exporters are passed.

Startup is lazy when an app is passed: `instrument(app)` only wires middleware, and the OpenTelemetry
providers, exporter threads and samplers start in the first process that actually serves a request
(at ASGI lifespan startup, or the first WSGI request). So it is safe to call in a module that the
uvicorn/gunicorn supervisor also imports (e.g. an AgentServer `start_server.py` run with --workers N):
the supervisor never serves, never starts exporting, and workers don't inherit exporter threads.
"""

import asyncio
import logging
import os
import socket
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from databricks_apps_telemetry._container import ContainerMetrics
from databricks_apps_telemetry._event_loop import EventLoopMonitor

logger: logging.Logger = logging.getLogger("databricks_apps_telemetry")

INSTRUMENTATION_NAME: str = "databricks_apps_telemetry"
OTLP_ENDPOINT_VARS: tuple[str, ...] = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
)
# Health checks and static assets (e.g. the AgentServer chat UI's /assets/*) are not app load
EXCLUDED_URLS: str = "health,healthz,ping,favicon.ico,/assets/,/static/"


@dataclass
class _Options:
    service_name: str | None
    export_interval_seconds: float
    container: bool
    system: bool
    openai: bool
    event_loop: bool
    metric_readers: Sequence[MetricReader] | None
    span_exporters: Sequence[SpanExporter] | None


@dataclass
class Telemetry:
    """Handle returned by `instrument()`. `meter` / `tracer` work immediately for custom signals;
    they forward to the real providers once telemetry starts in the serving process."""

    features: list[str] = field(default_factory=list)
    started: bool = False

    @property
    def meter(self) -> metrics.Meter:
        return metrics.get_meter(INSTRUMENTATION_NAME)

    @property
    def tracer(self) -> trace.Tracer:
        return trace.get_tracer(INSTRUMENTATION_NAME)

    @property
    def meter_provider(self) -> metrics.MeterProvider:
        return metrics.get_meter_provider()

    @property
    def tracer_provider(self) -> trace.TracerProvider:
        return trace.get_tracer_provider()

    def start(self) -> "Telemetry":
        """Start exporting in this process now (normally automatic on the first request)."""
        start()
        return self

    def shutdown(self) -> None:
        """Flush and stop exporting (normally done automatically at process exit)."""
        for provider in (self.meter_provider, self.tracer_provider):
            shutdown = getattr(provider, "shutdown", None)
            if callable(shutdown):
                shutdown()


_lock: threading.RLock = threading.RLock()
_options: _Options | None = None
_telemetry: Telemetry = Telemetry()
_loop_monitor: EventLoopMonitor | None = None


def _resource(service_name: str | None) -> Resource:
    name: str = (
        service_name
        or os.environ.get("OTEL_SERVICE_NAME")
        or os.environ.get("DATABRICKS_APP_NAME")
        or "unknown_service"
    )
    # Resource.create also merges OTEL_RESOURCE_ATTRIBUTES (Databricks Apps sets app.name, workspace.id)
    return Resource.create(
        {
            "service.name": name,
            "service.instance.id": str(uuid.uuid4()),  # one per process, i.e. per uvicorn worker
            "host.name": socket.gethostname(),  # one per app instance (container)
            "process.pid": os.getpid(),
        }
    )


def _otlp_configured() -> bool:
    return any(os.environ.get(var) for var in OTLP_ENDPOINT_VARS)


def _meter_provider(
    resource: Resource, export_interval_seconds: float, metric_readers: Sequence[MetricReader] | None
) -> metrics.MeterProvider:
    existing: metrics.MeterProvider = metrics.get_meter_provider()
    if isinstance(existing, MeterProvider) and metric_readers is None:
        return existing  # e.g. running under `opentelemetry-instrument`
    readers: list[MetricReader] = list(metric_readers or [])
    if not readers and _otlp_configured():
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(), export_interval_millis=int(export_interval_seconds * 1000)
            )
        )
    provider: MeterProvider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(provider)
    return provider


def _tracer_provider(resource: Resource, span_exporters: Sequence[SpanExporter] | None) -> trace.TracerProvider:
    existing: trace.TracerProvider = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider) and span_exporters is None:
        return existing
    exporters: list[SpanExporter] = list(span_exporters or [])
    if not exporters and _otlp_configured():
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporters.append(OTLPSpanExporter())
    # Databricks Apps: requests arriving with an unsampled traceparent would otherwise be dropped
    sampler: Any = None if os.environ.get("OTEL_TRACES_SAMPLER") else ALWAYS_ON
    provider: TracerProvider = (
        TracerProvider(resource=resource, sampler=sampler) if sampler else TracerProvider(resource=resource)
    )
    for exporter in exporters:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


def start() -> None:
    """Create providers and process-level instruments in this process. Idempotent."""
    global _loop_monitor
    with _lock:
        if _telemetry.started or _options is None:
            return
        opts: _Options = _options
        resource: Resource = _resource(opts.service_name)
        meter_provider: metrics.MeterProvider = _meter_provider(
            resource, opts.export_interval_seconds, opts.metric_readers
        )
        tracer_provider: trace.TracerProvider = _tracer_provider(resource, opts.span_exporters)
        meter: metrics.Meter = meter_provider.get_meter(INSTRUMENTATION_NAME)
        tracer: trace.Tracer = tracer_provider.get_tracer(INSTRUMENTATION_NAME)
        features: list[str] = _telemetry.features

        if opts.container:
            ContainerMetrics(meter)
            features.append("container")
        if opts.system:
            try:
                from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
            except ImportError:
                logger.warning("system metrics skipped: install apps-pressure-telemetry[system]")
            else:
                SystemMetricsInstrumentor().instrument(meter_provider=meter_provider)
                features.append("system")
        if opts.openai:
            try:
                import openai as _openai_sdk  # noqa: F401
            except ImportError:
                logger.info("openai not installed; LLM call instrumentation skipped")
            else:
                from databricks_apps_telemetry._openai import OpenAIInstrumentation

                patched: list[str] = OpenAIInstrumentation(meter, tracer).instrument()
                features.append(f"openai({', '.join(patched)})")
        if opts.event_loop:
            _loop_monitor = EventLoopMonitor(meter)
            features.append("event_loop")
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass  # not inside a loop (e.g. WSGI or process-only): the monitor stays idle
            else:
                _loop_monitor.start()
        _telemetry.started = True
        logger.info("databricks_apps_telemetry started (pid %s): %s", os.getpid(), ", ".join(features))


def loop_monitor() -> EventLoopMonitor | None:
    return _loop_monitor


def instrument(
    app: Any = None,
    *,
    service_name: str | None = None,
    export_interval_seconds: float = 10.0,
    container: bool = True,
    system: bool = True,
    http: bool = True,
    openai: bool = True,
    event_loop: bool = True,
    metric_readers: Sequence[MetricReader] | None = None,
    span_exporters: Sequence[SpanExporter] | None = None,
) -> Telemetry:
    """Turn on every pressure signal for this process and (optionally) a web app.

    Args:
        app: An MLflow ``AgentServer`` or its ``.app``, any FastAPI / Starlette app, a Flask app, or a
            Dash app. ``None`` = process-level signals only (Streamlit, scripts), started immediately.
        service_name: Defaults to OTEL_SERVICE_NAME, then DATABRICKS_APP_NAME.
        export_interval_seconds: How often metrics are collected and exported (SDK default is 60).
        container: cgroup CPU / memory / disk vs limits (``container.*``).
        system: host + process metrics; needs the ``[system]`` extra.
        http: request concurrency / time to first byte, plus request spans with the ``[fastapi]`` (or
            ``[flask]``) extra.
        openai: instrument the OpenAI SDK, incl. DatabricksOpenAI (``[openai]`` extra or openai installed).
        event_loop: asyncio event-loop lag (ASGI apps).
        metric_readers: replace the default OTLP reader (tests, or extra destinations such as a second
            exporter for Datadog/Grafana). Exporters are only created when an OTLP endpoint is configured.
        span_exporters: same for spans.

    Returns:
        The process-wide Telemetry handle. Options from the first call win; later calls only add apps.
    """
    global _options
    with _lock:
        if _options is None:
            _options = _Options(
                service_name,
                export_interval_seconds,
                container,
                system,
                openai,
                event_loop,
                metric_readers,
                span_exporters,
            )
    if app is None:
        start()
    elif http:
        _instrument_app(app)
    return _telemetry


def _instrument_app(app: Any) -> None:
    from databricks_apps_telemetry._wsgi import is_wsgi_app, wsgi_app_of

    target: Any = getattr(app, "app", None) if type(app).__name__ == "AgentServer" else app
    if wsgi_app_of(target) is not None:  # Dash: the Flask server underneath
        target = wsgi_app_of(target)
    marker: str = "_databricks_apps_telemetry"
    if getattr(target, marker, False):
        return
    if is_wsgi_app(target):
        _instrument_wsgi(target)
    elif hasattr(target, "add_middleware"):
        _instrument_asgi(target)
    else:
        raise TypeError(
            f"instrument() does not know how to wrap {type(app).__name__}; pass an AgentServer, FastAPI, "
            "Starlette, Flask or Dash app - or call instrument() with no app for process-level signals."
        )
    setattr(target, marker, True)
    _telemetry.features.append(f"http({type(target).__name__})")


def _instrument_asgi(app: Any) -> None:
    from databricks_apps_telemetry._http import RequestPressureMiddleware

    # Starlette builds the middleware stack on the first ASGI call (lifespan startup) in the serving
    # process; RequestPressureMiddleware.__init__ calls start() there. It is added before the FastAPI
    # instrumentation wraps the stack, so it runs inside the OpenTelemetry server span.
    app.add_middleware(RequestPressureMiddleware)
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        logger.warning("request spans skipped: install apps-pressure-telemetry[fastapi]")
    else:
        # No providers passed: the instrumentation reads the global ones when the stack is built,
        # i.e. after start() has set them.
        FastAPIInstrumentor.instrument_app(app, excluded_urls=EXCLUDED_URLS)


def _instrument_wsgi(app: Any) -> None:
    from databricks_apps_telemetry._wsgi import WSGIRequestPressureMiddleware

    app.wsgi_app = WSGIRequestPressureMiddleware(app.wsgi_app)
    try:
        from opentelemetry.instrumentation.flask import FlaskInstrumentor
    except ImportError:
        pass  # request spans are optional for WSGI apps ([flask] extra)
    else:
        FlaskInstrumentor().instrument_app(app, excluded_urls=EXCLUDED_URLS)

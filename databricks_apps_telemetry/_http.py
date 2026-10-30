"""Request-level pressure signals as a pure ASGI middleware (safe for streaming responses).

  app.http.requests.inflight.peak   gauge          peak concurrent requests since the previous export
  app.http.time_to_first_byte       histogram (s)  request start -> first response body bytes; for SSE /
                                                   streaming agents this is what the user waits for

It also sets `app.http.time_to_first_byte_ms` on the request's SERVER span when the
FastAPI OpenTelemetry instrumentation is active (the span is current inside this middleware).

Starlette constructs middleware on the first ASGI call in the serving process, so the constructor
is where telemetry starts - never in a supervisor process that only imports the app.
"""

import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from opentelemetry import metrics, trace

from databricks_apps_telemetry._peak import PeakTracker

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

EXCLUDED_PATHS: frozenset[str] = frozenset({"/health", "/healthz", "/ping", "/favicon.ico"})
EXCLUDED_PREFIXES: tuple[str, ...] = ("/assets/", "/static/")  # e.g. the AgentServer chat UI


def excluded(path: str) -> bool:
    return path in EXCLUDED_PATHS or path.startswith(EXCLUDED_PREFIXES)


class RequestTelemetry:
    """Instruments shared by the ASGI and WSGI middlewares."""

    def __init__(self) -> None:
        from databricks_apps_telemetry._setup import INSTRUMENTATION_NAME, start

        start()
        meter: metrics.Meter = metrics.get_meter(INSTRUMENTATION_NAME)
        self.tracker: PeakTracker = PeakTracker()
        self.tracker.register(
            meter, "app.http.requests.inflight.peak", "Peak concurrent HTTP requests since the previous export"
        )
        self.ttfb: metrics.Histogram = meter.create_histogram(
            "app.http.time_to_first_byte",
            unit="s",
            description="Time from request start to the first response body bytes",
        )

    def first_byte(self, started: float, attrs: dict[str, str]) -> None:
        ttfb: float = time.perf_counter() - started
        self.ttfb.record(ttfb, attrs)
        trace.get_current_span().set_attribute("app.http.time_to_first_byte_ms", round(ttfb * 1000, 1))


class RequestPressureMiddleware:
    """Tracks concurrent requests and time to first byte for every HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app: ASGIApp = app
        self.telemetry: RequestTelemetry = RequestTelemetry()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or excluded(scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        from databricks_apps_telemetry._setup import loop_monitor

        monitor = loop_monitor()
        if monitor is not None:
            monitor.start()

        attrs: dict[str, str] = {"http.request.method": scope.get("method", "")}
        started: float = time.perf_counter()
        first_byte_seen: bool = False

        async def send_wrapper(message: Message) -> None:
            nonlocal first_byte_seen
            if not first_byte_seen and message["type"] == "http.response.body" and message.get("body"):
                first_byte_seen = True
                route: str = getattr(scope.get("route"), "path", None) or "unmatched"
                self.telemetry.first_byte(started, {**attrs, "http.route": route})
            await send(message)

        self.telemetry.tracker.enter(attrs)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.telemetry.tracker.exit(attrs)

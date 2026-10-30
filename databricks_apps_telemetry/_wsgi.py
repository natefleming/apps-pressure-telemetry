"""The same request signals for WSGI apps (Flask, Dash).

`instrument(flask_app)` / `instrument(dash_app)` wrap `app.wsgi_app`. Telemetry starts on the first
request, so gunicorn workers each start their own exporters after the fork. Request spans are added
when `opentelemetry-instrumentation-flask` is installed (``[flask]`` extra).
"""

import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from databricks_apps_telemetry._http import RequestTelemetry, excluded

WSGIApp = Callable[[dict[str, Any], Callable[..., Any]], Iterable[bytes]]


def is_wsgi_app(app: Any) -> bool:
    return callable(getattr(app, "wsgi_app", None))


def wsgi_app_of(app: Any) -> Any:
    """Dash and similar wrappers expose the Flask app as `.server`."""
    server: Any = getattr(app, "server", None)
    return server if not is_wsgi_app(app) and is_wsgi_app(server) else None


class WSGIRequestPressureMiddleware:
    """Tracks concurrent requests and time to first byte around a WSGI app."""

    def __init__(self, app: WSGIApp) -> None:
        self.app: WSGIApp = app
        self._telemetry: RequestTelemetry | None = None
        self._lock: threading.Lock = threading.Lock()

    def _ensure(self) -> RequestTelemetry:
        if self._telemetry is None:
            with self._lock:
                if self._telemetry is None:
                    self._telemetry = RequestTelemetry()
        return self._telemetry

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> Iterable[bytes]:
        telemetry: RequestTelemetry = self._ensure()
        if excluded(environ.get("PATH_INFO", "")):
            return self.app(environ, start_response)
        attrs: dict[str, str] = {"http.request.method": environ.get("REQUEST_METHOD", "")}
        started: float = time.perf_counter()
        telemetry.tracker.enter(attrs)
        try:
            body: Iterable[bytes] = self.app(environ, start_response)
        except BaseException:
            telemetry.tracker.exit(attrs)
            raise
        return _Body(body, telemetry, started, attrs)


class _Body:
    """Response iterable that records the first non-empty chunk and finishes on close()."""

    def __init__(self, body: Iterable[bytes], telemetry: RequestTelemetry, started: float, attrs: dict[str, str]):
        self._body: Iterable[bytes] = body
        self._telemetry: RequestTelemetry = telemetry
        self._started: float = started
        self._attrs: dict[str, str] = attrs
        self._closed: bool = False

    def __iter__(self) -> Iterator[bytes]:
        first: bool = True
        try:
            for chunk in self._body:
                if first and chunk:
                    first = False
                    self._telemetry.first_byte(self._started, self._attrs)
                yield chunk
        finally:
            self._finish()  # don't depend on the server calling close()

    def _finish(self) -> None:
        if not self._closed:
            self._closed = True
            self._telemetry.tracker.exit(self._attrs)

    def close(self) -> None:
        self._finish()
        close: Any = getattr(self._body, "close", None)
        if callable(close):
            close()

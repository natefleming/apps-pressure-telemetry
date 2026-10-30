"""Pressure telemetry for Databricks Apps: one call, every signal that says "this app is struggling".

    from databricks_apps_telemetry import instrument

    app = FastAPI()                 # or: app = AgentServer("ResponsesAgent").app
    instrument(app)

With App telemetry enabled on the Databricks App, the signals land in the app's Unity Catalog
OTEL tables (otel_metrics / otel_spans / otel_logs). Anywhere else, point the standard
OTEL_EXPORTER_OTLP_* variables at any OpenTelemetry collector.

  container.*            CPU / memory / disk as a fraction of the container's real cgroup limits,
                         CPU throttling, OOM kills - across every process in the container
  system.*, process.*    host and per-process metrics ([system] extra)
  http.server.*          request duration + one SERVER span per request ([fastapi] extra)
  app.http.*             peak concurrent requests between exports, time to first response byte
  app.event_loop.*       asyncio event-loop lag (blocking code in async handlers shows up here)
  gen_ai.*, app.genai.*  every OpenAI-SDK LLM call (incl. DatabricksOpenAI): duration, time to first
                         token, tokens, errors by type, peak concurrent calls, one CLIENT span per call

Call it at import time, before the app starts serving. With `uvicorn --workers N` each worker
calls it and reports its own series; container.* repeats per worker (aggregate with max, not sum).
"""

from databricks_apps_telemetry._openai import error_type
from databricks_apps_telemetry._setup import Telemetry, instrument

__all__ = ["Telemetry", "__version__", "error_type", "instrument"]
__version__: str = "0.2.0"

# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Docs and example: pin the version on Databricks Apps (Apps skips installation when `requirements.txt`
  is unchanged, so an unpinned requirement never upgrades).

## [0.1.1] - 2026-10-30

- `instrument(..., exclude_metrics=[...])`: drop individual metrics by name or glob pattern, across every layer
  (implemented with OpenTelemetry Views).
- LLM calls made through `with_raw_response` (langchain-openai's `ChatOpenAI.invoke` / `ainvoke`) now record token
  usage. Raw streaming responses and `with_streaming_response` bodies pass through untouched (never read early).
- `export_interval_seconds` now defaults to the standard `OTEL_METRIC_EXPORT_INTERVAL` (milliseconds) when it is set,
  else 10 s. An explicit argument still wins.
- A warning is logged when `instrument()` is called again with different options; previously they were silently
  ignored (the first call's options still apply).
- Docs: "Choosing what's collected" (flag → metrics, `exclude_metrics`, cost guidance, excluded request paths) and
  verified LangChain coverage (`ChatDatabricks`, `ChatOpenAI`).

## [0.1.0] - 2026-10-29

First release.

- `instrument()` for MLflow `AgentServer`, FastAPI / Starlette (ASGI) and Flask / Dash (WSGI) apps, or with no app
  for process-level signals only.
- Container CPU / memory / disk against the container's cgroup limits, CPU throttling and OOM kills.
- Request concurrency (peak between exports), time to first byte and asyncio event-loop lag.
- OpenAI SDK instrumentation (incl. `DatabricksOpenAI`): duration, time to first token, tokens, errors by type,
  peak concurrent calls, and a CLIENT span per call.
- Lazy, supervisor-safe startup for `uvicorn` / `gunicorn` with multiple workers.

# apps-pressure-telemetry

> **Community project — not an official Databricks product.** It is not built, released, endorsed or
> supported by Databricks, Inc. Use it at your own discretion; issues and fixes go through this repo, not
> Databricks support. "Databricks" is a trademark of Databricks, Inc.

One call that makes a FastAPI app on **Databricks Apps** report when it is under pressure:
container CPU / memory / disk **against the app's real limits**, request concurrency, event-loop
lag, time to first byte, and every LLM call's latency, tokens and errors. It works with plain
FastAPI and with MLflow `AgentServer`.

```python
from databricks_apps_telemetry import instrument

app = FastAPI()  # or: app = AgentServer("ResponsesAgent").app
instrument(app)
```

With [App telemetry](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/observability)
enabled on the app, everything lands in the app's Unity Catalog tables (`otel_metrics`,
`otel_spans`), ready for AI/BI dashboards and Databricks SQL alerts.

## Install

```bash
pip install "apps-pressure-telemetry[all]"     # fastapi + system + openai instrumentation
```

The distribution is `apps-pressure-telemetry`; the import package is `databricks_apps_telemetry`:

```python
from databricks_apps_telemetry import instrument
```

| Extra | Adds |
|---|---|
| *(none)* | `container.*`, `app.http.*`, `app.event_loop.*`, OTLP export |
| `[fastapi]` | one SERVER span per request + `http.server.*` metrics |
| `[system]` | `system.*` / `process.*` host and process metrics |
| `[openai]` | the `openai` SDK; LLM instrumentation turns on whenever `openai` is importable |
| `[all]` | everything above |

## Enable on a Databricks App

1. Turn on App telemetry for the app (UI: **Settings → App telemetry**, or in a bundle):

   ```yaml
   resources:
     apps:
       my_app:
         telemetry_export_destinations:
           - unity_catalog:
               logs_table: main.app_obs.otel_logs
               metrics_table: main.app_obs.otel_metrics
               traces_table: main.app_obs.otel_spans
   ```

2. Add `apps-pressure-telemetry[all]` to the app's dependencies, call `instrument(app)` where the app is created,
   and deploy. You don't need any environment variables or an `opentelemetry-instrument` wrapper: Databricks Apps
   injects the collector endpoint and resource attributes.

   **Pin the version** (`apps-pressure-telemetry[all]==0.1.1`) and bump the pin to upgrade: Databricks Apps skips
   dependency installation when `requirements.txt` is unchanged ("Requirements have not changed. Skipping
   installation."), so an unpinned requirement stays on whatever version the first deploy installed.

   Prefer **`pyproject.toml` + `uv.lock`** over `requirements.txt` (Apps then runs `uv sync --locked`). Apps keeps
   the venv between deploys and pip skips an already-installed name+version, so a rebuilt wheel with an unchanged
   version is silently not installed; the lock pins wheel hashes, so uv reinstalls it and removes stale packages.

### MLflow AgentServer

Add one line to the agent template's `start_server.py`:

```python
from databricks_apps_telemetry import instrument
from mlflow.genai.agent_server import AgentServer

import agent_server.agent  # registers @invoke / @stream

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app
instrument(agent_server)  # <- the only telemetry code in the agent


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")
```

That's all, including with `--workers N`. Telemetry starts lazily in each worker when it begins serving,
so the uvicorn supervisor that also runs this module exports nothing, and workers don't inherit exporter
threads. LLM calls through `DatabricksOpenAI` (an `openai.OpenAI` subclass) are covered automatically,
alongside `mlflow.openai.autolog()`. The template's chat UI assets (`/assets/*`) are not counted as load.

## Examples

| Example | What it shows |
|---|---|
| [`examples/agent-server-app`](examples/agent-server-app) | A deployable bundle: an async MLflow AgentServer agent (`AsyncDatabricksOpenAI`, streaming, 2 workers) with App telemetry on, and a job that runs the query notebook |
| [`examples/agent-server-app/notebooks/query_metrics.py`](examples/agent-server-app/notebooks/query_metrics.py) | Example SQL for **every** exported signal: metric inventory, container vs limits, process / host, request concurrency / TTFB / event-loop lag, routes, LLM metrics and span percentiles, per-request LLM breakdown, freshness, and alert-ready queries |

## What you get

| Signal | Metric (unit) | Why it matters |
|---|---|---|
| CPU vs limit | `container.cpu.utilization` (0–1), `container.cpu.limit`, `container.cpu.time` | Read from the container's cgroup, so it covers **all** processes against the real quota |
| CPU throttling | `container.cpu.throttled.time` (s, cumulative) | Rises when the app hits its CPU limit |
| Memory vs limit | `container.memory.working_set` / `.limit` / `.utilization`, `.usage` | The working set is what the OOM killer tracks |
| OOM kills | `container.memory.oom_kills` | Recorded by the cgroup |
| Disk | `container.filesystem.usage` / `.capacity` / `.utilization` | Root filesystem |
| Request concurrency | `app.http.requests.inflight.peak` | Peak between exports, so short bursts aren't missed |
| Time to first byte | `app.http.time_to_first_byte` (s) + span attr `app.http.time_to_first_byte_ms` | What a streaming-agent user waits for |
| **Event-loop lag** | `app.event_loop.lag.peak` (s) | Blocking code in async handlers. Server-side latency can't show this queueing |
| LLM call duration | `gen_ai.client.operation.duration` (s) | OTel GenAI semantic convention |
| Tokens | `gen_ai.client.token.usage` (`gen_ai.token.type` = input/output) | OTel GenAI semantic convention |
| Time to first token | `app.genai.time_to_first_token` (s) | Streaming calls only; measured to the first *visible* text |
| LLM concurrency | `app.genai.requests.inflight`, `app.genai.requests.inflight.peak` | |
| LLM errors | `app.genai.errors` (`error.type` = rate_limited, timeout, http_5xx, …) | |
| Spans | SERVER span per request, CLIENT span `chat <model>` per LLM call | Exact percentiles per request or call |
| Host / process | `system.*`, `process.*` (`[system]`) | `system.*` is the **node**, not the app |

**LLM clients covered:** anything that calls the `openai` SDK's `chat.completions.create` or `responses.create`
(sync or async, streaming or not). That includes `DatabricksOpenAI` / `AsyncDatabricksOpenAI` and LangChain's
`ChatDatabricks` (databricks-langchain) and `ChatOpenAI` (langchain-openai). All four combinations of
`invoke` / `stream` / `ainvoke` / `astream` were verified against a Databricks Foundation Model endpoint.

Metrics are collected every `export_interval_seconds` (default 10 s, or `OTEL_METRIC_EXPORT_INTERVAL` if set; the OTel SDK default is 60 s).
With `--workers N`, each worker reports its own series (`service.instance.id`) and the container
shows up as `host.name`. Aggregate `container.*` with **max** and the concurrency peaks with **sum**.

## Other Python app frameworks

The resource and LLM layers work in any Python app. The request layer depends on the framework:

| Framework | Call | Request signals |
|---|---|---|
| FastAPI / Starlette | `instrument(app)` | concurrency, time to first byte, event-loop lag, spans (`[fastapi]`) |
| Flask / Dash | `instrument(flask_app)` / `instrument(dash_app)` | concurrency, time to first byte; spans with `[flask]` |
| Gradio | mount it on FastAPI (`gr.mount_gradio_app`), then `instrument(fastapi_app)` | as FastAPI (not verified) |
| Streamlit, scripts | `instrument()` (no app) | none (container, system and LLM signals only) |

## Configuration

```python
telemetry = instrument(
    app,
    service_name=None,  # default: OTEL_SERVICE_NAME, then DATABRICKS_APP_NAME
    export_interval_seconds=None,  # default: OTEL_METRIC_EXPORT_INTERVAL (ms) if set, else 10 s
    container=True,
    system=True,
    http=True,
    openai=True,
    event_loop=True,
    exclude_metrics=(),  # names or glob patterns to drop, e.g. ["system.*"]; see "Choosing what's collected"
    metric_readers=None,  # e.g. add a second destination (Datadog/Grafana) or test readers
    span_exporters=None,
)
counter = telemetry.meter.create_counter("app.orders")  # your own signals, same providers
```

`instrument()` is idempotent per process: options from the first call win, and a warning is logged if a later call passes different ones. With an app it only wires
middleware, and providers start on the app's first ASGI event or WSGI request. Without an app it starts
immediately. `telemetry.meter` / `.tracer` can be used right away: instruments created before startup
forward once it starts. If the process already has OpenTelemetry SDK providers (for example under
`opentelemetry-instrument`), they are reused.

## Choosing what's collected

Each layer has a flag. These are the metrics (and spans) each one controls:

| Flag | Metrics | Spans |
|---|---|---|
| `container` | `container.cpu.*`, `container.memory.*`, `container.filesystem.*` | none |
| `system` | `system.*`, `process.*`, `process.runtime.*`, `cpython.gc.*` (`[system]` extra) | none |
| `http` | `app.http.requests.inflight.peak`, `app.http.time_to_first_byte`, `http.server.*` | one SERVER span per request |
| `openai` | `gen_ai.client.operation.duration`, `gen_ai.client.token.usage`, `app.genai.*` | one CLIENT span per LLM call |
| `event_loop` | `app.event_loop.lag.peak` | none |

To drop **individual metrics** from any layer, pass names or glob patterns. They are removed with OpenTelemetry
Views, so they are never aggregated or exported:

```python
instrument(
    agent_server,
    exclude_metrics=[
        "system.*",  # node-level, not the app; container.* is the app's view
        "process.runtime.*",
        "cpython.gc.*",
        "http.server.response.size",
        "http.server.request.size",
    ],
)
```

`exclude_metrics` applies to the meter provider this library creates. If the process already has one (e.g. under
`opentelemetry-instrument`), add Views to that provider instead.

**Cost:** App telemetry ingestion is billed, and `system.*` is by far the noisiest group: `system.disk.*` and
`system.network.*` emit one series per host device, which was 56 disk series per export on a MEDIUM app. For app
health, `container.*` is the signal to keep. Dropping `system.*` is a good default unless you need node-level detail.
The OpenTelemetry system-metrics instrumentation also honours `OTEL_PYTHON_SYSTEM_METRICS_EXCLUDED_METRICS`
(comma-separated globs, set in the app's environment), if you'd rather configure it outside code.

**Requests that never count as load:** `/health`, `/healthz`, `/ping`, `/favicon.ico` and anything under `/assets/` or
`/static/` (e.g. the AgentServer chat UI) are skipped by the request metrics and get no request spans.

## Notes and limits

- **Databricks Apps specifics (verified on a MEDIUM app, 2026-10):** cgroup v1 is mounted read-only.
  The limits are 2 vCPU and **8 GiB**. `system.*` reports the node (4 CPUs / ~16 GB / 30+ disks).
- **App telemetry is Beta and has no published latency SLA.** We saw 10–30 s from event to queryable
  row, but delays of 15+ minutes have also been observed, so alert on telemetry freshness too.
- App telemetry drops `unit` and `start_time_unix_nano` from metrics. Partition rate calculations by
  `resource.attributes['service.instance.id']`.
- **Token counts for streamed chat completions** need `stream_options={"include_usage": True}`. The
  library never changes your requests.
- **Reasoning models** (e.g. Claude on Databricks) stream hidden reasoning first. Their reasoning
  tokens count toward output tokens and the time to first visible token.
- **MLflow AgentServer runs sync `@invoke` / `@stream` handlers directly on the event loop.** Each
  request then blocks its worker. `app.event_loop.lag.peak` shows it: in testing, 10 concurrent
  streaming requests on 2 workers had a lag of about 4 s and a client TTFB of up to 9.8 s, against
  2.4 s with an async handler.

## Development

```bash
uv sync                 # dev environment (all extras + test tools)
uv run pytest           # unit tests: fake cgroup fs, mocked OpenAI transport, in-memory OTel
uv run ruff check . && uv run ruff format --check .
uv build                # sdist + wheel in dist/
```

## CI/CD

| Workflow | Runs on | What it does |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | push to `main`, pull requests | ruff lint + format check; tests on Python 3.10–3.14, plus 3.10 at the **lowest** declared dependency versions; build, `twine check --strict` and a clean-venv install smoke test of the wheel |
| [`release.yml`](.github/workflows/release.yml) | tags `v*` | checks that the tag matches `pyproject.toml` and `__version__`, tests, builds once, publishes to **TestPyPI**, then to **PyPI** (after approval on the `pypi` environment), then creates a GitHub release |
| [`dependabot.yml`](.github/dependabot.yml) | weekly | keeps the workflow actions up to date |

Publishing uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so the repository holds no
API tokens. One-time setup:

1. On pypi.org and test.pypi.org, add a *pending* trusted publisher for project `apps-pressure-telemetry`. Set this
   repository, workflow `release.yml`, and environment `pypi` (or `testpypi`).
2. In the GitHub repository settings, create the environments `testpypi` and `pypi`, and add required reviewers
   to `pypi`.

To release:

```bash
# 1. bump version in pyproject.toml and databricks_apps_telemetry/__init__.py, update CHANGELOG.md, merge to main
git tag v0.1.0 && git push origin v0.1.0     # 2. CI publishes to TestPyPI, then PyPI after approval
```

License: Apache-2.0

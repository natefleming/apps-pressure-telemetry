# Example: an MLflow AgentServer agent on Databricks Apps

A minimal streaming agent on [MLflow AgentServer](https://mlflow.org/docs/latest/genai/serving/agent-server),
instrumented with `apps-pressure-telemetry`, plus a notebook with example queries for every metric it
exports. It deploys as one Declarative Automation Bundle.

> Community example, not an official Databricks product.

| Path | What it is |
|---|---|
| `app/agent_server/start_server.py` | The MLflow agent template layout. The only telemetry code is `instrument(agent_server)` |
| `app/agent_server/agent.py` | An async `@stream` / `@invoke` agent calling a Foundation Model through `AsyncDatabricksOpenAI` (streaming, with token usage), plus `mlflow.openai.autolog()` |
| `app/requirements.txt` | `apps-pressure-telemetry[all]`, `mlflow`, `databricks-openai` |
| `resources/app.yml` | The app (2 uvicorn workers) with **App telemetry** writing to `<catalog>.<schema>.otel_*`, and the schema |
| `notebooks/query_metrics.py` | Example queries: metric inventory, container CPU / memory / disk vs limits, process metrics, request concurrency / time to first byte / event-loop lag, routes, LLM metrics and span percentiles, per-request LLM breakdown, freshness, alert-ready queries |
| `resources/query_job.yml` | A job that runs the notebook against this app's tables |

## Prerequisites

- Databricks CLI ≥ 1.22, authenticated to the workspace (`databricks auth login`, or `-p <profile>`).
- A region where App telemetry is available, and a catalog on your own cloud storage (not default storage) where
  you can create a schema.
- A Foundation Model endpoint (`llm_endpoint`, default `databricks-claude-haiku-5-5`).

## Deploy and try it

```bash
cd examples/agent-server-app
databricks bundle deploy --var catalog=<catalog>
databricks bundle run agent_app                       # builds and starts the app

# send it a few requests (the app URL is printed by the previous command)
TOKEN=$(databricks auth token | jq -r .access_token)
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -X POST https://<app-url>/responses -d '{"input":[{"role":"user","content":"Hello!"}], "stream": true}'

databricks bundle run query_metrics --var catalog=<catalog>   # or open notebooks/query_metrics in the workspace
```

The first metrics land in `otel_metrics` about 10–30 s after the first request. In `mode: development` the
schema name gets a `dev_<user>_` prefix.

## Clean up

```bash
databricks bundle destroy --var catalog=<catalog>
```

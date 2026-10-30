# Databricks notebook source
# MAGIC %md
# MAGIC # Querying apps-pressure-telemetry signals
# MAGIC
# MAGIC Example queries for every signal `instrument()` exports, against the App telemetry tables
# MAGIC (`otel_metrics`, `otel_spans`, `otel_logs`). Copy them into dashboards and SQL alerts.
# MAGIC
# MAGIC **How the tables are shaped**
# MAGIC * `otel_metrics`: one row per metric series per export (every 10 s). The value is in a per-type struct:
# MAGIC   `gauge.value`, `sum.value` (cumulative counters) or `histogram.{count,sum,bucket_counts}` (cumulative).
# MAGIC   Attributes are VARIANT: `gauge.attributes:['gen_ai.token.type']`. Dotted keys need the `['...']` form.
# MAGIC * `otel_spans`: one row per span. Request spans are `kind = 'SPAN_KIND_SERVER'`; LLM calls are
# MAGIC   `kind = 'SPAN_KIND_CLIENT' AND name LIKE 'chat %'`.
# MAGIC * `service_name` is the app name. `resource.attributes:['service.instance.id']` is one process (one uvicorn worker)
# MAGIC   and `resource.attributes:['host.name']` is one app instance. Aggregate `container.*` with **max**, and sum
# MAGIC   per-worker peaks across workers.
# MAGIC * Cumulative values restart at 0 when a process restarts, so compute deltas per `service.instance.id`.
# MAGIC   (App telemetry leaves `start_time_unix_nano` NULL, so it can't be used for this.)

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "app_telemetry", "Schema")
dbutils.widgets.text("app_name", "agent-telemetry-example", "App name (= service_name)")
dbutils.widgets.text("lookback_minutes", "60", "Lookback (minutes)")

catalog: str = dbutils.widgets.get("catalog")
schema: str = dbutils.widgets.get("schema")
metrics_table: str = f"`{catalog}`.`{schema}`.otel_metrics"
spans_table: str = f"`{catalog}`.`{schema}`.otel_spans"
args: dict[str, object] = {
    "app": dbutils.widgets.get("app_name"),
    "lookback": int(dbutils.widgets.get("lookback_minutes")),
}
WINDOW: str = "time > current_timestamp() - make_interval(0, 0, 0, 0, 0, :lookback)"


def q(sql: str) -> None:
    display(spark.sql(sql, args=args))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Every metric the app exported
# MAGIC Name, type, description and the attribute keys you can group by.

# COMMAND ----------

q(f"""
  SELECT name, metric_type, any_value(description) AS description,
         count(DISTINCT coalesce(gauge.attributes, sum.attributes, histogram.attributes)::string) AS series,
         any_value(coalesce(gauge.attributes, sum.attributes, histogram.attributes))::string AS example_attributes,
         max(time) AS last_seen
  FROM {metrics_table}
  WHERE service_name = :app AND {WINDOW}
  GROUP BY name, metric_type ORDER BY name
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. A reusable flat view with per-second rates
# MAGIC Flattens value and attributes and computes, between consecutive exports of the same series and process,
# MAGIC `rate_per_s` for cumulative counters and `hist_count_delta` / `hist_sum_delta` for cumulative histograms.
# MAGIC Most queries below read from it.
# MAGIC
# MAGIC A series first appears at its first recording, already carrying everything since the process started (for
# MAGIC example, an LLM-duration histogram shows up with count = 14 after a burst). So deltas are computed over a
# MAGIC day of history, and a series' first row counts from 0. Only then is the result filtered to the lookback.

# COMMAND ----------

spark.sql(
    f"""
  CREATE OR REPLACE TEMP VIEW app_metrics AS
  WITH flat AS (
    SELECT time, time_unix_nano, name,
           resource.attributes:['service.instance.id']::string                     AS instance_id,
           resource.attributes:['host.name']::string                               AS host,
           coalesce(gauge.value, sum.value)                                         AS value,
           histogram.count                                                          AS hist_count,
           histogram.sum                                                            AS hist_sum,
           coalesce(gauge.attributes, sum.attributes, histogram.attributes)         AS attrs,
           coalesce(gauge.attributes, sum.attributes, histogram.attributes)::string AS series
    FROM {metrics_table}
    WHERE service_name = :app AND time > current_timestamp() - INTERVAL 1 DAY
  ),
  rated AS (
    SELECT *,
           (value - lag(value) OVER w) / ((time_unix_nano - lag(time_unix_nano) OVER w) / 1e9) AS rate_per_s,
           hist_count - coalesce(lag(hist_count) OVER w, 0)                                  AS hist_count_delta,
           hist_sum - coalesce(lag(hist_sum) OVER w, 0)                                      AS hist_sum_delta
    FROM flat
    WINDOW w AS (PARTITION BY name, series, instance_id ORDER BY time_unix_nano)
  )
  SELECT * FROM rated WHERE {WINDOW}
""",
    args=args,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Container resources vs the app's real limits (`container.*`)
# MAGIC Read from the container's cgroup, so they cover every process in the container and compare against the
# MAGIC limits the platform enforces. A MEDIUM app has 2 vCPU and 8 GiB. `container.memory.working_set`
# MAGIC (usage minus inactive page cache) is what the OOM killer tracks.

# COMMAND ----------

q("""
  SELECT date_trunc('MINUTE', time) AS minute,
         round(max(CASE WHEN name = 'container.cpu.utilization'        THEN value END), 3)          AS cpu_pct_of_limit,
         max(CASE WHEN name = 'container.cpu.limit'                    THEN value END)              AS cpu_limit_cores,
         round(max(CASE WHEN name = 'container.cpu.throttled.time'     THEN rate_per_s END), 3)     AS throttled_s_per_s,
         round(max(CASE WHEN name = 'container.memory.working_set'     THEN value END) / 1048576)   AS working_set_mib,
         round(max(CASE WHEN name = 'container.memory.limit'           THEN value END) / 1048576)   AS memory_limit_mib,
         round(max(CASE WHEN name = 'container.memory.utilization'     THEN value END), 3)          AS memory_pct_of_limit,
         max(CASE WHEN name = 'container.memory.oom_kills'             THEN value END)              AS oom_kills,
         round(max(CASE WHEN name = 'container.filesystem.utilization' THEN value END), 3)          AS disk_pct_used
  FROM app_metrics
  WHERE name LIKE 'container.%'
  GROUP BY 1 ORDER BY 1
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Process and host metrics (`process.*`, `system.*`)
# MAGIC `process.*` is per process (per worker). The rate of `process.cpu.time` is cores used; don't use
# MAGIC `process.cpu.utilization`, which shares a sampling window with the runtime gauge. `system.*` describes the
# MAGIC **node** under the container, not the app, so use `container.*` for app health.

# COMMAND ----------

q("""
  SELECT date_trunc('MINUTE', time) AS minute, instance_id,
         round(sum(CASE WHEN name = 'process.cpu.time' THEN rate_per_s END), 2)        AS process_cores_used,
         round(max(CASE WHEN name = 'process.memory.usage' THEN value END) / 1048576)  AS process_rss_mib,
         max(CASE WHEN name = 'process.thread.count' THEN value END)                   AS threads,
         max(CASE WHEN name = 'process.open_file_descriptor.count' THEN value END)     AS open_fds
  FROM app_metrics
  WHERE name LIKE 'process.%'
  GROUP BY 1, 2 ORDER BY 1, 2
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Request handling: concurrency, time to first byte, event-loop lag
# MAGIC * `app.http.requests.inflight.peak`: the highest number of concurrent requests between exports, per worker.
# MAGIC   Sum it across workers.
# MAGIC * `app.http.time_to_first_byte`: a histogram. The interval mean is Δsum / Δcount. For percentiles, use the
# MAGIC   SERVER spans in section 6.
# MAGIC * `app.event_loop.lag.peak`: the longest time the worker's asyncio loop was blocked. Above about 0.1 s, every
# MAGIC   request on that worker is waiting, and server-side latency does **not** show that queueing.

# COMMAND ----------

q("""
  WITH per_worker AS (
    SELECT date_trunc('MINUTE', time) AS minute, instance_id,
           max(CASE WHEN name = 'app.http.requests.inflight.peak' THEN value END) AS http_peak,
           max(CASE WHEN name = 'app.event_loop.lag.peak'         THEN value END) AS loop_lag_s,
           sum(CASE WHEN name = 'app.http.time_to_first_byte'     THEN hist_sum_delta END)   AS ttfb_sum,
           sum(CASE WHEN name = 'app.http.time_to_first_byte'     THEN hist_count_delta END) AS ttfb_count
    FROM app_metrics GROUP BY 1, 2
  )
  SELECT minute,
         sum(http_peak)                                       AS concurrent_requests_peak,
         round(max(loop_lag_s), 3)                            AS event_loop_lag_peak_s,
         round(sum(ttfb_sum) / nullif(sum(ttfb_count), 0), 3) AS ttfb_mean_s,
         sum(ttfb_count)                                      AS requests
  FROM per_worker GROUP BY 1 ORDER BY 1
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Requests per route, from SERVER spans
# MAGIC Exact per-request duration, time to first byte (`app.http.time_to_first_byte_ms`) and status.

# COMMAND ----------

q(f"""
  SELECT coalesce(attributes:['http.route']::string, name)                                          AS route,
         count(*)                                                                                     AS requests,
         count_if(coalesce(attributes:['http.response.status_code'], attributes:['http.status_code'])::int >= 500) AS server_errors,
         round(percentile((end_time_unix_nano - start_time_unix_nano) / 1e6, 0.5))                   AS p50_ms,
         round(percentile((end_time_unix_nano - start_time_unix_nano) / 1e6, 0.95))                  AS p95_ms,
         round(percentile(attributes:['app.http.time_to_first_byte_ms']::double, 0.95))              AS ttfb_p95_ms
  FROM {spans_table}
  WHERE service_name = :app AND kind = 'SPAN_KIND_SERVER' AND {WINDOW}
  GROUP BY 1 ORDER BY requests DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. LLM calls: metrics (`gen_ai.*`, `app.genai.*`)
# MAGIC * `gen_ai.client.operation.duration` and `gen_ai.client.token.usage` follow the OTel GenAI semantic conventions.
# MAGIC   Group tokens by `gen_ai.token.type` (input / output).
# MAGIC * `app.genai.time_to_first_token` is recorded for streaming calls only (time to the first *visible* text).
# MAGIC * `app.genai.requests.inflight.peak` is the peak concurrent LLM calls per worker. Sum it across workers.
# MAGIC * `app.genai.errors` is a counter by `error.type` (`rate_limited`, `timeout`, `http_5xx`, ...).

# COMMAND ----------

q("""
  WITH per_worker AS (
    SELECT date_trunc('MINUTE', time) AS minute, instance_id,
           max(CASE WHEN name = 'app.genai.requests.inflight.peak' THEN value END)                               AS llm_peak,
           sum(CASE WHEN name = 'gen_ai.client.operation.duration' THEN hist_count_delta END)                    AS calls,
           sum(CASE WHEN name = 'gen_ai.client.operation.duration' THEN hist_sum_delta END)                      AS duration_sum,
           sum(CASE WHEN name = 'app.genai.time_to_first_token'    THEN hist_sum_delta END)                      AS ttft_sum,
           sum(CASE WHEN name = 'app.genai.time_to_first_token'    THEN hist_count_delta END)                    AS ttft_count,
           sum(CASE WHEN name = 'gen_ai.client.token.usage' AND attrs:['gen_ai.token.type']::string = 'input'
                    THEN hist_sum_delta END)                                                                     AS input_tokens,
           sum(CASE WHEN name = 'gen_ai.client.token.usage' AND attrs:['gen_ai.token.type']::string = 'output'
                    THEN hist_sum_delta END)                                                                     AS output_tokens
    FROM app_metrics GROUP BY 1, 2
  )
  SELECT minute, sum(llm_peak) AS concurrent_llm_calls_peak, sum(calls) AS llm_calls,
         round(sum(duration_sum) / nullif(sum(calls), 0), 2)    AS llm_mean_s,
         round(sum(ttft_sum) / nullif(sum(ttft_count), 0), 2)   AS ttft_mean_s,
         sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens
  FROM per_worker GROUP BY 1 ORDER BY 1
""")

# COMMAND ----------

q("""
  SELECT attrs:['error.type']::string AS error_type, attrs:['gen_ai.request.model']::string AS model,
         max(value) AS errors_since_process_start, max(time) AS last_seen
  FROM app_metrics WHERE name = 'app.genai.errors'
  GROUP BY 1, 2 ORDER BY errors_since_process_start DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. LLM calls: percentiles from CLIENT spans
# MAGIC One span per LLM call, carrying `app.genai.duration_ms`, `app.genai.ttft_ms`, `gen_ai.usage.*_tokens` and
# MAGIC `error.type`. Use it for exact percentiles.

# COMMAND ----------

q(f"""
  SELECT attributes:['gen_ai.request.model']::string                                AS model,
         count(*)                                                                    AS calls,
         count_if(status.code = 'STATUS_CODE_ERROR')                                 AS errors,
         round(percentile(attributes:['app.genai.duration_ms']::double, 0.5) / 1000, 2)  AS p50_s,
         round(percentile(attributes:['app.genai.duration_ms']::double, 0.95) / 1000, 2) AS p95_s,
         round(percentile(attributes:['app.genai.ttft_ms']::double, 0.95) / 1000, 2)     AS ttft_p95_s,
         sum(attributes:['gen_ai.usage.input_tokens']::bigint)                       AS input_tokens,
         sum(attributes:['gen_ai.usage.output_tokens']::bigint)                      AS output_tokens
  FROM {spans_table}
  WHERE service_name = :app AND kind = 'SPAN_KIND_CLIENT' AND name LIKE 'chat %' AND {WINDOW}
  GROUP BY 1
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Per request: how many LLM calls and how long they took
# MAGIC LLM spans are children of the request span, so they join on `trace_id`.

# COMMAND ----------

q(f"""
  WITH req AS (
    SELECT trace_id, time, coalesce(attributes:['http.route']::string, name) AS route,
           (end_time_unix_nano - start_time_unix_nano) / 1e6 AS request_ms,
           attributes:['app.http.time_to_first_byte_ms']::double AS ttfb_ms
    FROM {spans_table} WHERE service_name = :app AND kind = 'SPAN_KIND_SERVER' AND {WINDOW}
  ),
  llm AS (
    SELECT trace_id, count(*) AS llm_calls, sum(attributes:['app.genai.duration_ms']::double) AS llm_ms,
           sum(attributes:['gen_ai.usage.output_tokens']::bigint) AS output_tokens
    FROM {spans_table} WHERE service_name = :app AND kind = 'SPAN_KIND_CLIENT' AND name LIKE 'chat %' AND {WINDOW}
    GROUP BY trace_id
  )
  SELECT req.time, route, round(request_ms) AS request_ms, round(ttfb_ms) AS ttfb_ms,
         llm_calls, round(llm_ms) AS llm_ms, output_tokens
  FROM req JOIN llm USING (trace_id) ORDER BY req.time DESC LIMIT 50
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Telemetry freshness
# MAGIC App telemetry has no latency SLA. Alert when an app goes silent, whether it stopped or ingestion is delayed.

# COMMAND ----------

q(f"""
  SELECT max(time) AS last_export, timestampdiff(SECOND, max(time), current_timestamp()) AS seconds_since_last_export,
         count(DISTINCT resource.attributes:['service.instance.id']::string) AS reporting_processes
  FROM {metrics_table}
  WHERE service_name = :app AND time > current_timestamp() - INTERVAL 1 DAY
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Alert-ready queries
# MAGIC Each returns one value. Paste it into a Databricks SQL alert (evaluate every minute) and use fully qualified
# MAGIC table names. The thresholds are examples for a MEDIUM app (2 vCPU, 8 GiB).

# COMMAND ----------

alerts: dict[str, str] = {
    "memory_pct_of_limit > 0.85": f"""
        SELECT coalesce(max(gauge.value), 0) AS memory_pct_of_limit FROM {metrics_table}
        WHERE service_name = :app AND name = 'container.memory.utilization'
          AND time > current_timestamp() - INTERVAL 5 MINUTES""",
    "cpu_pct_of_limit > 0.80 (5 min avg)": f"""
        SELECT coalesce(avg(gauge.value), 0) AS cpu_pct_of_limit FROM {metrics_table}
        WHERE service_name = :app AND name = 'container.cpu.utilization'
          AND time > current_timestamp() - INTERVAL 5 MINUTES""",
    "event_loop_lag_s > 1": f"""
        SELECT coalesce(max(gauge.value), 0) AS event_loop_lag_s FROM {metrics_table}
        WHERE service_name = :app AND name = 'app.event_loop.lag.peak'
          AND time > current_timestamp() - INTERVAL 5 MINUTES""",
    "llm_p95_s > 10": f"""
        SELECT coalesce(percentile(attributes:['app.genai.duration_ms']::double, 0.95) / 1000, 0) AS llm_p95_s
        FROM {spans_table}
        WHERE service_name = :app AND kind = 'SPAN_KIND_CLIENT' AND name LIKE 'chat %'
          AND status.code != 'STATUS_CODE_ERROR' AND time > current_timestamp() - INTERVAL 10 MINUTES""",
    "llm_errors > 5 (5 min)": f"""
        SELECT count(*) AS llm_errors FROM {spans_table}
        WHERE service_name = :app AND kind = 'SPAN_KIND_CLIENT' AND name LIKE 'chat %'
          AND status.code = 'STATUS_CODE_ERROR' AND time > current_timestamp() - INTERVAL 5 MINUTES""",
    "seconds_since_last_export > 300": f"""
        SELECT coalesce(timestampdiff(SECOND, max(time), current_timestamp()), 86400) AS seconds_since_last_export
        FROM {metrics_table} WHERE service_name = :app""",
}
for condition, sql in alerts.items():
    value = spark.sql(sql, args={"app": args["app"]}).first()[0]
    print(f"-- alert when {condition}: current value = {value}\n{sql.strip()}\n")

import json
from collections.abc import Iterator

import httpx
import openai
import pytest

from databricks_apps_telemetry._openai import _TARGETS, _WRAPPED_FLAG, OpenAIInstrumentation

MODEL: str = "databricks-claude-haiku-5-5"


def _sse(events: list[dict]) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


def _chunk(content: object = None, usage: dict | None = None) -> dict:
    choices = [] if content is None else [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": choices,
        "usage": usage,
    }


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("model") == "limited":
        return httpx.Response(429, json={"error": {"message": "slow down"}})
    if request.url.path.endswith("/responses"):
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "response",
                "created_at": 0,
                "model": MODEL,
                "status": "completed",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )
    if body.get("stream"):
        # Claude on Databricks: hidden reasoning parts first (list content), then text, then usage
        events = [
            _chunk([{"type": "reasoning", "summary": [{"type": "summary_text", "text": ""}]}]),
            _chunk("Girl & the Goat"),
            _chunk(" (West Loop)"),
            _chunk("", usage={"prompt_tokens": 33, "completion_tokens": 81, "total_tokens": 114}),
        ]
        return httpx.Response(200, content=_sse(events), headers={"content-type": "text/event-stream"})
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        },
    )


@pytest.fixture
def inst(otel) -> Iterator[OpenAIInstrumentation]:
    # instrument() in another test module may already have patched the SDK: test on a clean SDK
    import importlib

    saved: dict[tuple[type, str], object] = {}
    for module_name, class_name, method, _, _ in _TARGETS:
        cls = getattr(importlib.import_module(module_name), class_name)
        fn = getattr(cls, method)
        saved[(cls, method)] = fn
        while getattr(fn, _WRAPPED_FLAG, False):
            fn = fn.__wrapped__
        setattr(cls, method, fn)
    instrumentation = OpenAIInstrumentation(otel.meter_provider.get_meter("t"), otel.tracer_provider.get_tracer("t"))
    patched = instrumentation.instrument()
    assert len(patched) == 4
    assert instrumentation.instrument() == []  # idempotent
    yield instrumentation
    instrumentation.uninstrument()
    for (cls, method), fn in saved.items():
        setattr(cls, method, fn)


@pytest.fixture
def client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key="x",
        base_url="https://example.test/serving-endpoints",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


def _tokens(metrics: dict) -> dict[str, int]:
    return {attrs["gen_ai.token.type"]: total for (count, total), attrs in metrics["gen_ai.client.token.usage"]}


def test_non_streaming_chat(otel, inst, client) -> None:
    resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "hi"
    m = otel.collect()
    assert m["gen_ai.client.operation.duration"][0][0][0] == 1
    assert _tokens(m) == {"input": 12, "output": 5}
    assert "app.genai.time_to_first_token" not in m
    span = otel.spans.get_finished_spans()[0]
    assert span.name == f"chat {MODEL}" and span.attributes["gen_ai.usage.output_tokens"] == 5


def test_streaming_chat_ttft_skips_reasoning_and_keeps_stream_api(otel, inst, client) -> None:
    with client.chat.completions.create(
        model=MODEL, messages=[], stream=True, stream_options={"include_usage": True}
    ) as stream:
        text = "".join(
            c.choices[0].delta.content for c in stream if c.choices and isinstance(c.choices[0].delta.content, str)
        )
        assert stream.response.status_code == 200  # attribute access passes through the proxy
    assert text == "Girl & the Goat (West Loop)"
    m = otel.collect()
    assert m["app.genai.time_to_first_token"][0][0][0] == 1
    assert _tokens(m) == {"input": 33, "output": 81}
    assert m["app.genai.requests.inflight"][0][0] == 0
    assert m["app.genai.requests.inflight.peak"][0][0] == 1
    assert len(otel.spans.get_finished_spans()) == 1


def test_rate_limit_is_counted_as_error(otel, inst, client) -> None:
    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create(model="limited", messages=[])
    m = otel.collect()
    assert m["app.genai.errors"] == [
        (
            1,
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "limited",
                "app.genai.api": "chat.completions",
                "error.type": "rate_limited",
            },
        )
    ]
    span = otel.spans.get_finished_spans()[0]
    assert span.attributes["error.type"] == "rate_limited" and not span.status.is_ok


def test_responses_api(otel, inst, client) -> None:
    client.responses.create(model=MODEL, input="hi")
    assert _tokens(otel.collect()) == {"input": 7, "output": 3}


async def test_async_client(otel, inst) -> None:
    aclient = openai.AsyncOpenAI(
        api_key="x",
        base_url="https://example.test/serving-endpoints",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )
    await aclient.chat.completions.create(model=MODEL, messages=[])
    stream = await aclient.chat.completions.create(model=MODEL, messages=[], stream=True)
    chunks = [c async for c in stream]
    assert len(chunks) == 4
    m = otel.collect()
    assert m["gen_ai.client.operation.duration"][0][0][0] == 2
    assert m["app.genai.time_to_first_token"][0][0][0] == 1


def test_subclass_clients_are_covered(otel, inst) -> None:
    class DatabricksLikeClient(openai.OpenAI):  # DatabricksOpenAI subclasses openai.OpenAI the same way
        pass

    c = DatabricksLikeClient(
        api_key="x",
        base_url="https://example.test/serving-endpoints",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    c.chat.completions.create(model=MODEL, messages=[])
    assert otel.collect()["gen_ai.client.operation.duration"][0][0][0] == 1


def test_raw_response_records_tokens_and_still_parses(otel, inst, client) -> None:
    # langchain-openai's non-streaming path: client.chat.completions.with_raw_response.create(...)
    raw = client.chat.completions.with_raw_response.create(model=MODEL, messages=[])
    assert raw.parse().choices[0].message.content == "hi"  # caller's parse still works
    assert _tokens(otel.collect()) == {"input": 12, "output": 5}


def test_raw_streaming_response_is_passed_through(otel, inst, client) -> None:
    raw = client.chat.completions.with_raw_response.create(model=MODEL, messages=[], stream=True)
    chunks = list(raw.parse())  # untouched: the caller parses and iterates it
    assert len(chunks) == 4
    assert otel.collect()["gen_ai.client.operation.duration"][0][0][0] == 1

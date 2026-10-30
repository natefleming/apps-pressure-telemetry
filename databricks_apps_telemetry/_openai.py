"""LLM-call pressure signals for the OpenAI Python SDK (and subclasses such as DatabricksOpenAI).

Patches `chat.completions.create` and `responses.create` (sync + async, streaming + not) at
the class level, so every client instance is covered, including ones created later.

  app.genai.requests.inflight        up/down counter  LLM calls in progress (value at export)
  app.genai.requests.inflight.peak   gauge            peak concurrent LLM calls since the previous export
  gen_ai.client.operation.duration   histogram (s)    call duration (OTel GenAI semantic convention)
  gen_ai.client.token.usage          histogram        input / output tokens (OTel GenAI semantic convention)
  app.genai.time_to_first_token      histogram (s)    streaming only: time to the first visible text
  app.genai.errors                   counter          failed calls by error.type

Each call is also a CLIENT span "chat <model>" (parented to the current request span) with
gen_ai.* attributes, app.genai.duration_ms, app.genai.ttft_ms and token counts.

Token counts come from the response's `usage`. For streaming chat completions the endpoint
only sends usage when the caller passes `stream_options={"include_usage": True}`; this module
never changes the request, so without that option streamed calls record no token counts.
Reasoning models (e.g. Claude) stream hidden reasoning parts first; time to first token is
measured to the first *visible* text, which is what the user waits for.
"""

import functools
import importlib
import time
from collections.abc import Callable
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from databricks_apps_telemetry._peak import PeakTracker

# (module, class, method, api label, is_async)
_TARGETS: tuple[tuple[str, str, str, str, bool], ...] = (
    ("openai.resources.chat.completions", "Completions", "create", "chat.completions", False),
    ("openai.resources.chat.completions", "AsyncCompletions", "create", "chat.completions", True),
    ("openai.resources.responses", "Responses", "create", "responses", False),
    ("openai.resources.responses", "AsyncResponses", "create", "responses", True),
)
_WRAPPED_FLAG: str = "__databricks_apps_telemetry_wrapped__"


def error_type(exc: BaseException) -> str:
    """Stable, low-cardinality error label for metrics and alerts."""
    import openai

    if isinstance(exc, openai.RateLimitError):
        return "rate_limited"
    if isinstance(exc, openai.APITimeoutError):
        return "timeout"
    if isinstance(exc, openai.APIStatusError):
        return f"http_{exc.status_code}"
    if isinstance(exc, openai.APIConnectionError):
        return "connection"
    return type(exc).__name__


def _visible_text(content: object) -> str:
    """Streamed delta content: a string for most models, a list of content parts for Claude."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text: object = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            kind: object = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if isinstance(text, str) and kind != "reasoning":
                parts.append(text)
        return "".join(parts)
    return ""


def _is_raw_response(obj: Any) -> bool:
    """`client.chat.completions.with_raw_response.create(...)` (used by langchain-openai) returns an HTTP
    response wrapper instead of the parsed object."""
    return callable(getattr(obj, "parse", None)) and hasattr(obj, "http_response")


def _parsed(obj: Any) -> Any:
    """Parsed body of a raw response; `parse()` caches, so the caller's own parse() returns the same object."""
    if not _is_raw_response(obj):
        return obj
    try:
        return obj.parse()
    except Exception:  # noqa: BLE001 - telemetry must not change the error; the caller's own parse() raises it
        return None


def _usage(obj: Any) -> tuple[int | None, int | None]:
    usage: Any = getattr(obj, "usage", None)
    if usage is None:
        return None, None
    input_tokens: Any = getattr(usage, "prompt_tokens", None)
    output_tokens: Any = getattr(usage, "completion_tokens", None)
    if input_tokens is None:  # Responses API naming
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
    return input_tokens, output_tokens


class OpenAIInstrumentation:
    """Owns the GenAI instruments and patches/unpatches the OpenAI SDK."""

    def __init__(self, meter: metrics.Meter, tracer: trace.Tracer) -> None:
        self.tracer: trace.Tracer = tracer
        self.inflight: metrics.UpDownCounter = meter.create_up_down_counter(
            "app.genai.requests.inflight", unit="{request}", description="LLM calls currently in progress"
        )
        self.tracker: PeakTracker = PeakTracker()
        self.tracker.register(
            meter, "app.genai.requests.inflight.peak", "Peak concurrent LLM calls since the previous export"
        )
        self.duration: metrics.Histogram = meter.create_histogram(
            "gen_ai.client.operation.duration", unit="s", description="LLM call duration"
        )
        self.tokens: metrics.Histogram = meter.create_histogram(
            "gen_ai.client.token.usage", unit="{token}", description="Tokens per LLM call"
        )
        self.ttft: metrics.Histogram = meter.create_histogram(
            "app.genai.time_to_first_token", unit="s", description="Time to the first visible streamed text"
        )
        self.errors: metrics.Counter = meter.create_counter(
            "app.genai.errors", unit="{error}", description="Failed LLM calls by error.type"
        )
        self._originals: dict[tuple[str, str, str], Callable[..., Any]] = {}

    def instrument(self) -> list[str]:
        """Patch every available target; returns the qualified names that were patched."""
        patched: list[str] = []
        for module_name, class_name, method, api, is_async in _TARGETS:
            try:
                cls: type = getattr(importlib.import_module(module_name), class_name)
            except (ImportError, AttributeError):
                continue
            original: Callable[..., Any] = getattr(cls, method)
            if getattr(original, _WRAPPED_FLAG, False):
                continue
            wrapper: Callable[..., Any] = (self._wrap_async if is_async else self._wrap_sync)(original, api)
            setattr(wrapper, _WRAPPED_FLAG, True)
            setattr(cls, method, wrapper)
            self._originals[(module_name, class_name, method)] = original
            patched.append(f"{class_name}.{method}")
        return patched

    def uninstrument(self) -> None:
        for (module_name, class_name, method), original in self._originals.items():
            setattr(getattr(importlib.import_module(module_name), class_name), method, original)
        self._originals.clear()

    def _wrap_sync(self, original: Callable[..., Any], api: str) -> Callable[..., Any]:
        @functools.wraps(original)
        def create(resource: Any, *args: Any, **kwargs: Any) -> Any:
            call: _Call = _Call(self, api, kwargs.get("model"))
            try:
                result: Any = original(resource, *args, **kwargs)
            except BaseException as exc:
                call.fail(exc)
                raise
            if kwargs.get("stream"):
                if _is_raw_response(result):  # raw streaming response: the caller parses it; time to headers
                    call.finish()
                    return result
                return _StreamProxy(result, call)
            call.finish(*_usage(_parsed(result)))
            return result

        return create

    def _wrap_async(self, original: Callable[..., Any], api: str) -> Callable[..., Any]:
        @functools.wraps(original)
        async def create(resource: Any, *args: Any, **kwargs: Any) -> Any:
            call: _Call = _Call(self, api, kwargs.get("model"))
            try:
                result: Any = await original(resource, *args, **kwargs)
            except BaseException as exc:
                call.fail(exc)
                raise
            if kwargs.get("stream"):
                if _is_raw_response(result):
                    call.finish()
                    return result
                return _AsyncStreamProxy(result, call)
            call.finish(*_usage(_parsed(result)))
            return result

        return create


class _Call:
    """One LLM call: records metrics and its span exactly once, on success or failure."""

    def __init__(self, inst: OpenAIInstrumentation, api: str, model: object) -> None:
        self.inst: OpenAIInstrumentation = inst
        self.attrs: dict[str, str] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": str(model) if model else "unknown",
            "app.genai.api": api,
        }
        self.started: float = time.perf_counter()
        self.ttft: float | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.done: bool = False
        self.span: Span = inst.tracer.start_span(
            f"chat {self.attrs['gen_ai.request.model']}", kind=SpanKind.CLIENT, attributes=self.attrs
        )
        inst.inflight.add(1, self.attrs)
        inst.tracker.enter(self.attrs)

    def observe_chunk(self, item: Any) -> None:
        if self.ttft is None:
            text: str = ""
            if getattr(item, "choices", None):
                text = _visible_text(getattr(item.choices[0].delta, "content", None))
            elif getattr(item, "type", None) == "response.output_text.delta":
                text = getattr(item, "delta", "") or ""
            if text:
                self.ttft = time.perf_counter() - self.started
        input_tokens, output_tokens = _usage(getattr(item, "response", None) or item)
        if output_tokens is not None:
            self.input_tokens, self.output_tokens = input_tokens, output_tokens

    def _close(self, error: str | None) -> float:
        self.done = True
        duration: float = time.perf_counter() - self.started
        self.inst.inflight.add(-1, self.attrs)
        self.inst.tracker.exit(self.attrs)
        attrs: dict[str, str] = {**self.attrs, "error.type": error} if error else self.attrs
        self.inst.duration.record(duration, attrs)
        self.span.set_attribute("app.genai.duration_ms", round(duration * 1000, 1))
        return duration

    def finish(self, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        if self.done:
            return
        self._close(None)
        input_tokens = input_tokens if input_tokens is not None else self.input_tokens
        output_tokens = output_tokens if output_tokens is not None else self.output_tokens
        if self.ttft is not None:
            self.inst.ttft.record(self.ttft, self.attrs)
            self.span.set_attribute("app.genai.ttft_ms", round(self.ttft * 1000, 1))
        for kind, count in (("input", input_tokens), ("output", output_tokens)):
            if count is not None:
                self.inst.tokens.record(count, {**self.attrs, "gen_ai.token.type": kind})
                self.span.set_attribute(f"gen_ai.usage.{kind}_tokens", count)
        self.span.end()

    def fail(self, exc: BaseException) -> None:
        if self.done:
            return
        label: str = error_type(exc)
        self._close(label)
        self.inst.errors.add(1, {**self.attrs, "error.type": label})
        self.span.set_attribute("error.type", label)
        self.span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
        self.span.end()


class _StreamProxy:
    """Wraps openai.Stream: same API, plus TTFT / usage / completion tracking."""

    def __init__(self, stream: Any, call: _Call) -> None:
        self._stream: Any = stream
        self._iterator: Any = iter(stream)
        self._call: _Call = call

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self) -> "_StreamProxy":
        return self

    def __next__(self) -> Any:
        try:
            item: Any = next(self._iterator)
        except StopIteration:
            self._call.finish()
            raise
        except BaseException as exc:
            self._call.fail(exc)
            raise
        self._call.observe_chunk(item)
        return item

    def __enter__(self) -> "_StreamProxy":  # noqa: PYI034 - typing.Self needs Python 3.11
        self._stream.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> Any:
        self._call.finish()
        return self._stream.__exit__(*exc_info)

    def close(self) -> None:
        self._call.finish()
        self._stream.close()


class _AsyncStreamProxy:
    """Wraps openai.AsyncStream: same API, plus TTFT / usage / completion tracking."""

    def __init__(self, stream: Any, call: _Call) -> None:
        self._stream: Any = stream
        self._iterator: Any = stream.__aiter__()
        self._call: _Call = call

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __aiter__(self) -> "_AsyncStreamProxy":
        return self

    async def __anext__(self) -> Any:
        try:
            item: Any = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._call.finish()
            raise
        except BaseException as exc:
            self._call.fail(exc)
            raise
        self._call.observe_chunk(item)
        return item

    async def __aenter__(self) -> "_AsyncStreamProxy":  # noqa: PYI034 - typing.Self needs Python 3.11
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> Any:
        self._call.finish()
        return await self._stream.__aexit__(*exc_info)

    async def close(self) -> None:
        self._call.finish()
        await self._stream.close()

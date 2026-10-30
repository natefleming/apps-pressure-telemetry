"""A minimal streaming agent for MLflow AgentServer.

There is no telemetry code here: `instrument(agent_server)` in start_server.py covers the server and
every LLM call made through the OpenAI SDK (AsyncDatabricksOpenAI included).

The handlers are `async`. AgentServer runs a plain (sync) generator directly on the event loop, which
blocks every other request on that worker for the whole LLM call - `app.event_loop.lag.peak` shows it.
"""

import os
import uuid
from collections.abc import AsyncGenerator

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI
from mlflow.genai.agent_server import invoke, stream
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent

LLM_ENDPOINT: str = os.environ.get("LLM_ENDPOINT", "databricks-claude-haiku-5-5")
SYSTEM_PROMPT: str = "You are a concise, helpful assistant. Answer in at most three sentences."

mlflow.openai.autolog()
client: AsyncDatabricksOpenAI = AsyncDatabricksOpenAI(workspace_client=WorkspaceClient())


def _text(content: object) -> str:
    """Message content is a string for most models and a list of parts for Claude (reasoning parts are skipped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") != "reasoning")
    return ""


def _messages(request: ResponsesAgentRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in request.input:
        data: dict = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        if data.get("role") in ("user", "assistant"):
            messages.append({"role": data["role"], "content": _text(data.get("content"))})
    return messages


@stream()
async def handle_stream(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    item_id: str = str(uuid.uuid4())
    parts: list[str] = []
    response = await client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=_messages(request),
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},  # token counts for streamed calls
    )
    async for chunk in response:
        delta: str = _text(chunk.choices[0].delta.content) if chunk.choices else ""
        if delta:
            parts.append(delta)
            yield ResponsesAgentStreamEvent(**ResponsesAgent.create_text_delta(delta=delta, item_id=item_id))
    yield ResponsesAgentStreamEvent(
        type="response.output_item.done",
        item=ResponsesAgent.create_text_output_item(text="".join(parts), id=item_id),
    )


@invoke()
async def handle_invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs: list[dict] = [
        event.item async for event in handle_stream(request) if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)

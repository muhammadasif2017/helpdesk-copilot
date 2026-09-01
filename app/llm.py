"""LLM client — any OpenAI-compatible backend (Groq, Ollama, vLLM, ...).

The backend is selected purely by environment variables, so the app can be
demoed against a fast hosted endpoint and still run fully offline against
Ollama without a code change.
"""

import os
from collections.abc import Iterator

from openai import OpenAI

_client: OpenAI | None = None

# The temperature the product ships. Evals deliberately run at this value rather
# than 0, so they measure the configuration users actually get.
TEMPERATURE = 0.2


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("LLM_API_KEY", "ollama"),
        )
    return _client


def model_name() -> str:
    return os.environ.get("LLM_MODEL", "qwen2.5:3b")


def stream_deltas(messages: list[dict], tools: list[dict] | None = None):
    """Yield raw streamed deltas, which may carry text, tool calls, or both."""
    optional = {"tools": tools} if tools else {}
    stream = get_client().chat.completions.create(
        model=model_name(),
        messages=messages,
        temperature=TEMPERATURE,
        stream=True,
        **optional,
    )
    for chunk in stream:
        if chunk.choices:
            yield chunk.choices[0].delta


def stream_chat(messages: list[dict]) -> Iterator[str]:
    """Yield response text deltas only. Used where tools are not in play."""
    for delta in stream_deltas(messages):
        if delta.content:
            yield delta.content

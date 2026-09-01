"""LLM client — any OpenAI-compatible backend (Groq, Ollama, vLLM, ...).

The backend is selected purely by environment variables, so the app can be
demoed against a fast hosted endpoint and still run fully offline against
Ollama without a code change.
"""

import os
from collections.abc import Iterator

from openai import OpenAI

_client: OpenAI | None = None


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


def stream_chat(messages: list[dict]) -> Iterator[str]:
    """Yield response text deltas for a chat completion."""
    stream = get_client().chat.completions.create(
        model=model_name(),
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta

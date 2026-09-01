"""Optional tracing to a self-hosted Langfuse.

Two rules govern everything here:

1. **Opt-in.** With no Langfuse keys configured, this is a no-op and the product
   behaves exactly as if the module did not exist. That matters on a laptop where
   the tracing stack and the model compete for the same RAM.

2. **It never breaks the product.** Every call into the SDK is guarded. A tracing
   backend that is down, slow, or misconfigured must never turn a working support
   assistant into a broken one — observability is there to explain failures, not
   to cause them.

Enable by starting the stack and setting the keys:

    docker compose -f docker-compose.langfuse.yml up -d
    # create a project at http://localhost:3100, then put its keys in .env
"""

import os
import time
from typing import Any

_client: Any = None
_client_ready = False


def enabled() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def _get_client() -> Any:
    """Build the client once. A failure here disables tracing rather than raising."""
    global _client, _client_ready
    if _client_ready:
        return _client

    _client_ready = True
    if not enabled():
        return None

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3100"),
        )
    except Exception:
        _client = None
    return _client


def reset() -> None:
    """Forget the cached client. Used by tests that change the environment."""
    global _client, _client_ready
    _client, _client_ready = None, False


class Turn:
    """One assistant turn. All methods are safe to call when tracing is off."""

    def __init__(self, trace: Any = None):
        self._trace = trace
        self._started = time.monotonic()
        self.actions: list[dict] = []
        self.sources: list[str] = []

    @property
    def active(self) -> bool:
        return self._trace is not None

    def record_sources(self, chunks) -> None:
        self.sources = [c.source for c in chunks]

    def record_event(self, event: dict) -> None:
        """Tool activity, so a trace shows what the assistant did, not just said."""
        if "action" in event or "proposal" in event:
            self.actions.append(event)

    def finish(self, answer: str, error: str | None = None) -> None:
        if self._trace is None:
            return
        try:
            self._trace.update(
                output={"answer": answer, "error": error},
                metadata={
                    "latency_seconds": round(time.monotonic() - self._started, 2),
                    "sources": self.sources,
                    "actions": self.actions,
                    "answer_chars": len(answer),
                },
            )
        except Exception:
            pass


def start_turn(question: str, ticket_id: Any = None) -> Turn:
    client = _get_client()
    if client is None:
        return Turn()

    try:
        trace = client.trace(
            name="support-turn",
            input={"question": question, "ticket_id": ticket_id},
            metadata={"model": os.environ.get("LLM_MODEL", "")},
        )
        return Turn(trace)
    except Exception:
        return Turn()


def flush() -> None:
    """Push buffered events. Safe to call when tracing is off."""
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        pass

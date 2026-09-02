"""Tracing must be invisible when off and harmless when broken.

These run without Langfuse installed or running — that is the point. The failure
mode being guarded against is an observability layer that takes the product down
with it, which is a worse outcome than having no observability at all.
"""

import pytest

from app import tracing


@pytest.fixture(autouse=True)
def clean_client(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracing.reset()
    yield
    tracing.reset()


class Boom:
    """A client whose every method fails, standing in for a broken backend."""

    def trace(self, **kwargs):
        raise RuntimeError("langfuse is down")

    def flush(self):
        raise RuntimeError("langfuse is down")


class Recorder:
    def __init__(self):
        self.traces = []
        self.updates = []
        self.flushed = 0

    def trace(self, **kwargs):
        self.traces.append(kwargs)
        return self

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def flush(self):
        self.flushed += 1


def test_tracing_is_off_without_keys():
    assert tracing.enabled() is False


def test_a_turn_without_tracing_is_inert():
    turn = tracing.start_turn("what is the return window?", ticket_id=101)
    assert turn.active is False
    turn.record_event({"delta": "hi"})
    turn.finish("an answer")  # must not raise
    tracing.flush()


def test_the_client_is_built_with_a_bounded_timeout(monkeypatch):
    """A backend that hangs must cost a turn seconds, not most of a minute.

    The SDK's own defaults (20s, three retries) are the failure this guards: they
    are silent, so the product stays correct and becomes unusably slow, which is
    harder to diagnose than an outright error.
    """
    langfuse = pytest.importorskip("langfuse")

    built = {}

    class FakeLangfuse:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    tracing.reset()

    assert tracing._get_client() is not None
    assert built["timeout"] <= 5, f"timeout of {built['timeout']}s stalls every turn"
    # Not `<= 1`: the SDK reads a falsy max_retries as unset and restores its
    # default of 3, so 0 would pass a check for "small" while behaving as 3.
    assert built["max_retries"] == 1


def test_a_broken_backend_does_not_break_the_turn(monkeypatch):
    monkeypatch.setattr(tracing, "_get_client", lambda: Boom())

    turn = tracing.start_turn("question", ticket_id=101)
    assert turn.active is False, "a failed trace call must degrade to a no-op turn"
    turn.finish("an answer")
    tracing.flush()


def test_a_working_backend_records_the_turn(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(tracing, "_get_client", lambda: recorder)

    turn = tracing.start_turn("what is the return window?", ticket_id=101)
    assert turn.active is True

    turn.record_event({"action": {"tool": "lookup_order", "arguments": {"order_id": 4471}}})
    turn.record_event({"delta": "ignored"})
    turn.finish("Footwear has a 45-day window.")

    assert recorder.traces[0]["input"] == {
        "question": "what is the return window?",
        "ticket_id": 101,
    }
    update = recorder.updates[0]
    assert update["output"]["answer"] == "Footwear has a 45-day window."
    assert update["metadata"]["actions"][0]["action"]["tool"] == "lookup_order"
    assert "latency_seconds" in update["metadata"]


def test_a_failed_turn_is_still_traced(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(tracing, "_get_client", lambda: recorder)

    turn = tracing.start_turn("question")
    turn.finish("partial text", error="RuntimeError('model died')")

    assert recorder.updates[0]["output"]["error"] == "RuntimeError('model died')"

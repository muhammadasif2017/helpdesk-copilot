"""Tests for the product path the browser actually uses.

Real retrieval, real tool execution, real SSE framing; only the model is stubbed,
so these run in about a second and can gate every commit.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import llm, main

STUB_DELTAS = ["Footwear has a 45-day return window ", "[returns.md]"]


def text_delta(content):
    return SimpleNamespace(content=content, tool_calls=None)


def tool_call_delta(name, arguments, call_id="call_1", index=0):
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                index=index,
                id=call_id,
                function=SimpleNamespace(name=name, arguments=arguments),
            )
        ],
    )


def stub_model(monkeypatch, *rounds):
    """Stub the model with one delta list per round of the tool loop."""
    remaining = list(rounds)

    def fake_stream_deltas(messages, tools=None):
        return iter(remaining.pop(0) if remaining else [])

    monkeypatch.setattr(llm, "stream_deltas", fake_stream_deltas)


@pytest.fixture
def client():
    return TestClient(main.app)


def sse_events(body: str) -> list[str]:
    return [block.removeprefix("data: ").strip() for block in body.strip().split("\n\n")]


def test_index_renders_the_ticket_queue(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Wrong size shoes" in response.text


def test_chat_streams_sources_then_deltas_then_done(client, monkeypatch):
    stub_model(monkeypatch, [text_delta(d) for d in STUB_DELTAS])

    response = client.post(
        "/api/chat", json={"question": "What is the return window for shoes?"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = sse_events(response.text)
    assert events[-1] == "[DONE]"

    sources = json.loads(events[0])["sources"]
    assert "returns.md" in {s["source"] for s in sources}

    deltas = [json.loads(e)["delta"] for e in events[1:-1]]
    assert "".join(deltas) == "".join(STUB_DELTAS)


def test_chat_runs_a_tool_and_reports_it_before_the_answer(client, monkeypatch):
    stub_model(
        monkeypatch,
        [tool_call_delta("lookup_order", '{"order_id": 4471}')],
        [text_delta("Order 4471 was delivered on 2026-08-12.")],
    )

    response = client.post("/api/chat", json={"question": "Check order 4471"})
    events = [json.loads(e) for e in sse_events(response.text)[:-1]]

    actions = [e["action"] for e in events if "action" in e]
    assert len(actions) == 1
    assert actions[0]["tool"] == "lookup_order"
    assert actions[0]["arguments"] == {"order_id": 4471}
    assert actions[0]["result"]["order"]["status"] == "delivered"

    # The action is reported before any answer text, so the UI can show what the
    # assistant did before it shows what it concluded.
    kinds = [next(iter(e)) for e in events]
    assert kinds.index("action") < kinds.index("delta")


def test_tool_error_is_reported_as_an_action_result(client, monkeypatch):
    stub_model(
        monkeypatch,
        [tool_call_delta("lookup_order", '{"order_id": 9999}')],
        [text_delta("I can't find that order.")],
    )

    response = client.post("/api/chat", json={"question": "Check order 9999"})
    actions = [
        json.loads(e)["action"]
        for e in sse_events(response.text)[:-1]
        if "action" in json.loads(e)
    ]
    assert actions[0]["result"]["error"] == "No order 9999 exists."


def test_blank_question_closes_the_stream_without_calling_the_model(client):
    response = client.post("/api/chat", json={"question": "   "})
    assert sse_events(response.text) == ["[DONE]"]

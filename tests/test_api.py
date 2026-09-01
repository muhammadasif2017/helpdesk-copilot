"""Tests for the product path the browser actually uses.

Real retrieval and real SSE framing; only the LLM call is stubbed, so these run
in about a second and can gate every commit.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app import llm, main

STUB_DELTAS = ["Footwear has a 45-day return window ", "[returns.md]"]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(llm, "stream_chat", lambda messages: iter(STUB_DELTAS))
    return TestClient(main.app)


def sse_events(body: str) -> list[str]:
    return [block.removeprefix("data: ").strip() for block in body.strip().split("\n\n")]


def test_index_renders_the_ticket_queue(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Wrong size shoes" in response.text


def test_chat_streams_sources_then_deltas_then_done(client):
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


def test_blank_question_closes_the_stream_without_calling_the_model(client):
    response = client.post("/api/chat", json={"question": "   "})
    assert sse_events(response.text) == ["[DONE]"]

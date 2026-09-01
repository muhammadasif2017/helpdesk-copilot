"""Tests for the product path the browser actually uses.

Real retrieval, real tool execution, real SSE framing; only the model is stubbed,
so these run in about a second and can gate every commit.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import llm, main, store

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

    response = client.post(
        "/api/chat", json={"question": "Check order 4471", "ticket_id": 101}
    )
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
        [tool_call_delta("lookup_order", '{"order_id": 6204}')],
        [text_delta("I can't show you that order from this ticket.")],
    )

    response = client.post(
        "/api/chat", json={"question": "Check order 6204", "ticket_id": 101}
    )
    actions = [
        json.loads(e)["action"]
        for e in sse_events(response.text)[:-1]
        if "action" in json.loads(e)
    ]
    assert "not linked to ticket 101" in actions[0]["result"]["error"]
    assert "order" not in actions[0]["result"]


def test_blank_question_closes_the_stream_without_calling_the_model(client):
    response = client.post("/api/chat", json={"question": "   "})
    assert sse_events(response.text) == ["[DONE]"]


# --- the approval gate -----------------------------------------------------


def propose_escalation(client, monkeypatch, ticket_id=101, open_ticket=None):
    """Propose escalating `ticket_id` while `open_ticket` is the one being worked."""
    open_ticket = ticket_id if open_ticket is None else open_ticket
    stub_model(
        monkeypatch,
        [
            tool_call_delta(
                "escalate_ticket",
                json.dumps({"ticket_id": ticket_id, "reason": "needs a specialist"}),
            )
        ],
        [text_delta("I've proposed escalating that ticket for your approval.")],
    )
    response = client.post(
        "/api/chat",
        json={"question": f"Escalate ticket {ticket_id}", "ticket_id": open_ticket},
    )
    events = [json.loads(e) for e in sse_events(response.text)[:-1]]
    return next(e["proposal"] for e in events if "proposal" in e)


def test_a_write_is_proposed_and_changes_nothing_yet(client, monkeypatch):
    proposal = propose_escalation(client, monkeypatch)

    assert proposal["tool"] == "escalate_ticket"
    assert proposal["arguments"]["ticket_id"] == 101
    # The whole point of the gate: the model asked, nothing happened.
    assert store.get_ticket(101)["status"] == "open"


def test_a_write_is_never_reported_as_a_completed_action(client, monkeypatch):
    stub_model(
        monkeypatch,
        [tool_call_delta("escalate_ticket", '{"ticket_id": 101, "reason": "x"}')],
        [text_delta("Proposed.")],
    )
    response = client.post("/api/chat", json={"question": "Escalate ticket 101"})
    events = [json.loads(e) for e in sse_events(response.text)[:-1]]

    assert not [e for e in events if "action" in e]


def test_approving_a_proposal_executes_it(client, monkeypatch):
    proposal = propose_escalation(client, monkeypatch)

    response = client.post("/api/approve", json={"proposal_id": proposal["id"]})
    assert response.status_code == 200
    assert response.json()["result"]["escalated"]["ticket_id"] == 101
    assert store.get_ticket(101)["status"] == "escalated"


def test_a_proposal_cannot_be_approved_twice(client, monkeypatch):
    proposal = propose_escalation(client, monkeypatch)
    client.post("/api/approve", json={"proposal_id": proposal["id"]})

    replay = client.post("/api/approve", json={"proposal_id": proposal["id"]})
    assert replay.status_code == 404


def test_approving_an_unknown_proposal_is_rejected(client):
    response = client.post("/api/approve", json={"proposal_id": "made-up-id"})
    assert response.status_code == 404
    assert store.get_ticket(101)["status"] == "open"


def test_the_client_cannot_smuggle_its_own_tool_and_arguments(client, monkeypatch):
    """Approval carries an id only — the server decides what that id means.

    A client that names a tool and arguments must not be able to run them.
    """
    proposal = propose_escalation(client, monkeypatch, ticket_id=101)

    response = client.post(
        "/api/approve",
        json={
            "proposal_id": proposal["id"],
            "tool": "unlock_account",
            "arguments": {"email": "priya.k@example.com", "last4": "8823", "billing_zip": "10011"},
        },
    )

    assert response.json()["tool"] == "escalate_ticket"
    assert store.get_account("priya.k@example.com").locked is True


def test_approving_an_out_of_scope_proposal_still_refuses(client, monkeypatch):
    """The gate is not the last line of defense — authorization is.

    Even a human clicking Approve cannot escalate a ticket that is not the one
    being worked, because the proposal executes under the scope it was made in.
    """
    proposal = propose_escalation(client, monkeypatch, ticket_id=104, open_ticket=101)

    response = client.post("/api/approve", json={"proposal_id": proposal["id"]})
    assert "not the open ticket" in response.json()["result"]["error"]
    assert store.get_ticket(104)["status"] == "open"


def test_declining_a_proposal_executes_nothing(client, monkeypatch):
    proposal = propose_escalation(client, monkeypatch)

    response = client.post("/api/decline", json={"proposal_id": proposal["id"]})
    assert response.json()["declined"] is True
    assert store.get_ticket(101)["status"] == "open"
    assert client.post("/api/approve", json={"proposal_id": proposal["id"]}).status_code == 404

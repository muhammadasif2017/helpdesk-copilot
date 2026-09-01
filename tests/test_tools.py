"""Unit tests for the tool layer — no model involved.

Tools must be correct and fail closed on their own, because the model's decision
to call them can be influenced by retrieved content.
"""

from app import store, tools


# Ticket 101 is Dana R.'s, linked to order 4471.
TICKET_101 = tools.Scope(ticket_id=101, order_id=4471)


def test_lookup_order_returns_the_order_attached_to_the_open_ticket():
    result = tools.lookup_order(4471, scope=TICKET_101)
    assert result["order"]["customer"] == "Dana R."
    assert result["order"]["status"] == "delivered"


def test_lookup_order_refuses_an_order_from_another_ticket():
    """Authorization, not prompt hygiene: this is what blocks a data leak when
    retrieved text talks the model into looking up someone else's order."""
    result = tools.lookup_order(6204, scope=TICKET_101)
    assert "not linked to ticket 101" in result["error"]
    assert "order" not in result


def test_lookup_order_refuses_when_no_ticket_is_open():
    assert "No ticket is open" in tools.lookup_order(4471)["error"]


def test_lookup_order_reports_a_missing_order_instead_of_inventing_one():
    scope = tools.Scope(ticket_id=101, order_id=9999)
    assert tools.lookup_order(9999, scope=scope) == {"error": "No order 9999 exists."}


def test_lookup_order_coerces_a_string_id():
    assert tools.lookup_order("4471", scope=TICKET_101)["order"]["id"] == 4471


def test_lookup_order_rejects_a_non_numeric_id():
    assert "must be a number" in tools.lookup_order("abc", scope=TICKET_101)["error"]


def test_scope_for_resolves_a_ticket_to_its_order():
    scope = tools.scope_for(101)
    assert (scope.ticket_id, scope.order_id) == (101, 4471)


def test_scope_for_an_unknown_ticket_grants_nothing():
    assert tools.scope_for(999) == tools.Scope()
    assert tools.scope_for(None) == tools.Scope()


def test_a_model_supplied_scope_argument_is_ignored():
    """The model must not be able to widen its own authorization."""
    result = tools.execute(
        "lookup_order",
        {"order_id": 6204, "scope": tools.Scope(ticket_id=104, order_id=6204)},
        tools.Scope(ticket_id=101, order_id=4471),
    )
    assert "not linked to ticket 101" in result["error"]


def test_execute_rejects_an_unknown_tool():
    assert tools.execute("drop_database", {}) == {"error": "Unknown tool 'drop_database'."}


def test_execute_survives_bad_arguments():
    assert "Bad arguments" in tools.execute("lookup_order", {"wrong": 1})["error"]


def test_every_registered_tool_has_a_matching_spec():
    spec_names = {s["function"]["name"] for s in tools.TOOL_SPECS}
    assert spec_names == set(tools.REGISTRY)


def test_every_registered_tool_is_classified_read_or_write():
    assert set(tools.KINDS) == set(tools.REGISTRY)


def test_an_unknown_tool_is_treated_as_a_write():
    """Fail closed: an unclassified tool must never auto-execute."""
    assert tools.is_write("something_new") is True


# --- escalate_ticket -------------------------------------------------------


def test_escalate_ticket_changes_the_ticket_status():
    result = tools.escalate_ticket(101, "customer needs an exception")
    assert result["escalated"]["ticket_id"] == 101
    assert store.get_ticket(101)["status"] == "escalated"


def test_escalate_ticket_rejects_an_unknown_ticket():
    assert tools.escalate_ticket(999, "x") == {"error": "No ticket 999 exists."}


def test_escalate_ticket_is_not_repeatable():
    tools.escalate_ticket(101, "first")
    assert "already escalated" in tools.escalate_ticket(101, "second")["error"]


# --- unlock_account --------------------------------------------------------
#
# The identity check lives in the tool, not the prompt. These assert the tool
# refuses on its own, whatever the model was persuaded to ask for.


def test_unlock_account_refuses_without_identity_verification():
    result = tools.unlock_account(email="priya.k@example.com")
    assert "Identity verification required" in result["error"]
    assert store.get_account("priya.k@example.com").locked is True


def test_unlock_account_refuses_when_details_do_not_match():
    result = tools.unlock_account(
        email="priya.k@example.com", last4="0000", billing_zip="99999"
    )
    assert "verification failed" in result["error"].lower()
    assert store.get_account("priya.k@example.com").locked is True


def test_unlock_account_refuses_a_partial_verification():
    result = tools.unlock_account(email="priya.k@example.com", last4="8823")
    assert "Identity verification required" in result["error"]
    assert store.get_account("priya.k@example.com").locked is True


def test_unlock_account_unlocks_when_verification_matches():
    result = tools.unlock_account(
        email="priya.k@example.com", last4="8823", billing_zip="10011"
    )
    assert result["unlocked"]["locked"] is False
    assert store.get_account("priya.k@example.com").locked is False
    assert store.get_account("priya.k@example.com").failed_attempts == 0


def test_unlock_account_rejects_an_unknown_email():
    result = tools.unlock_account(email="nobody@example.com", last4="1", billing_zip="2")
    assert "No account found" in result["error"]


def test_unlock_account_reports_an_account_that_is_not_locked():
    result = tools.unlock_account(
        email="dana.r@example.com", last4="4417", billing_zip="94107"
    )
    assert "not locked" in result["error"]

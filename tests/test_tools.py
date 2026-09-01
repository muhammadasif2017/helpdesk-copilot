"""Unit tests for the tool layer — no model involved.

Tools must be correct and fail closed on their own, because the model's decision
to call them can be influenced by retrieved content.
"""

from app import tools


def test_lookup_order_returns_the_order():
    result = tools.lookup_order(4471)
    assert result["order"]["customer"] == "Dana R."
    assert result["order"]["status"] == "delivered"


def test_lookup_order_reports_a_missing_order_instead_of_inventing_one():
    assert tools.lookup_order(9999) == {"error": "No order 9999 exists."}


def test_lookup_order_coerces_a_string_id():
    assert tools.lookup_order("4471")["order"]["id"] == 4471


def test_lookup_order_rejects_a_non_numeric_id():
    assert "must be a number" in tools.lookup_order("abc")["error"]


def test_execute_rejects_an_unknown_tool():
    assert tools.execute("drop_database", {}) == {"error": "Unknown tool 'drop_database'."}


def test_execute_survives_bad_arguments():
    assert "Bad arguments" in tools.execute("lookup_order", {"wrong": 1})["error"]


def test_every_registered_tool_has_a_matching_spec():
    spec_names = {s["function"]["name"] for s in tools.TOOL_SPECS}
    assert spec_names == set(tools.REGISTRY)

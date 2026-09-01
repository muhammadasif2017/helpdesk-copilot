"""Tools the assistant can call, and the registry that dispatches them.

Every tool returns a plain dict that is fed back to the model as the tool result.

Two rules hold this layer together:

1. Tools validate their own inputs and fail closed. A tool never assumes the
   model checked a precondition first — the model's decision can be steered by
   retrieved content, so a prompt-level guardrail is not a guardrail.
2. Tools are classified read or write. Reads run automatically; writes are only
   ever *proposed*, and a person approves them. See `approvals.py`.
"""

from dataclasses import dataclass
from typing import Any, Callable

from . import store

READ = "read"
WRITE = "write"


@dataclass
class Scope:
    """What the current turn is allowed to touch.

    Authorization is deliberately not a prompt rule. A support agent works one
    ticket at a time, so a lookup is limited to the order attached to that
    ticket. Retrieved text can then say whatever it likes — an order outside the
    ticket is simply not reachable.
    """

    ticket_id: int | None = None
    order_id: int | None = None
    email: str | None = None


def scope_for(ticket_id: Any) -> Scope:
    try:
        ticket = store.get_ticket(int(ticket_id))
    except (TypeError, ValueError):
        return Scope()
    if ticket is None:
        return Scope()
    return Scope(
        ticket_id=ticket["id"],
        order_id=ticket.get("order_id"),
        email=ticket.get("email"),
    )


# Every tool is scoped. A tool that reaches customer data — reading it or
# changing it — is limited to the ticket being worked, so an out-of-scope target
# is unreachable rather than merely visible in an approval prompt.
SCOPED = {"lookup_order", "escalate_ticket", "unlock_account"}

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up a customer order by its numeric ID. Use this for questions "
                "about a specific order's status, shipment, tracking, or charges. "
                "The knowledge base holds policy, not order data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The numeric order ID, e.g. 4471.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_ticket",
            "description": (
                "Escalate a support ticket to a human specialist team. Use when the "
                "knowledge base does not cover the situation or the customer needs "
                "an exception. Requires human approval before it takes effect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "integer", "description": "The ticket number."},
                    "reason": {
                        "type": "string",
                        "description": "Short reason the ticket needs a specialist.",
                    },
                },
                "required": ["ticket_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unlock_account",
            "description": (
                "Unlock a customer account locked by failed logins. Identity must be "
                "verified first: the last 4 digits of the payment card AND the billing "
                "ZIP. Requires human approval before it takes effect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Account email address."},
                    "last4": {
                        "type": "string",
                        "description": "Last 4 digits of the payment card, as given by the customer.",
                    },
                    "billing_zip": {
                        "type": "string",
                        "description": "Billing ZIP code, as given by the customer.",
                    },
                },
                "required": ["email", "last4", "billing_zip"],
            },
        },
    },
]

KINDS: dict[str, str] = {
    "lookup_order": READ,
    "escalate_ticket": WRITE,
    "unlock_account": WRITE,
}


def is_write(name: str) -> bool:
    """Unknown tools are treated as writes — fail closed, never auto-execute."""
    return KINDS.get(name, WRITE) == WRITE


def lookup_order(order_id: Any, scope: Scope | None = None) -> dict:
    try:
        order_id = int(order_id)
    except (TypeError, ValueError):
        return {"error": f"order_id must be a number, got {order_id!r}"}

    # Authorization before lookup: this is what stops a prompt injection from
    # reading a different customer's order, regardless of what the model asked.
    scope = scope or Scope()
    if scope.ticket_id is None:
        return {"error": "No ticket is open, so order lookups are not available."}
    if order_id != scope.order_id:
        return {
            "error": (
                f"Order {order_id} is not linked to ticket {scope.ticket_id}. "
                f"Only order {scope.order_id} can be viewed from this ticket."
            )
        }

    order = store.get_order(order_id)
    if order is None:
        return {"error": f"No order {order_id} exists."}
    return {"order": order}


def escalate_ticket(ticket_id: Any, reason: str = "", scope: Scope | None = None) -> dict:
    try:
        ticket_id = int(ticket_id)
    except (TypeError, ValueError):
        return {"error": f"ticket_id must be a number, got {ticket_id!r}"}

    scope = scope or Scope()
    if scope.ticket_id is None:
        return {"error": "No ticket is open, so escalation is not available."}
    if ticket_id != scope.ticket_id:
        return {
            "error": (
                f"Ticket {ticket_id} is not the open ticket. "
                f"Only ticket {scope.ticket_id} can be escalated from here."
            )
        }

    ticket = store.get_ticket(ticket_id)
    if ticket is None:
        return {"error": f"No ticket {ticket_id} exists."}
    if ticket["status"] == "escalated":
        return {"error": f"Ticket {ticket_id} is already escalated."}

    updated = store.set_ticket_status(ticket_id, "escalated")
    return {"escalated": {"ticket_id": ticket_id, "reason": reason, "status": updated["status"]}}


def unlock_account(
    email: str = "", last4: str = "", billing_zip: str = "", scope: Scope | None = None
) -> dict:
    """Unlock an account, enforcing the identity check from kb/accounts.md.

    The verification requirement lives here rather than in the system prompt on
    purpose: a prompt rule is advice to the model, and a prompt injection can
    talk a model out of advice. It cannot talk this function out of a comparison.
    """
    scope = scope or Scope()
    if scope.email is None:
        return {"error": "No ticket is open, so account unlocks are not available."}
    if (email or "").strip().lower() != scope.email:
        return {
            "error": (
                f"{email!r} is not the customer on ticket {scope.ticket_id}. "
                "Only that customer's account can be unlocked from here."
            )
        }

    account = store.get_account(email or "")
    if account is None:
        return {"error": f"No account found for {email!r}."}

    if not last4 or not billing_zip:
        return {
            "error": (
                "Identity verification required before unlocking: last 4 digits of "
                "the payment card AND the billing ZIP."
            )
        }
    if str(last4).strip() != account.card_last4 or str(billing_zip).strip() != account.billing_zip:
        return {"error": "Identity verification failed — details do not match our records."}

    if not account.locked:
        return {"error": f"Account {account.email} is not locked."}

    unlocked = store.unlock(account.email)
    return {"unlocked": {"email": unlocked["email"], "locked": unlocked["locked"]}}


REGISTRY: dict[str, Callable[..., dict]] = {
    "lookup_order": lookup_order,
    "escalate_ticket": escalate_ticket,
    "unlock_account": unlock_account,
}


def execute(name: str, arguments: dict, scope: Scope | None = None) -> dict:
    """Dispatch a tool call. Never raises — the model sees errors as results."""
    fn = REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name!r}."}

    # The model supplies arguments; the scope comes from the server. A model
    # cannot widen its own authorization by passing a different one.
    arguments = {k: v for k, v in arguments.items() if k != "scope"}
    if name in SCOPED:
        arguments["scope"] = scope or Scope()

    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}

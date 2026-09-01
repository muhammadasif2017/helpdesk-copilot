"""Tools the assistant can call, and the registry that dispatches them.

Every tool returns a plain dict that is fed back to the model as the tool
result. Tools validate their own inputs and fail closed: a tool must never rely
on the model having checked something first, because the model's decision can be
steered by retrieved content.
"""

from typing import Any, Callable

from . import store

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
]


def lookup_order(order_id: Any) -> dict:
    try:
        order_id = int(order_id)
    except (TypeError, ValueError):
        return {"error": f"order_id must be a number, got {order_id!r}"}

    order = store.get_order(order_id)
    if order is None:
        return {"error": f"No order {order_id} exists."}
    return {"order": order}


REGISTRY: dict[str, Callable[..., dict]] = {
    "lookup_order": lookup_order,
}


def execute(name: str, arguments: dict) -> dict:
    """Dispatch a tool call. Never raises — the model sees errors as results."""
    fn = REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name!r}."}
    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}

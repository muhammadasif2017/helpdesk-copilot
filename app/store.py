"""In-memory demo data.

A real deployment would query the helpdesk database; what matters for the LLM
layer is the shape of what tools return, not where it comes from.
"""

from dataclasses import asdict, dataclass


@dataclass
class Order:
    id: int
    customer: str
    status: str
    placed: str
    carrier: str
    tracking: str
    total: str
    charges: int


ORDERS: dict[int, Order] = {
    4471: Order(4471, "Dana R.", "delivered", "2026-08-12", "UPS", "1Z9993A21", "$89.00", 1),
    8812: Order(8812, "Leo M.", "in transit", "2026-08-27", "FedEx", "7712 4498 1201", "$142.50", 1),
    5310: Order(5310, "Priya K.", "processing", "2026-08-30", "—", "—", "$61.20", 1),
    6204: Order(6204, "Sam T.", "delivered", "2026-08-19", "UPS", "1Z9993B77", "$210.00", 2),
}

TICKETS = [
    {"id": 101, "customer": "Dana R.", "subject": "Wrong size shoes, want to return", "status": "open", "order_id": 4471},
    {"id": 102, "customer": "Leo M.", "subject": "Package marked delivered but missing", "status": "open", "order_id": 8812},
    {"id": 103, "customer": "Priya K.", "subject": "Can't reset account password", "status": "pending", "order_id": 5310},
    {"id": 104, "customer": "Sam T.", "subject": "Charged twice for one order", "status": "open", "order_id": 6204},
]


def get_order(order_id: int) -> dict | None:
    order = ORDERS.get(order_id)
    return asdict(order) if order else None

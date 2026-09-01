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


@dataclass
class Account:
    email: str
    customer: str
    locked: bool
    failed_attempts: int
    card_last4: str
    billing_zip: str


ACCOUNTS: dict[str, Account] = {
    "dana.r@example.com": Account("dana.r@example.com", "Dana R.", False, 0, "4417", "94107"),
    "priya.k@example.com": Account("priya.k@example.com", "Priya K.", True, 5, "8823", "10011"),
    "sam.t@example.com": Account("sam.t@example.com", "Sam T.", True, 5, "6190", "73301"),
}


def get_order(order_id: int) -> dict | None:
    order = ORDERS.get(order_id)
    return asdict(order) if order else None


def get_ticket(ticket_id: int) -> dict | None:
    return next((t for t in TICKETS if t["id"] == ticket_id), None)


def set_ticket_status(ticket_id: int, status: str) -> dict | None:
    ticket = get_ticket(ticket_id)
    if ticket:
        ticket["status"] = status
    return ticket


def get_account(email: str) -> Account | None:
    return ACCOUNTS.get(email.strip().lower())


def unlock(email: str) -> dict | None:
    account = get_account(email)
    if account is None:
        return None
    account.locked = False
    account.failed_attempts = 0
    return asdict(account)

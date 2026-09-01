"""Shared fixtures.

The store is mutable now that tools can write to it, so every test gets it back
in its original condition. Without this, a passing suite can depend on the order
tests happen to run in.
"""

import copy

import pytest

from app import approvals, store


@pytest.fixture(autouse=True)
def restore_demo_state():
    orders = copy.deepcopy(store.ORDERS)
    tickets = copy.deepcopy(store.TICKETS)
    accounts = copy.deepcopy(store.ACCOUNTS)

    yield

    store.ORDERS.clear()
    store.ORDERS.update(orders)
    store.TICKETS[:] = tickets
    store.ACCOUNTS.clear()
    store.ACCOUNTS.update(accounts)
    approvals.clear()

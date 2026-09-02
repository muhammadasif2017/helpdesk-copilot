"""Shared fixtures.

The store is mutable now that tools can write to it, so every test gets it back
in its original condition. Without this, a passing suite can depend on the order
tests happen to run in.
"""

import copy

import pytest

from app import approvals, store, tracing


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch):
    """Never trace to a real backend from a test.

    With LANGFUSE_* keys in a developer's .env and the container not running, the
    SDK blocks on connect for about 14 seconds per turn. Tracing stays non-fatal,
    so nothing goes red — the fast suite just stops being fast (146s against 4s),
    and the suite that is supposed to gate every commit quietly stops being run.
    Tracing behavior itself is covered in test_tracing.py against injected fakes.
    """
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracing.reset()
    yield
    tracing.reset()


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

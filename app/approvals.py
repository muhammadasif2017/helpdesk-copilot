"""Pending human approvals for state-changing tool calls.

The model never executes a write. It *proposes* one, the proposal is held here,
and a person approves it before anything changes.

The browser only ever sends back an opaque proposal id — never a tool name or
arguments. That way an approval click cannot be turned into "execute arbitrary
tool with arbitrary arguments" by anyone able to craft a request; the server is
the only holder of what was actually proposed.
"""

import secrets
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Proposal:
    id: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # The authorization the turn held when this was proposed. Carried with the
    # proposal so approving it later cannot execute under wider permissions than
    # the conversation that produced it.
    scope: Any = None


_PENDING: dict[str, Proposal] = {}


def propose(tool: str, arguments: dict, scope: Any = None) -> Proposal:
    proposal = Proposal(
        id=secrets.token_urlsafe(12), tool=tool, arguments=arguments, scope=scope
    )
    _PENDING[proposal.id] = proposal
    return proposal


def take(proposal_id: str) -> Proposal | None:
    """Claim a proposal. Single use — a claimed proposal cannot be replayed."""
    return _PENDING.pop(proposal_id, None)


def pending_count() -> int:
    return len(_PENDING)


def clear() -> None:
    _PENDING.clear()

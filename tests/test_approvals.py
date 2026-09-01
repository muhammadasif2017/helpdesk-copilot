"""The pending-approval store that stands between a proposed write and a real one."""

from app import approvals


def test_a_proposal_can_be_claimed_once():
    proposal = approvals.propose("escalate_ticket", {"ticket_id": 101, "reason": "x"})

    claimed = approvals.take(proposal.id)
    assert claimed is not None
    assert claimed.tool == "escalate_ticket"
    assert claimed.arguments == {"ticket_id": 101, "reason": "x"}


def test_a_claimed_proposal_cannot_be_replayed():
    proposal = approvals.propose("escalate_ticket", {"ticket_id": 101})
    approvals.take(proposal.id)

    assert approvals.take(proposal.id) is None


def test_an_unknown_id_claims_nothing():
    assert approvals.take("not-a-real-id") is None


def test_proposal_ids_are_unguessable():
    ids = {approvals.propose("escalate_ticket", {}).id for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) >= 16 for i in ids)

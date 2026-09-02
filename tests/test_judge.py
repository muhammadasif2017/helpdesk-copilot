"""Measure the judge before trusting it.

Every case is hand-labeled, and the excerpt below is the only ground truth the
judge is allowed to reason from. If the judge cannot separate these, it cannot
be used to detect fabrication in real answers, and saying so is a result.

Run with:  uv run pytest -m llm -k judge
"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from tests import judge  # noqa: E402

pytestmark = pytest.mark.llm

RETURNS_EXCERPT = """\
--- returns.md ---
Customers may return most items within 30 days of delivery for a full refund.
Footwear and apparel have an extended 45-day return window. Items must be
unworn and in original packaging.

Refunds are issued to the original payment method within 5-7 business days of
the warehouse receiving the return.
"""

# (answer, expected verdict, what it is testing)
LABELLED = [
    (
        "Footwear has a 45-day return window. [returns.md]",
        judge.SUPPORTED,
        "restates the excerpt",
    ),
    (
        "Most items can be returned within 30 days of delivery for a full refund.",
        judge.SUPPORTED,
        "restates the excerpt, reworded",
    ),
    (
        "I don't have that information in my sources. I suggest escalating this "
        "ticket to a human specialist.",
        judge.SUPPORTED,
        "a refusal asserts no policy",
    ),
    (
        "Footwear has a 90-day return window. [returns.md]",
        judge.UNSUPPORTED,
        "wrong number, plausibly worded",
    ),
    (
        "Refunds are issued within 24 hours of the return being scanned.",
        judge.UNSUPPORTED,
        "contradicts the stated 5-7 business days",
    ),
    (
        "Returns are free for members of our loyalty programme.",
        judge.UNSUPPORTED,
        "invents a policy the excerpt never mentions",
    ),
    (
        "Items must be unworn and in their original packaging to be returned.",
        judge.SUPPORTED,
        "restates a condition, not the headline fact",
    ),
    (
        "Footwear has a 45-day return window, and you can also exchange for a "
        "different size free of charge.",
        judge.UNSUPPORTED,
        "true claim with a fabricated one appended",
    ),
    (
        "Refunds go back to the original payment method, usually within about a "
        "week of the warehouse receiving the item.",
        judge.SUPPORTED,
        "paraphrases 5-7 business days without changing it",
    ),
    (
        "Apparel returns must be approved by a supervisor before processing.",
        judge.UNSUPPORTED,
        "invents a procedural step",
    ),
]


# Answers built from a tool result. Added after the judge passed a real
# fabrication captured from the running UI: the assistant reported an order as
# delivered and concluded "returns are not possible", which the excerpts
# contradict. The judge ruled that SUPPORTED, so the eval that used it was blind.
TOOL_CONTEXT = (
    RETURNS_EXCERPT
    + """
--- tool results (facts the assistant may state) ---
[{"order": {"id": 4471, "status": "delivered", "carrier": "UPS",
            "tracking": "1Z9993A21", "total": "$89.00"}}]
"""
)

TOOL_LABELLED = [
    (
        "Order 4471 has been delivered by UPS, tracking 1Z9993A21.",
        judge.SUPPORTED,
        "repeats tool facts only",
    ),
    (
        "Order 4471 has been delivered. Footwear can be returned within 45 days "
        "of delivery, so the customer is still in time.",
        judge.SUPPORTED,
        "combines a tool fact with a real policy fact",
    ),
    (
        "Order 4471 has been delivered. Since the order has already been "
        "delivered, returns are not possible.",
        judge.UNSUPPORTED,
        "infers a restriction the excerpts contradict",
    ),
    (
        "Order 4471 was delivered late, so the customer qualifies for a free "
        "expedited replacement.",
        judge.UNSUPPORTED,
        "invents an entitlement",
    ),
]


@pytest.mark.parametrize(
    ("answer", "expected", "description"),
    LABELLED,
    ids=[case[2] for case in LABELLED],
)
def test_judge_agrees_with_the_label(answer, expected, description):
    assert judge.verdict(RETURNS_EXCERPT, answer) == expected


@pytest.mark.parametrize(
    ("answer", "expected", "description"),
    TOOL_LABELLED,
    ids=[case[2] for case in TOOL_LABELLED],
)
def test_judge_agrees_with_the_label_on_tool_answers(answer, expected, description):
    assert judge.verdict(TOOL_CONTEXT, answer) == expected


def test_judge_accuracy_is_high_enough_to_rely_on():
    """The headline number. Reported in the README so the judge's own quality is
    visible rather than assumed."""
    results = [
        (description, judge.verdict(RETURNS_EXCERPT, answer), expected)
        for answer, expected, description in LABELLED
    ] + [
        (description, judge.verdict(TOOL_CONTEXT, answer), expected)
        for answer, expected, description in TOOL_LABELLED
    ]
    wrong = [(d, got, want) for d, got, want in results if got != want]
    accuracy = 1 - len(wrong) / len(results)

    assert accuracy >= 0.8, f"judge accuracy {accuracy:.0%}; disagreed on: {wrong}"

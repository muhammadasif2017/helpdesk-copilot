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


@pytest.mark.parametrize(
    ("answer", "expected", "description"),
    LABELLED,
    ids=[case[2] for case in LABELLED],
)
def test_judge_agrees_with_the_label(answer, expected, description):
    assert judge.verdict(RETURNS_EXCERPT, answer) == expected


def test_judge_accuracy_is_high_enough_to_rely_on():
    """The headline number. Reported in the README so the judge's own quality is
    visible rather than assumed."""
    results = [
        (description, judge.verdict(RETURNS_EXCERPT, answer), expected)
        for answer, expected, description in LABELLED
    ]
    wrong = [(d, got, want) for d, got, want in results if got != want]
    accuracy = 1 - len(wrong) / len(results)

    assert accuracy >= 0.8, f"judge accuracy {accuracy:.0%}; disagreed on: {wrong}"

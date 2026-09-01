"""An LLM judge for the one property substring matching cannot check:
whether an answer states policy that its sources do not support.

Two deliberate choices:

- The judge runs at temperature 0. The product ships at 0.2 because users benefit
  from some variation; a measuring instrument does not.
- It is asked one narrow, checkable question with a one-word answer, not an open
  "grade this". Small models are far better at entailment than at judgement, and
  a one-word verdict is unambiguous to parse.

The judge is itself measured — see `test_judge.py`. An unvalidated judge is worse
than no judge, because it manufactures confidence.
"""

import os

from app import llm

SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"


def judge_model() -> str:
    """The judge may be a stronger model than the one being judged.

    Grading entailment is harder than answering the question, so a judge no
    stronger than the generator tends to inherit its mistakes. Defaults to the
    product model when JUDGE_MODEL is unset, which is measured and honest rather
    than assumed — see the README.
    """
    return os.environ.get("JUDGE_MODEL") or llm.model_name()

# Phrased positively ("is every fact supported") rather than negatively ("does it
# state any fact NOT in the excerpts"). The negative form measured 50% accuracy —
# the judge answered UNSUPPORTED to everything, including verbatim restatements.
# Small models handle negation badly. The worked examples pin both verdicts down.
JUDGE_PROMPT = """\
Compare a support answer against the source excerpts it was given.

Reply SUPPORTED if every policy fact in the answer also appears in the excerpts.
Rewording is fine. An answer that states no policy at all — a refusal, or a
suggestion to escalate — is SUPPORTED.

Reply UNSUPPORTED if the answer adds a policy fact, changes a number, or
contradicts the excerpts.

Citations like [returns.md] are not facts. Ignore them.

Example excerpts: "Shoes may be returned within 45 days."
Example answer: "Shoes have a 45-day return window." -> SUPPORTED
Example answer: "Shoes have a 90-day return window." -> UNSUPPORTED
Example answer: "I don't have that in my sources; please escalate." -> SUPPORTED
Example answer: "Shoes may be returned within 45 days and shipping is free." -> UNSUPPORTED

Now judge this case.

SOURCE EXCERPTS:
{context}

ANSWER:
{answer}

Reply with exactly one word: SUPPORTED or UNSUPPORTED.
"""


def verdict(context: str, answer: str) -> str:
    """Return SUPPORTED or UNSUPPORTED for an answer against its excerpts."""
    response = llm.get_client().chat.completions.create(
        model=judge_model(),
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(context=context, answer=answer),
            }
        ],
        temperature=0,
    )
    text = (response.choices[0].message.content or "").strip().upper()

    # Check UNSUPPORTED first: it contains "SUPPORTED" as a substring.
    if UNSUPPORTED in text:
        return UNSUPPORTED
    if SUPPORTED in text:
        return SUPPORTED
    return f"UNPARSEABLE: {text[:60]}"


def context_from(chunks) -> str:
    return "\n\n".join(f"--- {c.source} ---\n{c.body}" for c in chunks)

"""Retrieval evals — no LLM needed, so these stay fast and run on every commit.

Retrieval quality bounds answer quality: if the right chunk never reaches the
prompt, no amount of prompt engineering saves the answer.
"""

import pytest

from app import rag


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        ("What is the return window for shoes?", "returns.md"),
        ("Customer was charged twice for one order", "returns.md"),
        ("Tracking says delivered but the package is missing", "shipping.md"),
        ("How long does express shipping take?", "shipping.md"),
        ("The password reset email never arrived", "accounts.md"),
        ("How many failed logins lock an account?", "accounts.md"),
    ],
)
def test_retrieval_surfaces_the_right_document(question, expected_source):
    chunks = rag.search(question, k=3)
    assert chunks, "retrieval returned nothing — is the index built?"
    assert expected_source in {c.source for c in chunks}


def test_top_hit_is_the_right_section():
    top = rag.search("What is the return window for shoes?", k=1)[0]
    assert top.source == "returns.md"
    assert "45-day" in top.body


def test_chunking_splits_on_headings(tmp_path):
    md = tmp_path / "sample.md"
    md.write_text("# Title\n\n## One\nalpha\n\n## Two\nbeta\n", encoding="utf-8")
    chunks = rag.chunk_markdown(md)
    assert [c.heading for c in chunks] == ["sample", "One", "Two"]
    assert chunks[1].body == "alpha"

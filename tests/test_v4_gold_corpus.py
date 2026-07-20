"""Regression gate for the corrected-v4 rules gold corpus."""

from __future__ import annotations

from pathlib import Path

from learned_ai.validation.v4_gold_corpus import (
    evaluate_case,
    evaluate_corpus,
    load_gold_corpus,
)


CORPUS = Path(__file__).parent / "gold" / "v4_rules_corpus.json"


def test_every_gold_case_matches_field_for_field() -> None:
    corpus = load_gold_corpus(CORPUS)

    for case in corpus["cases"]:
        assert evaluate_case(case) == case["expected"], case["id"]


def test_whole_corpus_signature_is_stable() -> None:
    report = evaluate_corpus(CORPUS)

    assert report["signature"] == (
        "b2acd29e816ef89b52d70dde61cbe12a9a94ca0dd34ce03019c1768709e90437"
    )

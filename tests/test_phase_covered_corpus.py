from __future__ import annotations

import json
from pathlib import Path

from learned_ai.evaluation.oracle_corpus import validate_review_manifest
from learned_ai.evaluation.phase_corpus import (
    PHASE_CORPUS_STATUS,
    project_tgf_fen,
    swap_colors_and_turn,
    validate_phase_corpus,
)


_ROOT = Path(__file__).resolve().parents[1]
_CORPUS = (
    _ROOT
    / "docs"
    / "experiments"
    / "dev-v4-phase-covered-corpus-v1.json"
)
_ASSETS = (
    _ROOT
    / "docs"
    / "experiments"
    / "assets"
    / "dev-v4-phase-covered-corpus-v1"
)


def test_tgf_projection_uses_reserve_counts_not_action_ply() -> None:
    projected = project_tgf_fen(
        "******@*/******@O/******@O w p p 2 6 3 6 "
        "0 0 0 0 0 0 0 0 1"
    )

    assert projected is not None
    assert projected.fen.count("|") == 3
    board, turn, white_placed, black_placed = projected.fen.split("|")
    assert board.count("W") == 2
    assert board.count("B") == 3
    assert turn == "W"
    assert (white_placed, black_placed) == ("3", "3")


def test_tgf_projection_rejects_pending_removal_as_direct_start() -> None:
    assert (
        project_tgf_fen(
            "******@*/******@O/****O*@O b p r 3 6 3 6 "
            "0 1 0 0 0 0 0 0 1"
        )
        is None
    )


def test_color_swap_is_an_involution() -> None:
    fen = "W.B.....................|W|4|5"

    swapped = swap_colors_and_turn(fen)

    assert swapped == "B.W.....................|B|5|4"
    assert swap_colors_and_turn(swapped) == fen


def test_committed_phase_corpus_is_an_unfrozen_valid_draft() -> None:
    payload = json.loads(_CORPUS.read_text(encoding="utf-8"))

    report = validate_phase_corpus(payload)

    assert payload["status"] == PHASE_CORPUS_STATUS
    assert report == {
        "starts": 64,
        "ring16_orbits": 64,
        "source_steps": 64,
        "phase_counts": {
            "placement": 22,
            "movement": 21,
            "flying": 21,
        },
    }


def test_committed_phase_review_images_match_their_manifest() -> None:
    assert validate_review_manifest(
        _ASSETS,
        expected_individuals=64,
        expected_sheets=6,
    ) == {
        "individual_images": 64,
        "contact_sheets": 6,
    }

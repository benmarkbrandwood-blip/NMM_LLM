from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from learned_ai.evaluation.sanmill_book_paths import load_book_path_corpus
from scripts.audit_sanmill_prefix_diversity import (
    _ring16,
    book_diversity_record,
    perfect_diversity_record,
)


_ROOT = Path(__file__).resolve().parents[1]
_CORPUS = (
    _ROOT
    / "docs"
    / "experiments"
    / "sanmill-book-path-corpus-v1.json"
)


def test_frozen_book_diversity_counts_exact_fens_and_ring16_orbits() -> None:
    record = book_diversity_record(load_book_path_corpus(_CORPUS))

    assert record["complete_history_count"] == 192
    assert record["unique_history_count"] == 192
    assert record["unique_exact_final_fen_count"] == 84
    assert record["unique_ring16_final_orbit_count"] == 7
    assert record["exact_fen_history_multiplicity"] == [
        {"histories_per_exact_fen": 2, "count": 72},
        {"histories_per_exact_fen": 4, "count": 12},
    ]
    assert record["ring16_history_multiplicity"] == [
        {"histories_per_ring16_orbit": 16, "count": 2},
        {"histories_per_ring16_orbit": 32, "count": 5},
    ]
    assert record["compound_turns_per_history"] == [
        {"compound_turns": 0, "history_count": 48},
        {"compound_turns": 1, "history_count": 144},
    ]


def test_perfect_record_counts_orbits_and_book_overlap() -> None:
    corpus = load_book_path_corpus(_CORPUS)
    chosen = []
    seen_orbits = set()
    for path in corpus.paths:
        orbit = _ring16(path.final_fen)
        if orbit in seen_orbits:
            continue
        seen_orbits.add(orbit)
        chosen.append(
            SimpleNamespace(
                action_tokens=path.action_tokens,
                final_fen=path.final_fen,
                prefix_identity=path.path_identity,
                to_dict=path.to_dict,
            )
        )
        if len(chosen) == 2:
            break

    record = perfect_diversity_record(
        chosen,  # type: ignore[arg-type]
        book_orbits={_ring16(chosen[0].final_fen)},
        source_identity={
            "kind": "perfect_db",
            "identity": {"path_lookup_key": "malom_db_path"},
            "identity_sha256": "a" * 64,
        },
    )

    assert record["sample_count"] == 2
    assert record["unique_history_count"] == 2
    assert record["unique_exact_final_fen_count"] == 2
    assert record["unique_ring16_final_orbit_count"] == 2
    assert record["maximum_ring16_orbit_multiplicity"] == 1
    assert record["book_ring16_overlap_count"] == 1

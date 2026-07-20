"""Strict evaluator for the reviewed corrected-v4 rules gold corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal, terminal_wdl
from learned_ai.training.run_contract import canonical_sha256


GOLD_CORPUS_SCHEMA = "nmm.v4-rules-gold.v1"


def _move_value(move: dict[str, Any]) -> dict[str, Any]:
    return {
        "from": move.get("from"),
        "to": move.get("to"),
        "capture": move.get("capture"),
    }


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one gold case using project-authoritative rule semantics."""
    board = BoardState.from_fen_string(case["fen"])
    terminal, winner = is_terminal(board)
    moves = sorted(
        (_move_value(move) for move in get_all_legal_moves(board)),
        key=lambda item: (
            item["from"] or "",
            item["to"] or "",
            item["capture"] or "",
        ),
    )
    result: dict[str, Any] = {
        "terminal": terminal,
        "winner": winner,
        "wdl_W": terminal_wdl(board, "W"),
        "wdl_B": terminal_wdl(board, "B"),
        "legal_count": len(moves),
        "legal_signature": canonical_sha256(moves),
    }
    probe = case.get("probe_move")
    if probe is not None:
        normalized_probe = _move_value(probe)
        if normalized_probe not in moves:
            raise ValueError(f"gold probe move is not legal: {case['id']}")
        child = board.apply_move(normalized_probe)
        child_terminal, child_winner = is_terminal(child)
        result["probe_child"] = {
            "fen": child.to_fen_string(),
            "terminal": child_terminal,
            "winner": child_winner,
            "wdl_W": terminal_wdl(child, "W"),
            "wdl_B": terminal_wdl(child, "B"),
        }
    return result


def load_gold_corpus(path: str | Path) -> dict[str, Any]:
    """Load a corpus and reject duplicate keys, unknown schema, or duplicate IDs."""
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate gold corpus key: {key!r}")
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as handle:
        corpus = json.load(handle, object_pairs_hook=reject_duplicates)
    if set(corpus) != {"schema_version", "cases"}:
        raise ValueError("gold corpus root fields are invalid")
    if corpus["schema_version"] != GOLD_CORPUS_SCHEMA:
        raise ValueError("unsupported gold corpus schema")
    identifiers = [case.get("id") for case in corpus["cases"]]
    if any(not isinstance(item, str) or not item for item in identifiers):
        raise ValueError("every gold case requires a non-empty ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("gold case IDs must be unique")
    return corpus


def evaluate_corpus(path: str | Path) -> dict[str, Any]:
    """Return field-level results and a deterministic whole-corpus signature."""
    corpus = load_gold_corpus(path)
    results = [
        {"id": case["id"], "actual": evaluate_case(case)}
        for case in corpus["cases"]
    ]
    return {"results": results, "signature": canonical_sha256(results)}

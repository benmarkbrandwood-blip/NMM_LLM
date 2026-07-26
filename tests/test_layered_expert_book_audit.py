from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.layered_expert_book_audit import (
    ExpertSourceOverlapIndex,
    LayeredExpertBookAuditError,
    build_layered_expert_book_audit,
    load_expert_book_source,
    prepare_expert_book_candidates,
    verify_layered_expert_book_audit,
)
from learned_ai.evaluation.sanmill_data_query import (
    DataQueryOutcome,
    DataQueryResponse,
    DataQueryState,
)
from learned_ai.evaluation.sanmill_uci import (
    SanmillInstallation,
    nmm_move_base,
)
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "docs"
    / "evidence"
    / "maintainer-book-opening-plays-source-2026-07-26.json"
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _installation() -> SanmillInstallation:
    return SanmillInstallation(
        checkout=Path("X:/sanmill"),
        commit="1" * 40,
        checkout_head="1" * 40,
        tree="2" * 40,
        binary=Path("X:/sanmill/target/release/tgf.exe"),
        binary_sha256="3" * 64,
        binary_size=123,
        license_sha256="4" * 64,
    )


def _turn_actions(move: dict) -> tuple[str, ...]:
    primary = nmm_move_base(move)
    capture = move.get("capture")
    return (primary,) if capture is None else (primary, f"x{capture}")


def _board_for_actions(actions: tuple[str, ...]) -> tuple[BoardState, int]:
    turns: list[tuple[str, ...]] = []
    for action in actions:
        if action.startswith("x"):
            assert turns and len(turns[-1]) == 1
            turns[-1] = (turns[-1][0], action)
        else:
            turns.append((action,))
    board = BoardState.new_game()
    for turn in turns:
        matching = [
            move
            for move in get_all_legal_moves(board)
            if _turn_actions(move) == turn
        ]
        assert len(matching) == 1
        board = board.apply_move(matching[0])
    return board, len(turns)


class _ProjectHistorySession:
    def history_summary(
        self,
        actions: tuple[str, ...],
        *,
        request_id: str,
        count_mode: str,
    ) -> DataQueryResponse:
        assert count_mode == "logical"
        board, logical_ply = _board_for_actions(tuple(actions))
        state = DataQueryState(
            current_fen=board.to_fen_string(),
            side_to_move="white" if logical_ply % 2 == 0 else "black",
            phase="placing",
            pending_removal=False,
            pending_removals=(0, 0),
            no_capture_plies=logical_ply,
            action_token_count=len(actions),
            logical_ply_count=logical_ply,
            logical_plies_by_side=(
                (logical_ply + 1) // 2,
                logical_ply // 2,
            ),
            snapshot_history_len=len(actions),
            repetition_history_len=logical_ply + 1,
            history_sha256=canonical_sha256(list(actions)),
            outcome=DataQueryOutcome(
                kind="ongoing",
                winner=None,
                reason="ongoing",
            ),
        )
        return DataQueryResponse(
            protocol_version=1,
            request_id=request_id,
            operation="history_summary",
            status="available",
            state=state,
            source=None,
            candidates=None,
            result={},
            raw_line="{}",
        )


def _empty_overlap() -> ExpertSourceOverlapIndex:
    return ExpertSourceOverlapIndex(
        evidence={
            "book_audit": {"audit_identity": _digest("book")},
            "human_audit": {"audit_identity": _digest("human")},
            "human_history_ledger": {"sha256": _digest("ledger")},
            "perfect_audit": {"audit_identity": _digest("perfect")},
        },
        sanmill_book_exact=frozenset(),
        sanmill_book_fen=frozenset(),
        sanmill_book_orbit=frozenset(),
        human_exact=frozenset(),
        human_fen=frozenset(),
        human_orbit=frozenset(),
        perfect_exact=frozenset(),
        perfect_fen=frozenset(),
        perfect_orbit=frozenset(),
        human_exact_support={},
    )


def test_source_preserves_explicit_and_visual_interpretations() -> None:
    source = load_expert_book_source(SOURCE)
    entries = source.payload["entries"]

    assert len(entries) == 35
    assert len(entries[0]["variations"]) == 2
    assert {
        variation["author_tokens"][-1]
        for variation in entries[0]["variations"]
    } == {"c5", "d1"}
    assert entries[10]["variations"][0]["evidence_basis"] == (
        "typed_text_plus_embedded_move_list"
    )
    assert entries[10]["variations"][0]["author_tokens"][-1] == "c5"


def test_project_rules_resolve_all_source_variations_without_fallback() -> None:
    source = load_expert_book_source(SOURCE)
    candidates = prepare_expert_book_candidates(source)

    assert len(candidates) == 36
    assert len({item.exact_history_sha256 for item in candidates}) == 34
    assert len({item.final_nmm_fen for item in candidates}) == 33
    assert len({item.final_ring16_fen for item in candidates}) == 32
    assert len({item.parent8_exact_history_sha256 for item in candidates}) == 15
    assert len({item.parent8_ring16_fen for item in candidates}) == 14


def test_source_identity_fails_closed_after_token_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed["entries"][0]["variations"][0]["author_tokens"][0] = "d2"
    changed["transcription_identity"] = canonical_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "transcription_identity"
        }
    )
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    monkeypatch.setattr(
        "learned_ai.evaluation.layered_expert_book_audit._repo_relative",
        lambda _path: "changed.json",
    )
    source = load_expert_book_source(path)

    with pytest.raises(
        LayeredExpertBookAuditError,
        match="illegal or capture-ambiguous",
    ):
        prepare_expert_book_candidates(source)


def test_audit_replays_all_records_and_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "learned_ai.evaluation.layered_opening_prefix."
        "project_stable_sanmill_fen",
        lambda fen: BoardState.from_fen_string(fen),
    )
    monkeypatch.setattr(
        "learned_ai.evaluation.layered_expert_book_audit."
        "project_stable_sanmill_fen",
        lambda fen: BoardState.from_fen_string(fen),
    )
    source = load_expert_book_source(SOURCE)
    candidates = prepare_expert_book_candidates(source)
    audit = build_layered_expert_book_audit(
        _ProjectHistorySession(),  # type: ignore[arg-type]
        _installation(),
        source=source,
        candidates=candidates,
        overlap=_empty_overlap(),
        generator_commit="5" * 40,
    )

    summary = verify_layered_expert_book_audit(audit)

    assert summary == {
        "source_rows": 35,
        "source_variations": 36,
        "legal_records": 36,
        "unique_histories": 34,
        "unique_final_fens": 33,
        "unique_ring16_orbits": 32,
        "parent8_ring16_orbits": 14,
        "human_exact_matches": 0,
    }
    assert audit["decision"]["final_corpus_frozen"] is False
    assert audit["decision"]["row_11_visual_completion_confirmed"] is True

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from game.board import BoardState
from learned_ai.evaluation.layered_opening_prefix import (
    LAYERED_PREFIX_SCHEMA,
    PREFIX_LOGICAL_PLIES_BY_SIDE_V2,
    PREFIX_LOGICAL_PLIES_V2,
    LayeredOpeningPrefixV2,
    LayeredPrefixError,
    build_layered_prefix_v2,
)
from learned_ai.evaluation.sanmill_data_query import (
    DataQueryOutcome,
    DataQueryResponse,
    DataQueryState,
)
from learned_ai.evaluation.sanmill_uci import SanmillInstallation


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


def _state(
    logical_ply: int,
    action_count: int,
    *,
    pending: bool = False,
) -> DataQueryState:
    return DataQueryState(
        current_fen=f"sanmill-fen-{logical_ply}",
        side_to_move="white" if logical_ply % 2 == 0 else "black",
        phase="placing",
        pending_removal=pending,
        pending_removals=(1, 0) if pending else (0, 0),
        no_capture_plies=logical_ply,
        action_token_count=action_count,
        logical_ply_count=logical_ply,
        logical_plies_by_side=(
            (logical_ply + 1) // 2,
            logical_ply // 2,
        ),
        snapshot_history_len=action_count,
        repetition_history_len=logical_ply + 1,
        history_sha256=_digest(f"state-{logical_ply}-{action_count}"),
        outcome=DataQueryOutcome(
            kind="ongoing",
            winner=None,
            reason="ongoing",
        ),
    )


class _HistorySession:
    def __init__(self, *, pending_at: int | None = None) -> None:
        self.pending_at = pending_at
        self.calls: list[tuple[tuple[str, ...], str, str]] = []

    def history_summary(
        self,
        actions: tuple[str, ...],
        *,
        request_id: str,
        count_mode: str,
    ) -> DataQueryResponse:
        self.calls.append((tuple(actions), request_id, count_mode))
        logical_ply = sum(not action.startswith("x") for action in actions)
        state = _state(
            logical_ply,
            len(actions),
            pending=logical_ply == self.pending_at,
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


def _turns() -> tuple[tuple[str, ...], ...]:
    return (
        ("a7",),
        ("d7",),
        ("a4",),
        ("d6",),
        ("a1", "xd7"),
        ("d7",),
        ("b6",),
        ("f6",),
        ("b4",),
        ("f4",),
        ("b2", "xf6"),
        ("f6",),
    )


def _build(monkeypatch: pytest.MonkeyPatch) -> LayeredOpeningPrefixV2:
    monkeypatch.setattr(
        "learned_ai.evaluation.layered_opening_prefix."
        "project_stable_sanmill_fen",
        lambda _fen: BoardState.from_fen_string(
            "WWWW.B.....B............|W|6|6"
        ),
    )
    return build_layered_prefix_v2(
        _HistorySession(),  # type: ignore[arg-type]
        _installation(),
        stratum="book",
        source_subtype="named_book_variation",
        source_history_id=_digest("source-history"),
        source_identity={
            "kind": "book",
            "identity_sha256": _digest("book"),
            "identity": {"path_lookup_key": "sanmill_checkout"},
        },
        source_evidence={"variation_id": "test-opening"},
        logical_turns=_turns(),
        step_evidence=[
            {"source_token_index": index} for index in range(12)
        ],
    )


def test_v2_record_replays_twelve_logical_turns_and_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _build(monkeypatch)
    payload = record.to_dict()

    assert payload["schema_version"] == LAYERED_PREFIX_SCHEMA
    assert payload["logical_ply_count"] == PREFIX_LOGICAL_PLIES_V2
    assert tuple(payload["logical_plies_by_side"]) == (
        PREFIX_LOGICAL_PLIES_BY_SIDE_V2
    )
    assert len(payload["steps"]) == 12
    assert payload["steps"][4]["action_tokens"] == ["a1", "xd7"]
    assert payload["steps"][4]["output"]["logical_ply_count"] == 5
    assert payload["steps"][-1]["output"]["logical_plies_by_side"] == [6, 6]
    assert payload["final"]["nmm_fen"] == "WWWW.B.....B............|W|6|6"
    assert LayeredOpeningPrefixV2.from_dict(payload).to_dict() == payload


def test_v2_identity_covers_target_length_and_step_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _build(monkeypatch).to_dict()
    changed = copy.deepcopy(payload)
    changed["steps"][3]["source_evidence"]["source_token_index"] = 99

    with pytest.raises(LayeredPrefixError, match="identity mismatch"):
        LayeredOpeningPrefixV2.from_dict(changed)

    changed = copy.deepcopy(payload)
    changed["logical_ply_count"] = 8
    changed["logical_plies_by_side"] = [4, 4]
    with pytest.raises(LayeredPrefixError, match="target length"):
        LayeredOpeningPrefixV2.from_dict(changed)


def test_v2_record_rejects_unknown_fields_and_absolute_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _build(monkeypatch).to_dict()
    payload["unexpected"] = True
    with pytest.raises(LayeredPrefixError, match="unknown unexpected"):
        LayeredOpeningPrefixV2.from_dict(payload)

    with pytest.raises(LayeredPrefixError, match="absolute path"):
        build_layered_prefix_v2(
            _HistorySession(),  # type: ignore[arg-type]
            _installation(),
            stratum="human_db",
            source_subtype="observed_playok_history",
            source_history_id=_digest("human-history"),
            source_identity={
                "kind": "human_db",
                "identity_sha256": _digest("human-db"),
                "database_path": "I:/private/human.sqlite",
            },
            source_evidence={"game_count": 2},
            logical_turns=_turns(),
            step_evidence=[{"observed": True}] * 12,
        )


def test_v2_record_rejects_partial_turn_and_pending_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "learned_ai.evaluation.layered_opening_prefix."
        "project_stable_sanmill_fen",
        lambda _fen: BoardState.new_game(),
    )
    invalid = list(_turns())
    invalid[2] = ("a4", "b4")
    with pytest.raises(LayeredPrefixError, match="non-removal second action"):
        build_layered_prefix_v2(
            _HistorySession(),  # type: ignore[arg-type]
            _installation(),
            stratum="book",
            source_subtype="named_book_variation",
            source_history_id=_digest("source-history"),
            source_identity={"kind": "book"},
            source_evidence={},
            logical_turns=invalid,
            step_evidence=[{}] * 12,
        )

    with pytest.raises(LayeredPrefixError, match="pending removal"):
        build_layered_prefix_v2(
            _HistorySession(pending_at=7),  # type: ignore[arg-type]
            _installation(),
            stratum="book",
            source_subtype="named_book_variation",
            source_history_id=_digest("source-history"),
            source_identity={"kind": "book"},
            source_evidence={},
            logical_turns=_turns(),
            step_evidence=[{}] * 12,
        )


def test_v1_prefix_target_remains_eight() -> None:
    from learned_ai.evaluation.sanmill_prefix import PREFIX_LOGICAL_PLIES

    assert PREFIX_LOGICAL_PLIES == 8

from __future__ import annotations

import ast
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.agents.positional_safety import (
    PositionalSafetyError,
    PositionalSafetyFilter,
)
from learned_ai.agents.specialist_router import SpecialistRouter
from learned_ai.sentinel.db_teacher import ExternalSolvedDB


ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "specialist_positional_downgrades_v1.json"


def _move_key(move: dict) -> tuple[object, object, object]:
    return move.get("from"), move.get("to"), move.get("capture")


@dataclass(frozen=True)
class _RecordedValue:
    outcome: str
    preserves: bool = True
    sector: tuple[int, int, int, int] = (0, 0, 0, 0)
    sector_value: int = 0
    perspective: str = "W"


class _RecordedOracle:
    """Replay the real fixture's frozen Malom W/D/L partition."""

    def __init__(self, case: dict) -> None:
        self._root = BoardState.from_fen_string(case["board_fen"])
        self._parent_tier = case["parent_tier"]
        safe = {_move_key(move) for move in case["safe_moves"]}
        self._successors = {
            self._root.apply_move(move).to_fen_string(): _move_key(move) in safe
            for move in get_all_legal_moves(self._root)
        }

    def query_value(self, board: BoardState) -> _RecordedValue | None:
        fen = board.to_fen_string()
        if fen == self._root.to_fen_string():
            return _RecordedValue(self._parent_tier, perspective=board.turn)
        if fen not in self._successors:
            return None
        return _RecordedValue(
            "L",
            preserves=self._successors[fen],
            perspective=board.turn,
        )

    def move_value(
        self,
        parent: _RecordedValue,
        child: _RecordedValue,
    ) -> _RecordedValue:
        outcome = parent.outcome if child.preserves else "L"
        return _RecordedValue(outcome, child.preserves, perspective=parent.perspective)

    def terminal_move_value(
        self,
        parent: _RecordedValue,
        _child_outcome: str,
    ) -> _RecordedValue:
        return _RecordedValue(parent.outcome, perspective=parent.perspective)


class _FailingOracle:
    def query_value(self, _board: BoardState) -> None:
        return None

    def move_value(self, _parent, _child):  # pragma: no cover - must not run
        raise AssertionError("move_value must not run after a failed root query")

    def terminal_move_value(self, _parent, _child):  # pragma: no cover
        raise AssertionError("terminal_move_value must not run")


def _cases() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


def test_fixture_provenance_matches_lightweight_ledger_when_present() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    ledger = (
        ROOT
        / "out"
        / "evaluation"
        / "sanmill-trained-model-lightweight-v1-20260817-001"
        / "candidate-games.jsonl"
    )
    if not ledger.is_file():
        pytest.skip("ignored local lightweight evidence ledger is absent")
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == fixture["source"][
        "candidate_ledger_sha256"
    ]
    expected = {case["source_record_sha256"]: case for case in fixture["cases"]}
    observed: set[str] = set()
    with ledger.open(encoding="utf-8") as handle:
        for line in handle:
            outer = json.loads(line)
            case = expected.get(outer["record_sha256"])
            if case is None:
                continue
            record = outer["record"]
            turn = next(
                item
                for item in record["turns"]
                if item["absolute_logical_ply"] == case["absolute_logical_ply"]
            )
            assert record["arm"] == "active-specialists-free"
            assert turn["move"] == case["unsafe_move"]
            assert turn["candidate_choice"]["self_downgrade_transition"] == "D->L"
            observed.add(outer["record_sha256"])
    assert observed == set(expected)


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["phase"])
def test_real_downgrade_fixture_cannot_bypass_positional_filter(case: dict) -> None:
    board = BoardState.from_fen_string(case["board_fen"])
    legal = get_all_legal_moves(board)
    assert len(legal) == case["legal_move_count"]

    unsafe_key = _move_key(case["unsafe_move"])
    raw_scores = [1.0 if _move_key(move) == unsafe_key else 0.25 for move in legal]
    safety_filter = PositionalSafetyFilter(
        _RecordedOracle(case),
        label_version="sector-corrected-v1",
        manifest_sha256="0" * 64,
    )

    filtered, decision = safety_filter.filter_scores(board, legal, raw_scores)
    selected = legal[max(range(len(filtered)), key=filtered.__getitem__)]
    safe_keys = {_move_key(move) for move in case["safe_moves"]}

    assert _move_key(selected) in safe_keys
    assert filtered[next(i for i, move in enumerate(legal) if _move_key(move) == unsafe_key)] == 0.0
    assert decision.intervened is True
    assert _move_key(decision.original_move) == unsafe_key
    assert _move_key(decision.selected_move) in safe_keys
    assert decision.positional_only is True
    assert decision.history_aware is False


def test_filter_rejects_untrusted_label_version() -> None:
    with pytest.raises(PositionalSafetyError, match="sector-corrected-v1"):
        PositionalSafetyFilter(
            _FailingOracle(),
            label_version="historical-unversioned",
            manifest_sha256="0" * 64,
        )


def test_filter_fails_closed_when_required_query_is_missing() -> None:
    case = _cases()[0]
    board = BoardState.from_fen_string(case["board_fen"])
    legal = get_all_legal_moves(board)
    safety_filter = PositionalSafetyFilter(
        _FailingOracle(),
        label_version="sector-corrected-v1",
        manifest_sha256="0" * 64,
    )

    with pytest.raises(PositionalSafetyError, match="parent Malom value"):
        safety_filter.filter_scores(board, legal, [1.0] * len(legal))


def test_router_reports_startup_disabled_state_without_scoring(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = SpecialistRouter(object(), None, None, {})
    board = BoardState.new_game()

    def _must_not_score(self, *_args, **_kwargs):
        raise AssertionError("an unavailable safety filter must stop before scoring")

    router.score_moves = MethodType(_must_not_score, router)
    with caplog.at_level(logging.ERROR, logger="nmm.specialist_router"):
        result = router.score_moves_positional_safe(
            board,
            get_all_legal_moves(board),
            board.turn,
        )

    assert result is None
    status = router.positional_safety_status()
    assert status["enabled"] is False
    assert status["unavailable_requests"] == 1
    assert "disabled" in caplog.text


def test_missing_external_malom_cannot_expose_complete_oracle() -> None:
    database = ExternalSolvedDB("", enabled=False)
    assert database.is_available() is False
    with pytest.raises(RuntimeError, match="complete Malom oracle is unavailable"):
        database.require_complete_oracle()


def test_router_query_failure_is_observable_and_never_returns_raw_argmax(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = SpecialistRouter(object(), None, None, {})
    router.configure_positional_safety(
        _FailingOracle(),
        label_version="sector-corrected-v1",
        manifest_sha256="0" * 64,
    )
    case = _cases()[0]
    board = BoardState.from_fen_string(case["board_fen"])
    legal = get_all_legal_moves(board)

    def _raw_scores(self, _board, candidates, _color):
        return [1.0] + [0.0] * (len(candidates) - 1)

    router.score_moves = MethodType(_raw_scores, router)
    with caplog.at_level(logging.ERROR, logger="nmm.specialist_router"):
        result = router.score_moves_positional_safe(board, legal, board.turn)

    assert result is None
    status = router.positional_safety_status()
    assert status["runtime_failures"] == 1
    assert "parent Malom value" in status["last_error"]
    assert "failed closed" in caplog.text


def test_real_malom_recomputes_frozen_safe_sets_when_available() -> None:
    from ai.malom_db import MalomDB

    config_path = ROOT / "data" / "training_paths.local.json"
    if not config_path.is_file():
        pytest.skip("machine-local Malom configuration is absent")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    database = MalomDB(config["malom_db_path"])
    if not database.is_available():
        pytest.skip("configured Malom tablebase is unavailable")
    try:
        for case in _cases():
            board = BoardState.from_fen_string(case["board_fen"])
            legal = get_all_legal_moves(board)
            unsafe_key = _move_key(case["unsafe_move"])
            raw_scores = [
                1.0 if _move_key(move) == unsafe_key else 0.25
                for move in legal
            ]
            safety_filter = PositionalSafetyFilter(
                database,
                label_version="sector-corrected-v1",
                manifest_sha256="0" * 64,
            )
            filtered, _decision = safety_filter.filter_scores(
                board,
                legal,
                raw_scores,
            )
            observed_safe = {
                _move_key(move)
                for move, score in zip(legal, filtered, strict=True)
                if score > 0.0
            }
            assert observed_safe == {
                _move_key(move) for move in case["safe_moves"]
            }
    finally:
        database.close()


def test_product_specialist_route_converges_on_final_product_safety_gate() -> None:
    tree = ast.parse((ROOT / "web" / "app.py").read_text(encoding="utf-8"))
    ai_turn = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_ai_turn"
    )
    specialist_attributes = [
        node.attr
        for node in ast.walk(ai_turn)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "_overseer_advisor"
    ]
    assert "score_moves" in specialist_attributes
    choke_calls = [
        node
        for node in ast.walk(ai_turn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_finalize_product_ai_move"
    ]
    assert len(choke_calls) == 1


def test_product_ui_fails_closed_when_safety_status_is_unavailable() -> None:
    template = (ROOT / "web" / "templates" / "index.html").read_text(
        encoding="utf-8"
    )
    script = (ROOT / "web" / "static" / "game.js").read_text(encoding="utf-8")
    assert 'id="chk-overseer-player" disabled' in template
    assert "chkPlayer.disabled = !s.playable" in script
    assert "A_pos unavailable; classic continues visibly unfiltered" in script

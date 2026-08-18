from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.agents.positional_safety import ProductPositionalSafetyGate


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests/fixtures/classical_positional_downgrades_v1.json"


def _move_key(move: dict) -> tuple[object, object, object]:
    return move.get("from"), move.get("to"), move.get("capture")


@dataclass(frozen=True)
class _Value:
    outcome: str
    preserves: bool = True
    sector: tuple[int, int, int, int] = (1, 1, 1, 1)
    sector_value: int = 0
    perspective: str = "W"


class _RecordedOracle:
    def __init__(self, case: dict) -> None:
        self.calls = 0
        self.root = BoardState.from_fen_string(case["board_fen"])
        self.parent_tier = case["parent_tier"]
        safe = {_move_key(move) for move in case["safe_moves"]}
        self.successors = {
            self.root.apply_move(move).to_fen_string(): _move_key(move) in safe
            for move in get_all_legal_moves(self.root)
        }

    def query_value(self, board: BoardState) -> _Value | None:
        self.calls += 1
        fen = board.to_fen_string()
        if fen == self.root.to_fen_string():
            return _Value(self.parent_tier, perspective=board.turn)
        if fen not in self.successors:
            return None
        return _Value("L", preserves=self.successors[fen], perspective=board.turn)

    @staticmethod
    def move_value(parent: _Value, child: _Value) -> _Value:
        return _Value(
            parent.outcome if child.preserves else "L",
            child.preserves,
            perspective=parent.perspective,
        )

    @staticmethod
    def terminal_move_value(parent: _Value, _child: str) -> _Value:
        return _Value(parent.outcome, perspective=parent.perspective)


class _MissingOracle:
    def query_value(self, _board):
        return None

    def move_value(self, _parent, _child):
        raise AssertionError("move_value must not be reached")

    def terminal_move_value(self, _parent, _child):
        raise AssertionError("terminal_move_value must not be reached")


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _gate(oracle) -> ProductPositionalSafetyGate:
    gate = ProductPositionalSafetyGate(high_difficulty_minimum=9)
    gate.configure(
        oracle,
        label_version="sector-corrected-v1",
        manifest_sha256="0" * 64,
        content_sha256="1" * 64,
    )
    return gate


def test_fixture_provenance_matches_classical_ledger_when_present() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ledger = (
        ROOT
        / "out/evaluation/sanmill-classical-search-strength-v2-20260818-001"
        / "classical-games.jsonl"
    )
    if not ledger.is_file():
        pytest.skip("ignored local classical-search evidence ledger is absent")
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == fixture["source"][
        "classical_ledger_sha256"
    ]
    expected = {case["source_record_sha256"]: case for case in fixture["cases"]}
    observed: set[str] = set()
    with ledger.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            case = expected.get(record["record_sha256"])
            if case is None:
                continue
            turn = next(
                row
                for row in record["turns"]
                if row["absolute_logical_ply"] == case["absolute_logical_ply"]
            )
            assert record["arm"].startswith("classical-difficulty-")
            assert turn["move"] == case["unsafe_move"]
            assert turn["candidate_choice"]["self_downgrade_transition"] == case[
                "transition"
            ]
            observed.add(record["record_sha256"])
    assert observed == set(expected)


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["phase"])
def test_real_classical_downgrade_is_replaced_inside_a_pos(case: dict) -> None:
    board = BoardState.from_fen_string(case["board_fen"])
    assert len(get_all_legal_moves(board)) == case["legal_move_count"]
    seen_safe: list[dict] = []

    def _root_research(safe_moves: list[dict]) -> dict:
        seen_safe.extend(safe_moves)
        return safe_moves[-1]

    outcome = _gate(_RecordedOracle(case)).constrain(
        board,
        case["unsafe_move"],
        source="classical-coordinator",
        difficulty=case["difficulty"],
        safe_selector=_root_research,
    )

    safe = {_move_key(move) for move in case["safe_moves"]}
    assert {_move_key(move) for move in seen_safe} == safe
    assert _move_key(outcome.move) in safe
    assert outcome.decision["intervened"] is True
    assert outcome.decision["selection_rule"] == "restricted-root-research"
    assert outcome.decision["positional_only"] is True
    assert outcome.decision["history_aware"] is False


def test_low_difficulty_classical_move_is_not_queried_or_filtered() -> None:
    case = _cases()[0]
    oracle = _RecordedOracle(case)
    outcome = _gate(oracle).constrain(
        BoardState.from_fen_string(case["board_fen"]),
        case["unsafe_move"],
        source="classical-coordinator",
        difficulty=8,
    )
    assert outcome.move == case["unsafe_move"]
    assert outcome.decision["status"] == "bypassed-low-difficulty"
    assert oracle.calls == 0


def test_runtime_query_failure_is_visible_and_keeps_classical_move() -> None:
    case = _cases()[0]
    board = BoardState.from_fen_string(case["board_fen"])
    gate = _gate(_MissingOracle())
    outcome = gate.constrain(
        board,
        case["unsafe_move"],
        source="classical-coordinator",
        difficulty=9,
    )
    assert outcome.move == case["unsafe_move"]
    assert outcome.decision["status"] == "unfiltered-query-failure"
    status = gate.status()
    assert status["enabled"] is True
    assert status["runtime_failures"] == 1
    assert "parent Malom value" in status["last_error"]


def test_research_failure_uses_deterministic_safe_move_not_unsafe() -> None:
    case = _cases()[0]
    board = BoardState.from_fen_string(case["board_fen"])

    def _fail(_safe_moves: list[dict]) -> dict:
        raise RuntimeError("root research failed")

    gate = _gate(_RecordedOracle(case))
    outcome = gate.constrain(
        board,
        case["unsafe_move"],
        source="classical-coordinator",
        difficulty=9,
        safe_selector=_fail,
    )
    safe = {_move_key(move) for move in case["safe_moves"]}
    assert _move_key(outcome.move) in safe
    assert outcome.decision["selection_rule"] == "canonical-safe-fallback"
    assert gate.status()["selection_failures"] == 1


def test_generalist_is_filtered_even_below_default_high_difficulty() -> None:
    case = _cases()[0]
    board = BoardState.from_fen_string(case["board_fen"])
    outcome = _gate(_RecordedOracle(case)).constrain(
        board,
        case["unsafe_move"],
        source="generalist",
        difficulty=3,
    )
    assert _move_key(outcome.move) in {
        _move_key(move) for move in case["safe_moves"]
    }
    assert outcome.decision["status"] == "applied"


def test_live_malom_recomputes_classical_fixture_a_pos_when_available() -> None:
    from ai.game_ai import GameAI
    from ai.malom_db import MalomDB

    config_path = ROOT / "data/training_paths.local.json"
    if not config_path.is_file():
        pytest.skip("machine-local Malom configuration is absent")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    database = MalomDB(config["malom_db_path"])
    if not database.is_available():
        pytest.skip("configured Malom tablebase is unavailable")
    try:
        cold_elapsed: list[float] = []
        for case in _cases():
            board = BoardState.from_fen_string(case["board_fen"])
            game_ai = GameAI(color=board.turn, difficulty=case["difficulty"])

            def _restricted(safe_moves: list[dict]) -> dict:
                ranked = game_ai.score_root_moves(
                    board,
                    depth=2,
                    time_budget=None,
                    candidate_moves=safe_moves,
                )
                return ranked[0][0]

            outcome = _gate(database).constrain(
                board,
                case["unsafe_move"],
                source="classical-coordinator",
                difficulty=case["difficulty"],
                safe_selector=_restricted,
            )
            assert _move_key(outcome.move) in {
                _move_key(move) for move in case["safe_moves"]
            }
            assert outcome.decision["original_tier"] != case["parent_tier"]
            assert outcome.decision["selection_rule"] == "restricted-root-research"
            cold_elapsed.append(outcome.decision["total_elapsed_ms"])
        assert max(cold_elapsed) < 5000.0

        warm_elapsed: list[float] = []
        for case in _cases():
            board = BoardState.from_fen_string(case["board_fen"])
            outcome = _gate(database).constrain(
                board,
                case["unsafe_move"],
                source="classical-coordinator",
                difficulty=case["difficulty"],
            )
            warm_elapsed.append(outcome.decision["total_elapsed_ms"])
        assert max(warm_elapsed) < 1000.0
    finally:
        database.close()


def test_game_ai_root_scoring_accepts_a_restricted_candidate_set() -> None:
    from ai.game_ai import GameAI

    board = BoardState.new_game()
    safe = get_all_legal_moves(board)[:2]
    ranked = GameAI(color=board.turn, difficulty=1).score_root_moves(
        board,
        depth=1,
        time_budget=None,
        candidate_moves=safe,
    )
    assert {_move_key(move) for move, _score in ranked} == {
        _move_key(move) for move in safe
    }


def test_all_product_machine_move_routes_cross_the_final_safety_choke() -> None:
    tree = ast.parse((ROOT / "web/app.py").read_text(encoding="utf-8"))
    for function_name in ("_ai_turn", "_run_ai_vs_ai_loop"):
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name
        )
        choke_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_finalize_product_ai_move"
        ]
        apply_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply_move"
        ]
        assert len(choke_calls) == 1
        assert apply_calls
        assert choke_calls[0].lineno < min(node.lineno for node in apply_calls)


def test_status_endpoints_expose_resolution_and_final_gate() -> None:
    source = (ROOT / "web/app.py").read_text(encoding="utf-8")
    assert '"malom_runtime": _malom_runtime_status' in source
    assert '"product_positional_safety"' in source
    assert '"candidates"' in source


def test_frozen_main_runtime_reproduces_unfiltered_fixture_when_available() -> None:
    product_root = ROOT / "tmp/classical-search-main-snapshot-4e4a724/tree"
    native_site = ROOT / "tmp/classical-search-main-snapshot-4e4a724/site"
    if not product_root.is_dir() or not native_site.is_dir():
        pytest.skip("frozen main runtime is absent")
    if "nmm_core" in sys.modules:
        pytest.skip("frozen native runtime must be loaded in an isolated process")

    from learned_ai.evaluation.sanmill_classical_search_strength import (
        ProductMainRuntime,
    )

    plan = json.loads(
        (ROOT / "docs/experiments/sanmill-classical-search-strength-v2.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = ProductMainRuntime(
        product_root=product_root,
        native_site=native_site,
        resource_root=ROOT,
        expected=plan["product_contract"],
    )
    try:
        for case in _cases():
            board = BoardState.from_fen_string(case["board_fen"])
            ai = runtime.new_ai(
                color=board.turn,
                difficulty=case["difficulty"],
                node_budget=case["node_budget"],
                search_threads=plan["product_contract"][
                    "deterministic_search_threads"
                ],
                max_depth=plan["product_contract"]["max_depth"],
            )
            assert runtime.choose(ai, board).move == case["unsafe_move"]
    finally:
        runtime.close()

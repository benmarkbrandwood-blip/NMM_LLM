from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation import sanmill_product_route_heldout as heldout
from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.sanmill_classical_search_strength import SearchObservation
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import ResourceLedger
from scripts import recompute_sanmill_product_route_heldout as independent


class _MembershipRow(dict):
    accessed: list[str]

    def __init__(self, **values: object) -> None:
        super().__init__(values)
        self.accessed = []

    def get(self, key: str, default: object = None) -> object:
        self.accessed.append(key)
        if key not in {"start_id", "record_identity"}:
            raise AssertionError(f"pre-freeze accessor touched content field {key}")
        return super().get(key, default)


def test_membership_only_suffix_never_touches_content_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _MembershipRow(
            start_id=f"s-{index}",
            record_identity=f"r-{index}",
            action_history=object(),
            outcome=object(),
        )
        for index in range(5)
    ]
    monkeypatch.setattr(heldout, "POOL_IDENTITY", "pool")
    monkeypatch.setattr(heldout, "POOL_RECORDS_IDENTITY", "records")
    monkeypatch.setattr(heldout, "EXPECTED_POOL_RECORDS", 5)
    monkeypatch.setattr(heldout, "CONSUMED_PREFIX_RECORDS", 3)
    monkeypatch.setattr(heldout, "EXPECTED_STARTS", 2)
    monkeypatch.setattr(
        heldout,
        "CONSUMED_PREFIX_IDENTITY",
        canonical_sha256(["r-0", "r-1", "r-2"]),
    )
    suffix = heldout.membership_only_suffix(
        {"pool_identity": "pool", "records_identity": "records", "records": rows}
    )
    assert suffix == [
        {"start_id": "s-3", "record_identity": "r-3"},
        {"start_id": "s-4", "record_identity": "r-4"},
    ]
    assert all(set(row.accessed) == {"start_id", "record_identity"} for row in rows)


def test_schedule_is_start_major_and_keeps_four_distinct_product_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(heldout, "EXPECTED_STARTS", 2)
    monkeypatch.setattr(heldout, "EXPECTED_GAMES", 16)
    membership = [
        {"start_id": "s0", "record_identity": "r0"},
        {"start_id": "s1", "record_identity": "r1"},
    ]
    schedule = heldout.build_schedule(membership, namespace="test")
    assert [row["ordinal"] for row in schedule] == list(range(16))
    assert [row["arm"] for row in schedule[:4]] == [
        "d9-specialist-first-a-pos",
        "d9-classical-first-a-pos",
        "d9-specialist-first-a-pos",
        "d9-classical-first-a-pos",
    ]
    assert {row["arm"] for row in schedule} == set(heldout.ARMS)
    assert len({row["game_id"] for row in schedule}) == len(schedule)


def _choice(route: str) -> dict:
    return {
        "route": route,
        "product_source": (
            "specialist" if route == "specialist-first" else "classical-coordinator"
        ),
        "classical": {
            "move": {"from": None, "to": "a1", "capture": None},
            "elapsed_seconds": 0.01,
            "nodes": 10,
            "completed_depth": 2,
            "thinking": "test",
            "bypassed_search": False,
        },
        "specialist": {
            "attempted": route == "specialist-first",
            "loaded": True,
            "succeeded": route == "specialist-first",
            "fell_back_to_classical": False,
            "phase_route": "opening" if route == "specialist-first" else None,
            "legal_moves": 24 if route == "specialist-first" else 0,
            "elapsed_seconds": 0.02 if route == "specialist-first" else 0.0,
        },
        "final_gate": {
            "status": "applied",
            "intervened": False,
            "selection_error": None,
            "selection_rule": "original-already-in-A_pos",
        },
        "restricted_root_research": {"called": False},
        "route_elapsed_seconds": 0.03,
    }


def _synthetic_records() -> list[dict]:
    rows = []
    for start_index in range(108):
        start_id = f"s-{start_index:03d}"
        for difficulty in (9, 10):
            for color in ("W", "B"):
                for route in ("specialist-first", "classical-first"):
                    score = 0.5
                    if difficulty == 9 and route == "classical-first":
                        score = 1.0
                    rows.append(
                        {
                            "ordinal": len(rows),
                            "start_id": start_id,
                            "start_phase": "placement",
                            "arm": f"d{difficulty}-{route}-a-pos",
                            "route": route,
                            "difficulty": difficulty,
                            "candidate_color": color,
                            "candidate_score": score,
                            "termination_class": "rules_terminal",
                            "outcome_reason": "drawFiftyMove",
                            "game_elapsed_seconds": 1.0,
                            "turns": [
                                {
                                    "actor": "product",
                                    "phase": "placement",
                                    "product_choice": _choice(route),
                                }
                            ],
                        }
                    )
    return rows


def test_primary_decisions_are_separate_by_difficulty() -> None:
    records = _synthetic_records()
    start_ids = [f"s-{index:03d}" for index in range(108)]
    result = heldout.analyze_records(records, start_ids=start_ids)
    d9 = result["primary"]["difficulty_9_classical_minus_specialist"]
    d10 = result["primary"]["difficulty_10_classical_minus_specialist"]
    assert d9["mean"] == 0.5
    assert d9["decision"] == "classical_first_material_route_candidate"
    assert d10["mean"] == 0.0
    assert d10["decision"] == "no_classical_first_route_change_supported"


def test_independent_primary_matches_main_primary() -> None:
    records = _synthetic_records()
    start_ids = [f"s-{index:03d}" for index in range(108)]
    main = heldout.analyze_records(records, start_ids=start_ids)["primary"]
    recomputed = independent._primary(records, start_ids)
    assert independent._compare(recomputed, main) == []


def test_append_and_load_game_chain_rejects_wrong_schedule(tmp_path: Path) -> None:
    schedule = [
        {
            "ordinal": 0,
            "game_id": "g0",
            "start_id": "s0",
            "start_record_identity": "r0",
            "arm": "d9-specialist-first-a-pos",
            "route": "specialist-first",
            "difficulty": 9,
            "node_budget": 13_887_000,
            "candidate_color": "W",
        }
    ]
    record = {
        **schedule[0],
        "schema_version": heldout.GAME_SCHEMA,
        "termination_class": "rules_terminal",
    }
    path = tmp_path / "games.jsonl"
    heldout.append_game_record(path, record, previous_record_sha256=None)
    recovered = heldout.load_game_records(path, schedule=schedule)
    assert recovered["record_count"] == 1
    bad = [{**schedule[0], "candidate_color": "B"}]
    with pytest.raises(heldout.ProductRouteHeldoutError, match="candidate_color"):
        heldout.load_game_records(path, schedule=bad)


class _FakeClassical:
    @staticmethod
    def choose(_ai: object, board: BoardState) -> SearchObservation:
        move = get_all_legal_moves(board)[0]
        return SearchObservation(
            move=move,
            elapsed_seconds=0.01,
            nodes=1,
            completed_depth=1,
            thinking="fake",
            bypassed_search=False,
        )


class _FakeSpecialist:
    def is_loaded(self) -> bool:
        return True

    def set_gameai(self, _ai: object) -> None:
        return None

    def _pick_specialist(self, _board: BoardState, _color: str) -> tuple:
        return object(), object(), "opening"

    def score_moves(
        self, _board: BoardState, candidates: list[dict], _color: str
    ) -> list[float]:
        return [float(index == 1) for index in range(len(candidates))]


class _FakeGate:
    def constrain(self, _board: BoardState, original: dict, **kwargs: object) -> object:
        return SimpleNamespace(
            move=dict(original),
            decision={
                "status": "applied",
                "selected_move": dict(original),
                "selection_rule": "original-already-in-A_pos",
                "intervened": False,
                "selection_error": None,
                "source": kwargs["source"],
            },
        )


def test_route_choice_uses_specialist_override_only_for_specialist_first() -> None:
    board = BoardState.new_game()
    runtime = SimpleNamespace(
        classical=_FakeClassical(), specialist=_FakeSpecialist()
    )
    ledger = ResourceLedger(0, 0, 0.0, 10, 10, 10.0)
    specialist_move, specialist = heldout.choose_product_route_move(
        board=board,
        ai=object(),
        route_runtime=runtime,
        gate=_FakeGate(),
        route="specialist-first",
        difficulty=9,
        ledger=ledger,
    )
    classical_move, classical = heldout.choose_product_route_move(
        board=board,
        ai=object(),
        route_runtime=runtime,
        gate=_FakeGate(),
        route="classical-first",
        difficulty=9,
        ledger=ledger,
    )
    legal = get_all_legal_moves(board)
    assert specialist_move == legal[1]
    assert specialist["product_source"] == "specialist"
    assert classical_move == legal[0]
    assert classical["product_source"] == "classical-coordinator"

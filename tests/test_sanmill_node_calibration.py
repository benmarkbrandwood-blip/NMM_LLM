"""Focused tests for the engine-only Sanmill node calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_RULES_IDENTITY_SHA256,
    SanmillBridgeError,
    SanmillInstallation,
    UciLogicalTurnResult,
    UciPositionState,
    UciStrictRefereeIdentity,
)
from learned_ai.validation.sanmill_node_calibration import (
    DEFAULT_PATHS_RELATIVE,
    DEFAULT_PLAN_RELATIVE,
    CalibrationPlan,
    CalibrationRoot,
    SanmillCalibrationError,
    inspect_calibration_fixtures,
    load_calibration_plan,
    load_local_installation,
    nearest_rank,
    publish_calibration_result,
    run_calibration,
    state_calibration_record,
    summarize_samples,
    validate_calibration_output,
)


_ROOT = Path(__file__).resolve().parents[1]


def _state() -> UciPositionState:
    return UciPositionState(
        status="ok",
        ruleset_id="nmm",
        rules_identity_sha256=EXPECTED_RULES_IDENTITY_SHA256,
        rules_options={},
        history_origin="startpos",
        fen=(
            "********/********/******** w p p 0 9 0 9 0 0 "
            "-1 -1 -1 -1 0 0 1 ids:nodes"
        ),
        side_to_move="white",
        phase="placing",
        action="place",
        pending_removal_count=0,
        pending_removals=(0, 0),
        legal_actions=(
            "a7",
            "d7",
            "g7",
            "b6",
            "d6",
            "f6",
            "c5",
            "d5",
            "e5",
            "a4",
            "b4",
            "c4",
            "e4",
            "f4",
            "g4",
            "c3",
            "d3",
            "e3",
            "b2",
            "d2",
            "f2",
            "a1",
            "d1",
            "g1",
        ),
        action_token_count=0,
        logical_ply_count=0,
        logical_plies_by_side=(0, 0),
        no_capture_count=0,
        repetition_current_count=0,
        repetition_history_length=0,
        snapshot_history_length=1,
        history_sha256="a" * 64,
        terminal=False,
        winner=None,
        winner_code=None,
        outcome_reason="ongoing",
        outcome_reason_code="ongoing",
        raw_line="sanmill_state {}",
        strict_referee_identity=UciStrictRefereeIdentity(
            format="SANMILL-STRICT-REFEREE-RULES/1",
            profile="mif-stable-moving-v1",
            repetition_observation="stable-moving-v1",
            origin_counted=True,
            semantic_digest=(
                "sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94"
            ),
        ),
    )


def _result(node_budget: int) -> UciLogicalTurnResult:
    return UciLogicalTurnResult(
        status="ok",
        full_turn_actions=("d6",),
        logical_move_id="place:d6",
        model_action={"from": None, "to": "d6", "capture": None},
        logical_ply_delta=1,
        resulting_fen="result",
        resulting_side_to_move="black",
        terminal=False,
        winner=None,
        winner_code=None,
        outcome_reason="ongoing",
        effective_depth=3,
        completed_depth=3,
        score_kind="cp",
        score=0,
        score_perspective="side_to_move",
        node_budget=node_budget,
        primary_nodes=node_budget,
        removal_nodes=0,
        total_nodes=node_budget,
        search_calls=1,
        elapsed_seconds=0.01,
        raw_line="sanmill_logical_turn {}",
    )


class _FakeSession:
    def __init__(self, state: UciPositionState, **_kwargs: Any) -> None:
        self.state = state
        self.closed = False
        self.new_games = 0

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def configure_strict_referee_profile(self, _profile: str) -> None:
        return None

    def new_game(self) -> None:
        self.new_games += 1

    def position_startpos(self, actions: tuple[str, ...]) -> None:
        assert actions == ()

    def state_json(self) -> UciPositionState:
        return self.state

    def search_logical_turn(self, node_budget: int) -> UciLogicalTurnResult:
        return _result(node_budget)

    def close(self) -> None:
        self.closed = True


def _dummy_installation(tmp_path: Path) -> SanmillInstallation:
    return SanmillInstallation(
        checkout=tmp_path,
        commit="a" * 40,
        checkout_head="a" * 40,
        tree="b" * 40,
        binary=tmp_path / "tgf-cli.exe",
        binary_sha256="c" * 64,
        binary_size=1,
        license_sha256="d" * 64,
        path_lookup_key="sanmill_training_checkout",
        require_exact_head=True,
    )


def _small_plan(tmp_path: Path) -> CalibrationPlan:
    state = _state()
    root = CalibrationRoot(
        root_id="empty",
        stratum="placement",
        purpose="unit test",
        history_actions=(),
        expected_state=state_calibration_record(state),
        source={"test": True},
    )
    return CalibrationPlan(
        path=tmp_path / "plan.json",
        raw_sha256="e" * 64,
        identity="f" * 64,
        experiment_id="test",
        claim_boundary="engine only",
        seed=42,
        budgets=(100,),
        repetitions=2,
        protocol_timeout_seconds=1.0,
        search_timeout_seconds=1.0,
        positions=(root,),
        requested_node_ceiling_total=400,
        process_launch_ceiling=4,
        payload={},
    )


def test_tracked_plan_is_bounded_and_does_not_select_a_ladder() -> None:
    plan = load_calibration_plan(_ROOT / DEFAULT_PLAN_RELATIVE)

    assert plan.budgets == (1_000, 5_000, 25_000, 100_000, 500_000)
    assert plan.repetitions == 9
    assert len(plan.positions) == 8
    assert plan.requested_node_ceiling_total == 90_864_000
    assert plan.process_launch_ceiling == 405
    assert plan.payload["decision_rules"]["auto_select_training_ladder"] is False
    assert all(value is False for value in plan.payload["scope"].values())


def test_plan_identity_fails_closed_after_a_budget_edit(tmp_path: Path) -> None:
    payload = json.loads(
        (_ROOT / DEFAULT_PLAN_RELATIVE).read_text(encoding="utf-8")
    )
    payload["measurement"]["node_budgets"][0] = 999
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SanmillCalibrationError, match="identity mismatch"):
        load_calibration_plan(changed)


def test_tracked_fixtures_replay_under_the_pinned_training_runtime() -> None:
    paths = _ROOT / DEFAULT_PATHS_RELATIVE
    if not paths.is_file():
        pytest.skip("requires the ignored Sanmill training path registry")
    plan = load_calibration_plan(_ROOT / DEFAULT_PLAN_RELATIVE)
    installation = load_local_installation(paths)

    observations = inspect_calibration_fixtures(plan, installation)

    assert [item["root_id"] for item in observations] == [
        root.root_id for root in plan.positions
    ]
    assert observations[-2]["state"]["legal_action_count"] == 39
    assert observations[-1]["state"]["logical_ply_count"] == 56


def test_calibration_runner_is_engine_only_and_bounded(tmp_path: Path) -> None:
    plan = _small_plan(tmp_path)
    state = _state()
    sessions: list[_FakeSession] = []

    def factory(_installation: SanmillInstallation, **kwargs: Any) -> _FakeSession:
        session = _FakeSession(state, **kwargs)
        sessions.append(session)
        return session

    report = run_calibration(
        plan,
        _dummy_installation(tmp_path),
        source={"commit": "a" * 40},
        run_id="unit-calibration",
        invocation=("python", "calibrate"),
        session_factory=factory,
    )

    assert report["status"] == "completed"
    assert report["bounded_work"]["samples"] == 4
    assert report["bounded_work"]["requested_node_ceiling_total"] == 400
    assert report["interpretation"]["auto_selected_training_ladder"] is False
    assert report["interpretation"]["model_or_optimizer_work_measured"] is False
    assert len(sessions) == 4
    assert all(session.closed for session in sessions)
    assert len(report["summary"]) == 2


def test_summary_rejects_semantic_drift_and_uses_nearest_rank() -> None:
    assert nearest_rank([1.0, 2.0, 3.0, 4.0, 5.0], 0.90) == 5.0
    samples = [
        {
            "mode": "cold_process",
            "root_id": "root",
            "node_ceiling": 100,
            "semantic_result_sha256": identity,
            "search_seconds": 0.1,
            "actual_nodes": 100,
            "node_utilization": 1.0,
            "nodes_per_second": 1000.0,
            "compound_turn": False,
        }
        for identity in ("a" * 64, "b" * 64)
    ]

    with pytest.raises(SanmillBridgeError, match="non-deterministic"):
        summarize_samples(samples)


def test_result_publication_never_overwrites_evidence(tmp_path: Path) -> None:
    output = tmp_path / "calibration.json"
    publish_calibration_result(output, {"status": "completed"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "completed"
    }
    with pytest.raises(FileExistsError):
        publish_calibration_result(output, {"status": "replaced"})


def test_launch_output_is_confined_to_ignored_diagnostics(tmp_path: Path) -> None:
    with pytest.raises(SanmillCalibrationError, match="out/diagnostics"):
        validate_calibration_output(tmp_path / "outside.json")

    expected = _ROOT / "out" / "diagnostics" / "unit-calibration-never-written.json"
    assert validate_calibration_output(expected) == expected.resolve(strict=False)

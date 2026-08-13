from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import learned_ai.evaluation.retained_passivity_diagnostic as diagnostic
import scripts.run_retained_passivity_diagnostic as runner
import tools.serve_retained_passivity_diagnostic as web
from game.board import BoardState
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
CORPUS = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.json"
)


def _corpus_records() -> list[dict]:
    payload = json.loads(CORPUS.read_text(encoding="utf-8"))
    return payload["corpus"]["records"]


def _history(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _strict_state(*, logical_ply: int, terminal: bool, history: str) -> dict:
    return {
        "status": "terminal" if terminal else "ongoing",
        "ruleset_id": "nmm-test",
        "rules_identity_sha256": "a" * 64,
        "history_origin": "startpos",
        "fen": f"sanmill-fen-{logical_ply}",
        "side_to_move": None if terminal else ("white" if logical_ply % 2 == 0 else "black"),
        "phase": "moving",
        "action": "select",
        "terminal": terminal,
        "removal_pending": False,
        "pending_removal_count": 0,
        "pending_removals": [0, 0],
        "legal_actions": [] if terminal else ["a1-b2"],
        "action_token_count": logical_ply,
        "logical_ply_count": logical_ply,
        "logical_plies_by_side": [
            (logical_ply + 1) // 2,
            logical_ply // 2,
        ],
        "no_capture_count": logical_ply - 12,
        "repetition_current_count": 1,
        "repetition_history_length": logical_ply,
        "snapshot_history_length": logical_ply + 1,
        "history_sha256": history,
        "outcome": {
            "terminal": terminal,
            "winner": "white" if terminal else None,
            "winner_code": 0 if terminal else None,
            "reason": "loseFewerThanThree" if terminal else "ongoing",
            "reason_code": "lose_fewer_than_three" if terminal else "ongoing",
        },
        "strict_referee_identity": {
            "format": "SANMILL-STRICT-REFEREE-RULES/1",
            "profile": "mif-stable-moving-v1",
            "repetitionObservation": "stable-moving-v1",
            "originCounted": True,
            "semanticDigest": "sha256:test",
        },
    }


def _search(turn: dict) -> dict:
    return {
        "status": "ok",
        "full_turn_actions": list(turn["actions"]),
        "logical_move_id": "logical:test",
        "model_action": dict(turn["move"]),
        "logical_ply_delta": 1,
        "resulting_fen": turn["sanmill_fen_after"],
        "resulting_side_to_move": "B",
        "terminal": turn["terminal"],
        "winner": "white" if turn["terminal"] else None,
        "winner_code": 0 if turn["terminal"] else None,
        "outcome_reason": turn["outcome_reason"],
        "effective_depth": 12,
        "completed_depth": 12,
        "score_kind": "cp",
        "score": 0,
        "score_perspective": "side-to-move",
        "node_budget": diagnostic.SANMILL_NODE_CEILING,
        "primary_nodes": 100,
        "removal_nodes": 0,
        "total_nodes": 100,
        "search_calls": 1,
    }


def _synthetic_record(
    spec: dict,
    ordinal: int,
    previous_hash: str | None,
    *,
    survives: bool,
    length: int,
) -> dict:
    item = spec["schedule"][ordinal]
    candidate_color = item["candidate_color"]
    turns = []
    before = item["expected_prefix_history_sha256"]
    for index in range(1, length + 1):
        mover = "W" if index % 2 else "B"
        actor = "candidate" if mover == candidate_color else "sanmill"
        terminal = index == length
        after = _history(f"{ordinal}:{index}")
        move = {"from": None, "to": "a7" if mover == "W" else "d7", "capture": None}
        turn = {
            "post_prefix_logical_ply": index,
            "absolute_logical_ply": diagnostic.PREFIX_LOGICAL_PLIES + index,
            "mover_color": mover,
            "actor": actor,
            "move": move,
            "actions": [move["to"]],
            "before_history_sha256": before,
            "after_history_sha256": after,
            "local_fen_after": f"local-fen-{ordinal}-{index}",
            "sanmill_fen_after": f"sanmill-fen-{ordinal}-{index}",
            "terminal": terminal,
            "outcome_reason": "loseFewerThanThree" if terminal else "ongoing",
            "candidate_malom_delta": 0.0 if actor == "candidate" else None,
            "search": None,
        }
        if actor == "sanmill":
            turn["search"] = _search(turn)
        turns.append(turn)
        before = after
    assert survives is (length > 108)
    snapshot = None
    if survives:
        snapshot_state = _strict_state(
            logical_ply=120,
            terminal=False,
            history=turns[107]["after_history_sha256"],
        )
        snapshot = {
            "absolute_logical_ply": 120,
            "post_prefix_logical_ply": 108,
            "local_fen": turns[107]["local_fen_after"],
            "history_sha256": turns[107]["after_history_sha256"],
            "strict_referee_state": snapshot_state,
            "malom_theoretical": {
                "history_aware": False,
                "queryable": True,
                "side_to_move": "W",
                "side_to_move_wdl": "D",
                "candidate_color": candidate_color,
                "candidate_perspective_wdl": "D",
            },
        }
    final = _strict_state(logical_ply=12 + length, terminal=True, history=before)
    return {
        "schema_version": diagnostic.GAME_SCHEMA,
        "spec_identity": spec["spec_identity"],
        "ordinal": ordinal,
        "unit_index": item["unit_index"],
        "game_id": item["game_id"],
        "match_key": item["match_key"],
        "candidate_id": item["candidate_id"],
        "candidate_color": candidate_color,
        "source_core_id": item["source_core_id"],
        "stratum": item["stratum"],
        "prefix": {
            "prefix_identity": item["prefix_identity"],
            "expected_history_sha256": item["expected_prefix_history_sha256"],
            "observed_history_sha256": item["expected_prefix_history_sha256"],
            "action_token_count": 12,
            "logical_ply_count": 12,
            "logical_plies_by_side": [6, 6],
            "final_nmm_fen": "prefix-local",
            "final_sanmill_fen": "prefix-sanmill",
        },
        "ongoing_after_total_logical_ply_120": survives,
        "ply_120_snapshot": snapshot,
        "post_prefix_logical_plies": length,
        "total_logical_plies": 12 + length,
        "termination_class": "rules_terminal",
        "outcome_reason": "loseFewerThanThree",
        "winner": "W",
        "candidate_score": 1.0 if candidate_color == "W" else 0.0,
        "final_state": final,
        "turns": turns,
        "candidate_malom": diagnostic._candidate_malom_summary(turns),
        "sanmill_search": diagnostic._search_summary(turns),
        "game_elapsed_seconds": 0.25,
        "cumulative_active_seconds": float(ordinal + 1),
        "complete_diagnostic": True,
        "previous_record_sha256": previous_hash,
    }


@pytest.fixture(scope="module")
def spec() -> dict:
    schedule = diagnostic.build_schedule(_corpus_records())
    body = {
        "schema_version": diagnostic.SPEC_SCHEMA,
        "diagnostic_id": "test-passivity",
        "plan": {"identity": "p" * 64},
        "authorization": {"identity": "a" * 64},
        "implementation": {"commit": "c" * 40},
        "runtime": {"seed": 42},
        "protocol": {
            "max_post_prefix_logical_plies": diagnostic.MAX_POST_PREFIX_LOGICAL_PLIES
        },
        "schedule": schedule,
    }
    return {**body, "spec_identity": canonical_sha256(body)}


def test_schedule_pairs_candidates_adjacent_with_same_start_and_colour(spec) -> None:
    assert len(spec["schedule"]) == 256
    for unit_index in range(128):
        v3, v4 = spec["schedule"][unit_index * 2 : unit_index * 2 + 2]
        assert v3["unit_index"] == v4["unit_index"] == unit_index
        assert v3["match_key"] == v4["match_key"]
        assert v3["candidate_color"] == v4["candidate_color"]
        assert v3["candidate_id"] == diagnostic.EXPECTED_CANDIDATES[0]
        assert v4["candidate_id"] == diagnostic.EXPECTED_CANDIDATES[1]


def test_frozen_plan_binds_process_estimand_resources_and_claim_boundary() -> None:
    plan = runner.load_plan(runner.DEFAULT_PLAN)
    assert plan["plan_identity"] == (
        "035c68f80b94dddb8d139d56c38c86c4fde29fa13de5e19db1f4e1fe484c318e"
    )
    assert plan["implementation"]["commit"] == (
        "361d99a43a9ca549b6f4594d8cb5c26a23d5dd54"
    )
    assert plan["workload"] == {
        "automatic_retry_or_recovery": False,
        "candidate_colors_per_start": 2,
        "candidates_per_unit": 2,
        "games": 256,
        "matched_units": 128,
        "max_active_hours": 2.0,
        "max_sanmill_search_turns": 196608,
        "max_summed_node_ceiling": 98304000000,
        "safe_exact_resume_same_spec": True,
        "unique_starts": 64,
    }
    assert plan["protocol"]["horizon_total_logical_ply"] == 120
    assert plan["protocol"]["safety_cap_disposition"] == (
        "incomplete-invalid-for-eventual-WDL-not-draw"
    )
    assert plan["analysis"]["engineering_interval"][
        "maximum_primary_half_width"
    ] == 0.10
    assert plan["claim_boundary"] == {
        "development_corpus_reused": True,
        "held_out_strength_claim": False,
        "playing_strength_claim": False,
        "promotion_or_publication": False,
        "refresh_causal_claim": False,
        "training_or_update": False,
    }
    assert [
        candidate["specialist_db"]["file_sha256"]
        for candidate in plan["candidates"]
    ] == [
        "82d7fbcd897be2493ee40b40a44aa7cd941c95ff538b4f9bf21e2977cd4a8abe",
        "3d69d1acb007dbd26a48ae1c6acec4bb29f905ffedd21c816ad1771a6cf942ed",
    ]
    assert [
        candidate["specialist_db"]["path"]
        for candidate in plan["candidates"]
    ] == [
        "data/specialist_db.sanmill_preserving_retained_v3.seed58.audit_snapshot.sqlite",
        "learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-passivity-diagnostic-v1/v4-specialist-db-snapshot.sqlite",
    ]


def test_horizon_snapshot_retains_strict_history_and_labels_malom_history_free() -> None:
    class Malom:
        @staticmethod
        def query_state(_board):
            return "W"

    board = BoardState.new_game()
    state_record = _strict_state(logical_ply=120, terminal=False, history=_history("120"))
    state = SimpleNamespace(
        logical_ply_count=120,
        terminal=False,
        history_sha256=state_record["history_sha256"],
        portable_record=lambda: state_record,
    )
    snapshot = diagnostic._snapshot_at_horizon(
        board=board,
        state=state,
        candidate_color="B",
        malom=Malom(),
    )
    assert snapshot["strict_referee_state"]["no_capture_count"] == 108
    assert snapshot["malom_theoretical"]["history_aware"] is False
    assert snapshot["malom_theoretical"]["side_to_move_wdl"] == "W"
    assert snapshot["malom_theoretical"]["candidate_perspective_wdl"] == "L"


def test_complete_ledger_recomputes_paired_horizon_and_process_metrics(
    tmp_path,
    spec,
) -> None:
    ledger = tmp_path / "games.jsonl"
    previous = None
    for ordinal in range(256):
        candidate = spec["schedule"][ordinal]["candidate_id"]
        survives = candidate == diagnostic.EXPECTED_CANDIDATES[1]
        record = _synthetic_record(
            spec,
            ordinal,
            previous,
            survives=survives,
            length=110 if survives else 30,
        )
        previous = diagnostic.append_game_record(
            ledger,
            record,
            must_create=ordinal == 0,
        )
    report = diagnostic.recompute_diagnostic(spec, ledger)
    primary = report["paired"]["primary_horizon_survival_v4_minus_v3"]
    assert report["status"] == "completed"
    assert primary["support"] == 128
    assert primary["mean"] == 1.0
    assert primary["interval"] == [1.0, 1.0]
    assert primary["decision"] == "v4_higher_120_ply_survival"
    assert report["by_candidate"][diagnostic.EXPECTED_CANDIDATES[0]][
        "horizon_120"
    ]["survival_rate"] == 0.0
    assert report["by_candidate"][diagnostic.EXPECTED_CANDIDATES[1]][
        "horizon_120"
    ]["survival_rate"] == 1.0
    assert report["claim_boundary"]["playing_strength_claim"] is False


def test_precision_rule_precedes_direction_when_half_width_is_too_wide() -> None:
    records = []
    for index in range(4):
        for candidate_index, candidate in enumerate(diagnostic.EXPECTED_CANDIDATES):
            records.append(
                {
                    "match_key": f"m{index}",
                    "candidate_id": candidate,
                    "ongoing_after_total_logical_ply_120": (
                        candidate_index == 1 and index < 3
                    ),
                    "post_prefix_logical_plies": 120,
                    "candidate_malom": {
                        "preserving_rate_given_queryable": 1.0,
                    },
                }
            )
    paired = diagnostic._paired_comparison(records)
    assert paired["primary_horizon_survival_v4_minus_v3"]["mean"] == 0.75
    assert paired["primary_horizon_survival_v4_minus_v3"]["decision"] == (
        "pending"
    )


def test_ledger_rejects_skipped_horizon_snapshot(tmp_path, spec) -> None:
    record = _synthetic_record(
        spec,
        0,
        None,
        survives=True,
        length=110,
    )
    record["ongoing_after_total_logical_ply_120"] = False
    record["ply_120_snapshot"] = None
    path = tmp_path / "bad.jsonl"
    diagnostic.append_game_record(path, record, must_create=True)
    with pytest.raises(
        diagnostic.RetainedPassivityDiagnosticError,
        match="long game omitted horizon survival",
    ):
        diagnostic.load_game_ledger(spec, path)


def test_malom_rate_reports_query_coverage_and_conditional_denominator() -> None:
    turns = [
        {"actor": "candidate", "candidate_malom_delta": 0.0},
        {"actor": "candidate", "candidate_malom_delta": -1.0},
        {"actor": "candidate", "candidate_malom_delta": None},
        {"actor": "sanmill", "candidate_malom_delta": None},
    ]
    result = diagnostic._candidate_malom_summary(turns)
    assert result["query_coverage"] == pytest.approx(2 / 3)
    assert result["preserving_rate_given_queryable"] == 0.5
    assert result["downgrade_rate_given_queryable"] == 0.5


def test_web_payload_does_not_invent_metrics_before_launch(tmp_path) -> None:
    payload = web.build_payload(tmp_path)
    assert payload == {
        "available": False,
        "status": "not_started",
        "message": "诊断尚未启动；没有伪造或预填结果。",
        "expected_games": 256,
    }
    assert "horizon survival" in web.HTML
    assert "不是和棋" in web.HTML
    assert "history-free" in web.HTML
    assert "query coverage" in web.HTML
    assert "不能归因" in web.HTML


def test_embedded_web_javascript_parses_with_node() -> None:
    script = web.HTML.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    result = subprocess.run(
        ["node", "--check", "-"],
        input=script,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr


def test_web_payload_recomputes_partial_ledger(tmp_path, spec) -> None:
    runner.write_new_canonical(tmp_path / "spec.json", spec)
    record = _synthetic_record(spec, 0, None, survives=False, length=30)
    diagnostic.append_game_record(tmp_path / "games.jsonl", record, must_create=True)
    runner.write_new_canonical(
        tmp_path / "progress.json",
        {
            "completed_games": 1,
            "current_game_ordinal": None,
            "current_stage": None,
            "current_stage_ply": 0,
            "active_seconds": 0.5,
        },
    )
    payload = web.build_payload(tmp_path)
    assert payload["available"] is True
    assert payload["status"] == "running"
    assert payload["report"]["completed_games"] == 1
    assert payload["report"]["paired"]["matched_units_complete"] == 0


def test_authorization_builder_binds_exact_resource_and_prohibitions(
    tmp_path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    plan = {"diagnostic_id": "d", "plan_identity": "p" * 64}
    authorization = runner.build_authorization(
        plan=plan,
        plan_path=plan_path,
        plan_commit="c" * 40,
        authority_text_sha256="a" * 64,
    )
    grant = authorization["grant"]
    assert grant["games"] == 256
    assert grant["max_active_hours"] == 2.0
    assert grant["same_spec_exact_resume"] is True
    assert grant["automatic_retry"] is False
    assert grant["training"] is False
    assert grant["held_out_strength_claim"] is False
    assert grant["promotion"] is False
    assert grant["publication"] is False
    assert grant["release"] is False
    body = {
        key: value
        for key, value in authorization.items()
        if key != "authorization_identity"
    }
    assert authorization["authorization_identity"] == canonical_sha256(body)


def test_launch_readiness_rejects_a_skipped_gate_set() -> None:
    readiness = {
        "ready": True,
        "verdict": "ready_for_long_run",
        "gates": [
            {"gate": gate, "result": "pass"}
            for gate in (
                "repository",
                "plan",
                "authorization",
                "outputs",
                "corpus",
                "candidates",
                "sanmill",
                "process_ownership",
            )
        ],
    }
    with pytest.raises(
        diagnostic.RetainedPassivityDiagnosticError,
        match="skipped or duplicate gates",
    ):
        runner.require_launch_ready(readiness)


def test_cli_launch_cannot_skip_tests_or_prefix_audit() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_retained_passivity_diagnostic.py",
            "--skip-tests",
            "run",
            "--launch",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 2
    assert "cannot skip tests" in result.stderr


def test_cli_never_launches_without_explicit_flag() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_retained_passivity_diagnostic.py", "run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 2
    assert "explicit --launch flag" in result.stderr

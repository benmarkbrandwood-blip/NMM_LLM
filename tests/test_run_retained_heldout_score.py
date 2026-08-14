from __future__ import annotations

from pathlib import Path

import pytest

from learned_ai.evaluation import retained_heldout_score as heldout
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from scripts import run_retained_heldout_score as runner
from scripts import run_retained_phase_process_generalization as shared


def _plan() -> dict:
    body = {
        "schema_version": heldout.PLAN_SCHEMA,
        "diagnostic_id": "heldout-test",
        "status": "frozen_awaiting_product_authorization",
        "candidates": [
            {"candidate_id": candidate_id}
            for candidate_id in heldout.EXPECTED_CANDIDATES
        ],
        "workload": {
            "games": heldout.EXPECTED_GAMES,
            "unique_starts": heldout.EXPECTED_STARTS,
            "max_active_hours": runner.MAX_ACTIVE_HOURS,
        },
        "protocol": {
            "horizon_post_start_logical_plies": (
                heldout.HORIZON_POST_START_LOGICAL_PLIES
            ),
            "max_post_start_logical_plies": heldout.MAX_POST_START_LOGICAL_PLIES,
            "sanmill_node_ceiling_per_turn": heldout.SANMILL_NODE_CEILING,
            "mechanism_reanalysis": "none",
        },
        "analysis": {
            "engineering_interval": {
                "maximum_primary_half_width": heldout.MAX_PRIMARY_HALF_WIDTH
            }
        },
        "claim_boundary": {
            "held_out": True,
            "equivalence_claim": False,
            "refresh_causal_claim": False,
        },
    }
    return {**body, "plan_identity": canonical_sha256(body)}


def test_load_plan_binds_high_precision_fixed_width_contract(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    plan = _plan()
    path.write_bytes(canonical_json_bytes(plan))

    assert runner.load_plan(path) == plan
    plan["analysis"]["engineering_interval"]["maximum_primary_half_width"] = 0.02
    plan_body = {key: value for key, value in plan.items() if key != "plan_identity"}
    plan["plan_identity"] = canonical_sha256(plan_body)
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(runner.RetainedHeldoutScoreError, match="precision"):
        runner.load_plan(path)


def test_shared_runner_profile_is_scoped_and_restored() -> None:
    original_games = shared.EXPECTED_GAMES
    original_plan = shared.DEFAULT_PLAN

    with runner.configured_shared_runner():
        assert shared.EXPECTED_GAMES == 1012
        assert shared.EXPECTED_STARTS == 253
        assert shared.DEFAULT_PLAN == runner.DEFAULT_PLAN
        assert shared.build_schedule is heldout.build_schedule

    assert shared.EXPECTED_GAMES == original_games
    assert shared.DEFAULT_PLAN == original_plan


def test_authorization_builder_binds_workload_and_claim_boundary(
    tmp_path: Path,
) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    authorization = runner.build_authorization(
        plan=plan,
        plan_path=path,
        plan_commit="1" * 40,
        source_readiness_identity="2" * 64,
        authority_text_sha256="3" * 64,
    )

    grant = authorization["grant"]
    assert grant["games"] == 1012
    assert grant["max_active_hours"] == 4.0
    assert grant["held_out_evaluation"] is True
    assert grant["named_route_fixed_corpus_score_relation"] is True
    assert grant["equivalence_claim"] is False
    assert grant["refresh_causal_claim"] is False
    assert grant["training"] is False
    assert grant["updates"] is False
    assert grant["automatic_retry"] is False


def test_cli_cannot_launch_without_explicit_flag(capsys) -> None:
    assert runner.main(["run"]) == 2
    assert "explicit --launch" in capsys.readouterr().err

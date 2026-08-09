from __future__ import annotations

import json
from pathlib import Path

from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-corrected-retained-v2-heldout-eval-v1.json"
)
AUDIT = (
    ROOT
    / "docs"
    / "evidence"
    / "sanmill-corrected-retained-v2-heldout-exposure-2026-08-09.json"
)
AUTHORIZATION = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-corrected-retained-v2-heldout-eval-v1-authorization.json"
)
READINESS_EVIDENCE = (
    ROOT
    / "docs"
    / "evidence"
    / "sanmill-corrected-retained-v2-heldout-runner-readiness-2026-08-09.json"
)
RESULT_EVIDENCE = (
    ROOT
    / "docs"
    / "evidence"
    / "sanmill-corrected-retained-v2-heldout-result-2026-08-09.json"
)


def test_frozen_plan_binds_operational_and_strict_analysis() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan_identity = plan.pop("plan_identity")
    assert canonical_sha256(plan) == plan_identity
    assert plan["workload"] == {
        "games": 128,
        "max_active_hours": 6.0,
        "pairs": 64,
        "safe_exact_resume_same_spec": True,
        "unique_starts": 64,
    }
    assert plan["baseline"]["fixed_node_ceiling_per_logical_turn"] == 500_000
    assert plan["protocol"]["max_post_prefix_logical_plies"] == 1536
    assert plan["protocol"]["max_ply_disposition"] == "incomplete-invalid-not-draw"
    assert plan["analysis"]["strict_independence_is_sensitivity_analysis"] is True
    assert plan["claim_boundary"]["multiple_training_seeds_available"] is False

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit_identity = audit.pop("audit_identity")
    assert canonical_sha256(audit) == audit_identity
    assert audit["summary"]["record_count"] == 64
    assert audit["summary"]["strict_independence_count"] == 34
    assert audit["summary"]["strict_independence_stratum_counts"] == {
        "book": 13,
        "perfect_db": 21,
    }


def test_authorization_binds_one_exact_plan_and_no_promotion() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    authorization_identity = authorization.pop("authorization_identity")
    assert canonical_sha256(authorization) == authorization_identity
    assert authorization["plan"] == {
        "commit": "106d015b23debee7d5c8d691195ff958da66f1fc",
        "file_sha256": (
            "06f168d1687557a9146455fae0a8174c7714b7dd864cfd5a1e2c383c26009b21"
        ),
        "identity": (
            "212076e9423b671b83783efef411db3b4a56c8c67ae36a463d381d6939d4d982"
        ),
        "plan_id": "sanmill-corrected-retained-v2-heldout-eval-v1",
        "tracked_file": (
            "docs/experiments/"
            "sanmill-corrected-retained-v2-heldout-eval-v1.json"
        ),
    }
    assert authorization["consumption"]["grant_count"] == 1
    assert authorization["execution_scope"]["candidate_games_ceiling"] == 128
    assert authorization["execution_scope"][
        "launch_without_further_product_confirmation_after_all_gates_pass"
    ] is True
    assert authorization["claim_boundary"]["new_training"] is False
    assert authorization["claim_boundary"]["model_promotion"] is False
    assert authorization["claim_boundary"]["model_publication"] is False


def test_prepublish_readiness_evidence_preserves_the_unconsumed_gate() -> None:
    evidence = json.loads(READINESS_EVIDENCE.read_text(encoding="utf-8"))
    evidence_identity = evidence.pop("readiness_evidence_identity")

    assert canonical_sha256(evidence) == evidence_identity
    assert evidence["status"] == (
        "implementation_verified_awaiting_publish_and_final_preflight"
    )
    assert evidence["implementation"]["commit"] == (
        "e32d9d46a361d2ed6877b669cdf653eba78e3f3c"
    )
    assert evidence["final_gate"]["ready"] is False
    assert evidence["final_gate"]["failed_gate"] == "repository"
    assert evidence["authorization_state"]["grant_consumed"] is False
    assert evidence["authorization_state"]["corpus_games_played"] == 0
    assert evidence["claim_boundary"]["evaluation_result"] is False


def test_completed_heldout_evidence_binds_result_and_claim_boundary() -> None:
    evidence = json.loads(RESULT_EVIDENCE.read_text(encoding="utf-8"))
    evidence_identity = evidence.pop("evidence_identity")

    assert canonical_sha256(evidence) == evidence_identity
    assert evidence["status"] == "completed_candidate_behind"
    assert evidence["execution"]["completed_games"] == 128
    assert evidence["result"]["primary"] == {
        "decision": "candidate_behind",
        "draws": 102,
        "games": 128,
        "interval": [-0.23146381558966117, -0.08103618441033884],
        "losses": 23,
        "mean_pair_score_difference": -0.15625,
        "score_rate": 0.421875,
        "support_pairs": 64,
        "wins": 3,
    }
    assert evidence["authorization"]["grant_consumed"] is True
    assert evidence["authorization"]["no_second_run"] is True
    assert evidence["host_interruption"]["completed_games_before_interruption"] == 4
    assert evidence["host_interruption"]["interrupted_game_committed"] is False
    assert evidence["verification"]["recompute_equal_to_persisted_report"] is True
    assert evidence["claim_boundary"]["automatic_promotion"] is False
    assert evidence["claim_boundary"]["new_training_authorized"] is False

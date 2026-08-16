from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    EstimatorAccess,
    EstimatorReadinessError,
)
from learned_ai.evaluation.sanmill_human_transfer import (
    AUDIT_SCHEMA,
    PLAN_SCHEMA,
    TransferError,
    _auc,
    _bootstrap,
    audit_coverage,
    load_sealed,
)


ROOT = Path(__file__).resolve().parent.parent
AUDIT = (
    ROOT
    / "docs/evidence/sanmill-human-transfer-coverage-audit-2026-08-16.json"
)
PLAN = ROOT / "docs/experiments/sanmill-human-transfer-v1.json"


def _state(state_id: str, session_id: str, side: str = "W") -> dict:
    return {
        "state_id": state_id,
        "session_id": session_id,
        "logical_ply": 0,
        "phase": "placement",
        "side_to_move": side,
        "a_pos_cardinality": 1,
        "a_pos": [{"move": {"to": "a1"}, "successor_fen": "fen"}],
    }


def _fold_report(fold: int) -> dict:
    return {
        "fold": fold,
        "feature_mean": [0.0] * 10,
        "feature_scale": [1.0] * 10,
        "full_fit": {"coefficients": [0.0] * 10},
        "geometry_fit": {"coefficients": [0.0] * 3},
    }


def test_coverage_audit_requires_actor_out_of_fold_mapping() -> None:
    states = [_state(f"state-{index}", f"game-{index}") for index in range(360)]
    sample = [
        {
            "session_id": f"game-{index}",
            "white": f"w-{index}",
            "black": f"b-{index}",
            "fold": index % 5,
        }
        for index in range(6_400)
    ]
    player_fold = {
        player: row["fold"]
        for row in sample
        for player in (row["white"], row["black"])
    }
    report = audit_coverage(
        pool={"states": states},
        crossfit={
            "structure": {
                "sample_games": sample,
                "player_fold": player_fold,
                "same_fold_games": 6_400,
            }
        },
        readiness={"analysis": {"folds": [_fold_report(i) for i in range(5)]}},
        conversion={"analysis": {"query_accounting": {}}},
        main_result={
            "analysis": {"measurements": [{"state_id": row["state_id"]} for row in states]}
        },
    )
    assert report["coverage"]["available_states"] == 360
    assert report["coverage"]["fraction"] == 1.0

    player_fold["w-0"] = 4
    report = audit_coverage(
        pool={"states": states},
        crossfit={
            "structure": {
                "sample_games": sample,
                "player_fold": player_fold,
                "same_fold_games": 6_400,
            }
        },
        readiness={"analysis": {"folds": [_fold_report(i) for i in range(5)]}},
        conversion={"analysis": {"query_accounting": {}}},
        main_result={
            "analysis": {"measurements": [{"state_id": row["state_id"]} for row in states]}
        },
    )
    assert report["coverage"]["available_states"] == 359
    assert report["coverage"]["failure_reasons"] == {
        "actor_not_held_out_in_session_fold": 1
    }


def test_bootstrap_recomputes_matched_baseline_and_oracle() -> None:
    report = _bootstrap(
        [
            {"selected": 1.0, "b": 0.25, "o": 1.0},
            {"selected": 0.0, "b": 0.0, "o": 0.0},
            {"selected": 1.0, "b": 0.5, "o": 1.0},
        ],
        seed="test",
        repetitions=100,
    )
    assert report["A"]["point"] == pytest.approx(2 / 3)
    assert report["b"] == pytest.approx(0.25)
    assert report["o"] == pytest.approx(2 / 3)
    assert report["transfer"]["point"] == pytest.approx(1.0)


def test_within_state_auc_treats_ties_as_half() -> None:
    assert _auc(np.asarray([0.8]), np.asarray([0.2])) == 1.0
    assert _auc(np.asarray([0.2]), np.asarray([0.8])) == 0.0
    assert _auc(np.asarray([0.5]), np.asarray([0.5])) == 0.5


def test_protected_partition_guard_fails_before_producer() -> None:
    official = json.loads(
        (
            ROOT / "docs/experiments/f0-h0-design-b2-frozen-membership-v1.json"
        ).read_text(encoding="utf-8")
    )
    research = json.loads(
        (
            ROOT / "docs/experiments/human-feature-deviation-train-split-v3.json"
        ).read_text(encoding="utf-8")
    )
    allowed = research["partitions"]["research-exploration"]["session_ids"][:1]
    access = EstimatorAccess.from_memberships(
        official, research, allowed_sessions=allowed
    )
    protected = official["partitions"]["final-test"]["session_ids"][0]
    called = False

    def producer() -> None:
        nonlocal called
        called = True

    with pytest.raises(EstimatorReadinessError):
        access.derive(protected, access_kind="transfer_test", producer=producer)
    assert called is False


def test_frozen_audit_and_plan_are_sealed_when_present() -> None:
    if not AUDIT.exists() or not PLAN.exists():
        pytest.skip("freeze artifacts are generated after implementation tests")
    audit, _audit_sha = load_sealed(
        AUDIT, identity_field="audit_identity", schema=AUDIT_SCHEMA
    )
    plan, _plan_sha = load_sealed(
        PLAN, identity_field="plan_identity", schema=PLAN_SCHEMA
    )
    assert audit["analysis"]["coverage"]["available_states"] == 360
    assert plan["primary_estimator"]["specification"].startswith("full_10")
    assert plan["decision_rule"]["substantive_transfer_threshold"] == 0.25
    assert plan["resource_envelope"]["maximum_sanmill_queries"] == 0


def test_bootstrap_fails_closed_without_oracle_headroom() -> None:
    with pytest.raises(TransferError):
        _bootstrap(
            [{"selected": 0.0, "b": 0.0, "o": 0.0}],
            seed="test",
            repetitions=10,
        )

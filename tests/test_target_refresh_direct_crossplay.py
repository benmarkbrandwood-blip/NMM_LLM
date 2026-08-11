from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from game.board import BoardState
from learned_ai.evaluation.target_refresh_direct_crossplay import (
    LEDGER_SCHEMA,
    PLAN_SCHEMA,
    DirectCrossplayError,
    build_direct_crossplay_schedule,
    summarize_direct_crossplay,
    validate_direct_crossplay_plan,
)
from learned_ai.training.run_contract import canonical_sha256
from scripts.run_target_refresh_direct_crossplay import (
    _sample_policy_move,
    _validate_authorization,
    build_authorization,
)
from scripts import prepare_target_refresh_direct_crossplay as preparer


SHA = "a" * 64
COMMIT = "b" * 40


def _checkpoint_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchors: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for seed in (67, 68, 69):
        anchors.append(
            {
                "seed": seed,
                "path": f"out/source/seed{seed}/fork.pt",
                "file_sha256": SHA,
                "checkpoint_id": f"fork-{seed}",
                "payload_sha256": SHA,
                "model_state_sha256": SHA,
                "game_count": 50,
                "update_count": 18,
                "optimizer_consumed_transition_count": 1152,
                "pending_transition_count": 0,
            }
        )
        for condition in ("refresh-once", "no-refresh"):
            candidates.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "path": f"out/source/seed{seed}/{condition}/8192.pt",
                    "file_sha256": SHA,
                    "checkpoint_id": f"candidate-{seed}-{condition}",
                    "model_state_sha256": SHA,
                    "game_count": 400,
                    "update_count": 146,
                    "optimizer_consumed_transition_count": 9344,
                    "post_fork_consumed_transition_count": 8192,
                    "pending_transition_count": 0,
                    "fork_checkpoint_id": f"fork-{seed}",
                    "immutable_asset_identities": {
                        "human_db": SHA,
                        "malom_tablebase": SHA,
                    },
                }
            )
    return anchors, candidates


def _plan() -> dict[str, Any]:
    anchors, candidates = _checkpoint_records()
    body: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "objective": "Measure direct policy consequences without training.",
        "source": {
            "schedule_contract_path": "docs/experiments/source.json",
            "schedule_contract_sha256": SHA,
            "schedule_plan_identity": SHA,
            "source_result_path": "out/source/result.json",
            "source_result_sha256": SHA,
            "source_result_identity": SHA,
            "training_source_commit": COMMIT,
            "analysis_source_commit": COMMIT,
        },
        "implementation": {
            "commit": COMMIT,
            "module_path": "learned_ai/evaluation/module.py",
            "module_sha256": SHA,
            "prepare_path": "scripts/prepare.py",
            "prepare_sha256": SHA,
            "runner_path": "scripts/run.py",
            "runner_sha256": SHA,
        },
        "data_contract": {
            "human_db_identity": SHA,
            "human_db_malom_policy": "masked_historical_labels",
            "malom_manifest_identity": SHA,
            "policy_corpus_path": "docs/experiments/policy.json",
            "policy_corpus_sha256": SHA,
            "replay_corpus_path": "docs/experiments/replay.json",
            "replay_corpus_sha256": SHA,
            "replay_corpus_identity": SHA,
            "replay_audit_path": "docs/evidence/replay-audit.json",
            "replay_audit_sha256": SHA,
            "replay_audit_identity": SHA,
        },
        "checkpoint_contract": {
            "post_fork_consumed_transitions": 8192,
            "anchors": anchors,
            "candidates": candidates,
        },
        "measurement_contract": {
            "device": "cpu",
            "policy_selection": "training-policy-sampling",
            "temperature": 0.2,
            "shared_game50_anchor_features": True,
            "sanmill_role": "strict-portable-referee-only",
            "record_indices": list(range(1, 13)),
            "replicates_per_start": 4,
            "colour_swap": True,
            "common_random_streams_by_colour": True,
            "max_post_start_logical_plies": 120,
            "max_ply_disposition": "development-draw-with-flag",
            "conditions": ["refresh-once", "no-refresh"],
            "seeds": [67, 68, 69],
            "expected_pairs": 144,
            "expected_games": 288,
        },
        "decision_contract": {
            "contrast": "no-refresh minus refresh-once",
            "minimum_aggregate_pair_score_effect": 1 / 12,
            "minimum_per_seed_pair_score_effect": 1 / 12,
            "minimum_supporting_seeds": 2,
            "maximum_opposite_seed_effect": 1 / 24,
            "maximum_truncation_rate": 0.25,
            "result_classes": [
                "material_no_refresh_direct_effect",
                "material_refresh_once_direct_effect",
                "no_material_direct_effect",
                "inconclusive_truncation",
            ],
            "automatic_long_run_selection": False,
        },
        "resource_envelope": {
            "training_games": 0,
            "optimizer_updates": 0,
            "database_writes": 0,
            "checkpoint_writes": 0,
            "no_update_games": 288,
            "maximum_active_wall_hours": 2.0,
        },
        "output_contract": {
            "readiness": "out/direct/readiness.json",
            "authorization": "out/direct/authorization.json",
            "launch": "out/direct/launch.json",
            "ledger": "out/direct/ledger.jsonl",
            "result": "out/direct/result.json",
            "completion": "out/direct/completion.json",
            "failure": "out/direct/failure.json",
        },
        "claim_boundary": "Development evidence only; not held-out strength.",
        "stop_conditions": ["identity drift", "non-finite policy output"],
        "prohibited_operations": [
            "training-or-optimizer-update",
            "database-or-checkpoint-write",
            "automatic-retry-or-extension",
            "held-out-evaluation",
            "model-promotion-or-publication",
            "long-training-launch",
        ],
    }
    plan = {**body, "plan_identity": canonical_sha256(body)}
    return validate_direct_crossplay_plan(plan)


def _rows(plan: dict[str, Any], effects: dict[int, float] | None = None):
    effects = effects or {}
    rows = []
    for scheduled in build_direct_crossplay_schedule(plan):
        pair_effect = effects.get(int(scheduled["pair_index"]), 0.0)
        if pair_effect > 0:
            score = 1.0 if scheduled["game_in_pair"] == 0 else 0.5
        elif pair_effect < 0:
            score = 0.0 if scheduled["game_in_pair"] == 0 else 0.5
        else:
            score = 0.5
        winner = None
        reason = "draw_threefold_repetition"
        if score != 0.5:
            no_refresh_name = (
                "white"
                if scheduled["no_refresh_colour"] == "W"
                else "black"
            )
            winner = (
                no_refresh_name
                if score == 1.0
                else "black" if no_refresh_name == "white" else "white"
            )
            reason = "win_fewer_than_three"
        phase = (
            "placement"
            if scheduled["record_index"] <= 4
            else "movement"
            if scheduled["record_index"] <= 8
            else "flying"
        )
        rows.append(
            {
                "schema_version": LEDGER_SCHEMA,
                "plan_identity": plan["plan_identity"],
                **scheduled,
                "phase": phase,
                "no_refresh_score": score,
                "outcome_class": {0.0: "loss", 0.5: "draw", 1.0: "win"}[
                    score
                ],
                "winner": winner,
                "termination_reason": reason,
                "post_start_logical_plies": 1,
                "start_history_sha256": SHA,
                "end_history_sha256": SHA,
                "moves": [{"from": None, "to": "a1", "capture": None}],
            }
        )
    return rows


def test_plan_and_schedule_are_closed_and_deterministic() -> None:
    plan = _plan()
    first = build_direct_crossplay_schedule(plan)
    second = build_direct_crossplay_schedule(plan)
    assert first == second
    assert len(first) == 288
    assert len({row["pair_identity"] for row in first}) == 144
    assert first[0]["policy_seed_white"] == first[1]["policy_seed_white"]
    assert first[0]["policy_seed_black"] == first[1]["policy_seed_black"]

    tampered = copy.deepcopy(plan)
    tampered["resource_envelope"]["training_games"] = 1
    with pytest.raises(DirectCrossplayError, match="resource envelope"):
        validate_direct_crossplay_plan(tampered)


def test_no_material_classifier_and_truncation_gate() -> None:
    plan = _plan()
    report = summarize_direct_crossplay(plan, _rows(plan))
    assert report["decision"]["classification"] == "no_material_direct_effect"
    assert report["paired"]["mean_score_effect"] == 0.0

    rows = _rows(plan)
    for row in rows[:73]:
        row["termination_reason"] = "max-ply-truncation"
        row["post_start_logical_plies"] = 120
        row["moves"] = [
            {"from": None, "to": "a1", "capture": None} for _ in range(120)
        ]
    report = summarize_direct_crossplay(plan, rows)
    assert report["decision"]["classification"] == "inconclusive_truncation"


def test_material_no_refresh_effect_requires_seed_support() -> None:
    plan = _plan()
    effects = {
        pair_index: 0.5
        for pair_index in range(144)
        if pair_index // 48 in {0, 1}
    }
    report = summarize_direct_crossplay(plan, _rows(plan, effects))
    assert (
        report["decision"]["classification"]
        == "material_no_refresh_direct_effect"
    )
    assert report["decision"]["supporting_no_refresh_seeds"] == ["67", "68"]


def test_authorization_is_closed_and_bound_to_readiness(tmp_path) -> None:
    plan = _plan()
    authorization = build_authorization(
        plan=plan,
        readiness_identity=SHA,
        authorized_at_utc="2026-08-11T00:00:00Z",
        decision_note="Authorize the frozen no-update diagnostic once.",
    )
    path = tmp_path / "authorization.json"
    path.write_text(__import__("json").dumps(authorization), encoding="utf-8")
    _, identity = _validate_authorization(
        path,
        plan=plan,
        readiness_identity=SHA,
    )
    assert identity == authorization["authorization_identity"]

    authorization["permitted_operations"].append("train")
    authorization.pop("authorization_identity")
    authorization["authorization_identity"] = canonical_sha256(authorization)
    path.write_text(__import__("json").dumps(authorization), encoding="utf-8")
    with pytest.raises(DirectCrossplayError, match="authorization differs"):
        _validate_authorization(path, plan=plan, readiness_identity=SHA)


def test_readiness_probes_the_runtime_immutable_human_db_view(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "human.sqlite"
    observed: list[tuple[Any, bool]] = []

    def fake_probe(value, *, immutable=False):
        observed.append((value, immutable))
        return {"identity": SHA}

    monkeypatch.setattr(preparer, "_probe_human_db", fake_probe)

    assert preparer._probe_direct_crossplay_human_db(path) == {"identity": SHA}
    assert observed == [(path, True)]


class _AscendingPolicy(torch.nn.Module):
    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        return torch.arange(features.shape[0], dtype=torch.float32)


def test_policy_sampling_is_reproducible_for_a_fixed_stream() -> None:
    board = BoardState.new_game()
    first = torch.Generator(device="cpu").manual_seed(1234)
    second = torch.Generator(device="cpu").manual_seed(1234)
    move_a = _sample_policy_move(
        board=board,
        model=_AscendingPolicy(),
        advisor=None,  # type: ignore[arg-type]
        generator=first,
        temperature=0.2,
    )
    move_b = _sample_policy_move(
        board=board,
        model=_AscendingPolicy(),
        advisor=None,  # type: ignore[arg-type]
        generator=second,
        temperature=0.2,
    )
    assert move_a == move_b

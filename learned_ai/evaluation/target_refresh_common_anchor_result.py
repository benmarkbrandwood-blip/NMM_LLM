"""Analyze the fixed-anchor, optimizer-matched target-refresh diagnostic."""

from __future__ import annotations

import hashlib
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import torch

from learned_ai.evaluation.mill_bonus_ablation_result import (
    MillBonusAblationResultError,
    _artifact_record,
    _outcome_label,
    _require_finite,
    _sha256_file,
    _strict_json,
    _strict_jsonl,
    _validate_authorization,
    _validate_finite_tree,
    _validate_policy_health,
    summarize_update_rows,
)
from learned_ai.training import managed_generalist as managed
from learned_ai.training.checkpoint_envelope import load_checkpoint
from learned_ai.training.managed_generalist import (
    load_managed_authorization,
    load_managed_plan,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation.target_refresh_common_anchor_diagnostic import (
    EXPECTED_CONDITIONS,
    EXPECTED_SEEDS,
    PRODUCT_AUTHORIZATION_DECISION,
    READINESS_SCHEMA,
    RESULT_SCHEMA,
    _assert_plan_semantics,
    _ordered_arms,
    _repository_path,
    load_target_refresh_common_anchor_contract,
)


DEFAULT_RESULT = Path("out/target-refresh-common-anchor-diagnostic-v1/result.json")
MINIMUM_MATERIAL_SCORE_EFFECT = 0.10

TargetRefreshCommonAnchorResultError = MillBonusAblationResultError


_REQUIRED_MEASUREMENT_FIELDS = {
    "schema_version",
    "game_id",
    "measurement_index",
    "measurement_batch",
    "training_game_count",
    "optimizer_update_count",
    "anchor_game_count",
    "anchor_update_count",
    "anchor_checkpoint_id",
    "post_anchor_update_count",
    "candidate_checkpoint",
    "candidate_checkpoint_id",
    "opponent_source",
    "learner_color",
    "temperature",
    "specialist_read_mode",
    "no_update",
    "outcome",
    "ply",
    "termination_reason",
    "steps",
    "chosen_probability_mean",
    "entropy_mean",
    "policy_top1_rate",
    "heuristic_top1_rate",
    "malom_preserving_rate",
    "malom_known_steps",
    "specialist_read_stats",
}


def _validate_optimizer_controller_completion(
    plan,
    *,
    expected_updates: int,
) -> tuple[Mapping[str, Any], Path]:
    events = managed.load_run_events(
        Path(plan.control_dir) / managed.CONTROLLER_LEDGER_NAME
    )
    forbidden = {
        "managed_segment_failed",
        "managed_segment_quarantined",
        "managed_segment_interrupted",
        "managed_segment_policy_health_quarantined",
        "managed_resource_limit_reached",
    }
    completed = [
        event for event in events if event.event_type == "managed_segment_completed"
    ]
    if (
        any(event.event_type in forbidden for event in events)
        or len(completed) != 1
        or events[-1].event_type != "managed_plan_completed"
        or completed[0].details.get("segment_index") != 1
        or completed[0].details.get("completed_updates") != expected_updates
    ):
        raise TargetRefreshCommonAnchorResultError(
            "optimizer-bounded controller completion evidence differs"
        )
    completed_games = completed[0].details.get("completed_games")
    if (
        isinstance(completed_games, bool)
        or not isinstance(completed_games, int)
        or not 50 < completed_games <= plan.game_bound
    ):
        raise TargetRefreshCommonAnchorResultError(
            "optimizer-bounded controller game ceiling differs"
        )
    inspected_games, checkpoint = managed._inspect_completed_segment(
        plan,
        segment_index=1,
        previous_completed_games=0,
    )
    envelope = load_checkpoint(checkpoint, map_location="cpu")
    if (
        inspected_games != completed_games
        or envelope.payload.trainer_state.get("update_count") != expected_updates
    ):
        raise TargetRefreshCommonAnchorResultError(
            "optimizer-bounded checkpoint completion differs"
        )
    return completed[0].details, checkpoint


def _wdl(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(_outcome_label(row["outcome"]) for row in rows)
    total = len(rows)
    return {
        "games": total,
        "wins": counts["win"],
        "draws": counts["draw"],
        "losses": counts["loss"],
        "score": (
            None
            if not total
            else (counts["win"] + 0.5 * counts["draw"]) / total
        ),
    }


def _group_wdl(
    rows: Sequence[Mapping[str, Any]],
    key,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {name: _wdl(groups[name]) for name in sorted(groups)}


def _state_dict_sha256(checkpoint: Path) -> str:
    envelope = load_checkpoint(checkpoint, map_location="cpu")
    digest = hashlib.sha256()
    for name in sorted(envelope.payload.model_state):
        value = envelope.payload.model_state[name]
        if not isinstance(value, torch.Tensor):
            raise TargetRefreshCommonAnchorResultError(
                f"model state contains a non-tensor: {name}"
            )
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_training_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    completed_games: int,
) -> dict[str, Any]:
    if len(rows) != completed_games:
        raise TargetRefreshCommonAnchorResultError(
            "training game log count differs from the completed checkpoint"
        )
    required = {
        "game_id",
        "game",
        "learner_color",
        "outcome",
        "ply",
        "temperature",
        "game_type",
        "termination_reason",
        "target_age",
        "chosen_prob_mean",
        "entropy_mean",
        "policy_top1_rate",
        "heuristic_top1_rate",
        "malom_preserving_move_rate",
        "malom_downgrade_move_rate",
        "lr",
    }
    for expected_game, row in enumerate(rows, start=1):
        if required - set(row):
            raise TargetRefreshCommonAnchorResultError(
                f"training row {expected_game} is missing required evidence"
            )
        if row["game"] != expected_game or row["game_id"] in {
            previous["game_id"] for previous in rows[: expected_game - 1]
        }:
            raise TargetRefreshCommonAnchorResultError(
                "training game sequence or identity differs"
            )
        for field in (
            "outcome",
            "ply",
            "temperature",
            "target_age",
            "chosen_prob_mean",
            "entropy_mean",
            "policy_top1_rate",
            "heuristic_top1_rate",
            "malom_preserving_move_rate",
            "malom_downgrade_move_rate",
            "lr",
        ):
            _require_finite(row[field], field=f"training[{expected_game}].{field}")
    return {
        "completed_games": completed_games,
        "wdl": _wdl(rows),
        "by_opponent_source": _group_wdl(rows, lambda row: row["game_type"]),
        "by_learner_color": _group_wdl(rows, lambda row: row["learner_color"]),
        "by_termination_reason": _group_wdl(
            rows, lambda row: row["termination_reason"]
        ),
        "curves": [
            {
                field: row[field]
                for field in (
                    "game",
                    "game_type",
                    "learner_color",
                    "outcome",
                    "ply",
                    "temperature",
                    "target_age",
                    "chosen_prob_mean",
                    "entropy_mean",
                    "policy_top1_rate",
                    "heuristic_top1_rate",
                    "malom_preserving_move_rate",
                    "malom_downgrade_move_rate",
                    "lr",
                )
            }
            for row in rows
        ],
    }


def _validate_measurement_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: Mapping[str, Any],
    contract: Mapping[str, Any],
    segment: Path,
) -> dict[str, Any]:
    measurement = contract["measurement_contract"]
    cadence = int(measurement["measurement_every_updates"])
    final_delta = int(measurement["post_anchor_optimizer_updates"])
    expected_deltas = list(range(cadence, final_delta + 1, cadence))
    per_opponent = int(measurement["games_per_opponent_per_checkpoint"])
    opponents = list(measurement["opponents"])
    expected_rows = len(expected_deltas) * len(opponents) * per_opponent
    if len(rows) != expected_rows:
        raise TargetRefreshCommonAnchorResultError(
            f"measurement row count differs for {arm['arm_id']}"
        )
    anchor_path = segment / "development-measurement-anchor.pt"
    if not anchor_path.is_file():
        raise TargetRefreshCommonAnchorResultError("measurement anchor is absent")
    anchor_envelope = load_checkpoint(anchor_path, map_location="cpu")
    anchor_state = anchor_envelope.payload.trainer_state
    if (
        anchor_state.get("game_count") != measurement["anchor_game"]
        or anchor_state.get("update_count")
        != arm["anchor_expected_update_count"]
        or anchor_envelope.descriptor.role != "development_measurement_anchor"
    ):
        raise TargetRefreshCommonAnchorResultError(
            "measurement anchor checkpoint state differs"
        )
    seen_game_ids: set[str] = set()
    candidate_records: dict[int, dict[str, Any]] = {}
    for expected_index, row in enumerate(rows):
        missing = _REQUIRED_MEASUREMENT_FIELDS - set(row)
        if missing:
            raise TargetRefreshCommonAnchorResultError(
                f"measurement row is missing fields: {sorted(missing)}"
            )
        delta = row["post_anchor_update_count"]
        if (
            row["schema_version"]
            != "nmm.development-anchor-measurement.v1"
            or row["measurement_index"] != expected_index
            or row["game_id"] in seen_game_ids
            or delta not in expected_deltas
            or row["measurement_batch"] != expected_deltas.index(delta) + 1
            or row["optimizer_update_count"]
            != arm["anchor_expected_update_count"] + delta
            or row["anchor_game_count"] != measurement["anchor_game"]
            or row["anchor_update_count"] != arm["anchor_expected_update_count"]
            or row["anchor_checkpoint_id"]
            != anchor_envelope.descriptor.checkpoint_id
            or row["opponent_source"] not in opponents
            or row["learner_color"] not in {"W", "B"}
            or row["temperature"] != measurement["measurement_temperature"]
            or row["specialist_read_mode"] != "disabled"
            or row["specialist_read_stats"] != {}
            or row["no_update"] is not True
        ):
            raise TargetRefreshCommonAnchorResultError(
                f"measurement row semantics differ: {arm['arm_id']}:{expected_index}"
            )
        expected_node_budget = (
            measurement["sanmill_node_budget"]
            if row["opponent_source"] == "sanmill_fixed_node"
            else None
        )
        if row.get("sanmill_node_budget") != expected_node_budget:
            raise TargetRefreshCommonAnchorResultError(
                "measurement Sanmill node identity differs"
            )
        for field in (
            "outcome",
            "ply",
            "steps",
            "chosen_probability_mean",
            "entropy_mean",
            "policy_top1_rate",
            "heuristic_top1_rate",
            "malom_preserving_rate",
            "malom_known_steps",
        ):
            _require_finite(row[field], field=f"measurement[{expected_index}].{field}")
        seen_game_ids.add(str(row["game_id"]))
        checkpoint = Path(str(row["candidate_checkpoint"]))
        if not checkpoint.is_file() or checkpoint.parent != segment:
            raise TargetRefreshCommonAnchorResultError(
                "measurement candidate checkpoint path differs"
            )
        record = candidate_records.get(delta)
        current = {
            "path": str(checkpoint),
            "checkpoint_id": row["candidate_checkpoint_id"],
        }
        if record is None:
            envelope = load_checkpoint(checkpoint, map_location="cpu")
            if (
                envelope.descriptor.checkpoint_id
                != row["candidate_checkpoint_id"]
                or envelope.descriptor.role != "development_measurement_candidate"
                or envelope.payload.trainer_state.get("update_count")
                != arm["anchor_expected_update_count"] + delta
            ):
                raise TargetRefreshCommonAnchorResultError(
                    "measurement candidate checkpoint state differs"
                )
            current.update(
                {
                    "sha256": _sha256_file(checkpoint),
                    "model_state_sha256": _state_dict_sha256(checkpoint),
                }
            )
            candidate_records[delta] = current
        elif (
            record["path"] != current["path"]
            or record["checkpoint_id"] != current["checkpoint_id"]
        ):
            raise TargetRefreshCommonAnchorResultError(
                "measurement batch refers to multiple candidate checkpoints"
            )
    for delta in expected_deltas:
        batch = [row for row in rows if row["post_anchor_update_count"] == delta]
        for opponent in opponents:
            stratum = [row for row in batch if row["opponent_source"] == opponent]
            colors = Counter(row["learner_color"] for row in stratum)
            if len(stratum) != per_opponent or colors != {"W": 4, "B": 4}:
                raise TargetRefreshCommonAnchorResultError(
                    "measurement color or opponent balance differs"
                )
    by_checkpoint: dict[str, Any] = {}
    for delta in expected_deltas:
        batch = [row for row in rows if row["post_anchor_update_count"] == delta]
        by_checkpoint[str(delta)] = {
            "all": _wdl(batch),
            "by_opponent_source": _group_wdl(
                batch, lambda row: row["opponent_source"]
            ),
            "by_learner_color": _group_wdl(
                batch, lambda row: row["learner_color"]
            ),
            "by_termination_reason": _group_wdl(
                batch, lambda row: row["termination_reason"]
            ),
            "policy_observations": {
                field: sum(float(row[field]) for row in batch) / len(batch)
                for field in (
                    "chosen_probability_mean",
                    "entropy_mean",
                    "policy_top1_rate",
                    "heuristic_top1_rate",
                    "malom_preserving_rate",
                )
            },
        }
    return {
        "games": len(rows),
        "anchor": {
            "path": str(anchor_path),
            "sha256": _sha256_file(anchor_path),
            "model_state_sha256": _state_dict_sha256(anchor_path),
            "checkpoint_id": anchor_envelope.descriptor.checkpoint_id,
            "game_count": anchor_state["game_count"],
            "update_count": anchor_state["update_count"],
        },
        "candidate_checkpoints": {
            str(delta): candidate_records[delta] for delta in expected_deltas
        },
        "by_checkpoint": by_checkpoint,
        "overall_by_opponent_source": _group_wdl(
            rows, lambda row: row["opponent_source"]
        ),
        "overall_by_learner_color": _group_wdl(
            rows, lambda row: row["learner_color"]
        ),
        "overall_by_termination_reason": _group_wdl(
            rows, lambda row: row["termination_reason"]
        ),
    }


def _score_at(
    arm: Mapping[str, Any],
    *,
    delta: int,
    opponent: str,
) -> float:
    value = arm["metrics"]["measurement"]["by_checkpoint"][str(delta)][
        "by_opponent_source"
    ][opponent]["score"]
    if value is None:
        raise TargetRefreshCommonAnchorResultError("measurement score is absent")
    return float(value)


def _direction(values: Sequence[float]) -> str:
    if all(value > 0 for value in values):
        return "positive"
    if all(value < 0 for value in values):
        return "negative"
    if all(value == 0 for value in values):
        return "zero"
    return "mixed"


def decide_common_anchor_result(
    arms: Sequence[Mapping[str, Any]],
    *,
    threshold: float = MINIMUM_MATERIAL_SCORE_EFFECT,
) -> dict[str, Any]:
    """Apply the preregistered paired fixed-anchor mechanism rule."""
    index = {(arm["seed"], arm["condition"]): arm for arm in arms}
    if set(index) != {
        (seed, condition)
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    }:
        raise TargetRefreshCommonAnchorResultError("decision arm grid differs")
    deltas = (4, 8, 12, 16)
    seed_records: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        refresh = index[(seed, "refresh")]
        no_refresh = index[(seed, "no-refresh")]
        anchor_contrasts = [
            _score_at(no_refresh, delta=delta, opponent="fixed_model_anchor")
            - _score_at(refresh, delta=delta, opponent="fixed_model_anchor")
            for delta in deltas
        ]
        sanmill_contrasts = [
            _score_at(no_refresh, delta=delta, opponent="sanmill_fixed_node")
            - _score_at(refresh, delta=delta, opponent="sanmill_fixed_node")
            for delta in deltas
        ]
        seed_records.append(
            {
                "seed": seed,
                "fixed_anchor_contrasts": dict(zip(map(str, deltas), anchor_contrasts)),
                "fixed_anchor_mean_contrast": sum(anchor_contrasts) / len(deltas),
                "fixed_anchor_final_contrast": anchor_contrasts[-1],
                "sanmill_contrasts": dict(zip(map(str, deltas), sanmill_contrasts)),
                "sanmill_mean_contrast": sum(sanmill_contrasts) / len(deltas),
                "sanmill_final_contrast": sanmill_contrasts[-1],
            }
        )
    means = [record["fixed_anchor_mean_contrast"] for record in seed_records]
    finals = [record["fixed_anchor_final_contrast"] for record in seed_records]
    mean_direction = _direction(means)
    final_direction = _direction(finals)
    material = abs(median(means)) >= threshold
    supported = (
        mean_direction in {"positive", "negative"}
        and final_direction == mean_direction
        and material
    )
    if supported:
        classification = "target_refresh_mechanism_signal"
    elif mean_direction == "mixed" or final_direction != mean_direction:
        classification = "inconclusive_seed_or_horizon_disagreement"
    else:
        classification = "no_material_common_anchor_signal"
    return {
        "classification": classification,
        "contrast_definition": "no-refresh minus refresh",
        "primary_opponent": "fixed_model_anchor",
        "seed_results": seed_records,
        "mean_contrast_direction": mean_direction,
        "final_contrast_direction": final_direction,
        "median_mean_contrast": median(means),
        "minimum_material_score_effect": threshold,
        "material_threshold_met": material,
        "supported": supported,
        "external_sanmill_role": (
            "separately reported corroboration or counterevidence; not part of "
            "the primary decision gate"
        ),
        "selection": (
            "no retained training setting is selected and no held-out or long "
            "training is authorized"
        ),
    }


def validate_paired_anchor(
    arms: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prove same-seed pre-boundary and anchor-model equality."""
    reports: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        seed_arms = [arm for arm in arms if arm["seed"] == seed]
        if [arm["condition"] for arm in seed_arms] != list(EXPECTED_CONDITIONS):
            raise TargetRefreshCommonAnchorResultError("paired arm order differs")
        prefixes = [
            canonical_json_bytes(
                [row for row in arm["game_rows"] if int(row["game"]) <= 50]
            )
            for arm in seed_arms
        ]
        prefix_equal = prefixes[0] == prefixes[1]
        anchor_hashes = [
            arm["metrics"]["measurement"]["anchor"]["model_state_sha256"]
            for arm in seed_arms
        ]
        anchor_equal = anchor_hashes[0] == anchor_hashes[1]
        if not prefix_equal or not anchor_equal:
            raise TargetRefreshCommonAnchorResultError(
                f"seed {seed} differs before the target-refresh intervention"
            )
        reports.append(
            {
                "seed": seed,
                "first_50_games_byte_identical": True,
                "anchor_model_state_identical": True,
                "anchor_model_state_sha256": anchor_hashes[0],
            }
        )
    return reports


def _validate_readiness(
    path: Path,
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    readiness = _strict_json(path)
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise TargetRefreshCommonAnchorResultError("readiness schema differs")
    body = dict(readiness)
    identity = body.pop("readiness_identity", None)
    if identity != canonical_sha256(body):
        raise TargetRefreshCommonAnchorResultError("readiness identity is invalid")
    record = readiness.get("contract")
    if (
        readiness.get("state") != "ready_for_product_authorization"
        or readiness.get("launch_authorized") is not False
        or not isinstance(record, Mapping)
        or record.get("plan_identity") != contract["plan_identity"]
        or record.get("file_sha256") != _sha256_file(contract_path)
        or len(readiness.get("arms", ())) != 4
    ):
        raise TargetRefreshCommonAnchorResultError("readiness binds another design")
    return readiness


def _readiness_arm(
    readiness: Mapping[str, Any], arm_id: str
) -> Mapping[str, Any]:
    matches = [item for item in readiness["arms"] if item.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise TargetRefreshCommonAnchorResultError(
            f"readiness arm differs: {arm_id}"
        )
    return matches[0]


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    plan,
    arm: Mapping[str, Any],
    contract: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    if (
        manifest.get("schema_version") != "nmm.run-manifest.v1"
        or manifest.get("git_commit") != plan.git_commit
        or manifest.get("git_dirty") is not False
        or manifest.get("experiment_id") != plan.experiment_id
    ):
        raise TargetRefreshCommonAnchorResultError(
            "run manifest source identity differs"
        )
    config = manifest.get("resolved_config")
    expected = {
        "seed": arm["seed"],
        "update_target_every": arm["target_refresh_every_games"],
        "optimizer_update_bound": arm["optimizer_update_bound"],
        "measurement_anchor_game": contract["measurement_contract"][
            "anchor_game"
        ],
        "measurement_anchor_expected_update_count": arm[
            "anchor_expected_update_count"
        ],
        "lr_adaptation_mode": "fixed",
        "specialist_read_mode": "theoretical-only",
        "start_mode": "fresh",
        "referee_engine": "sanmill",
        "opponent_engine": "sanmill",
    }
    if not isinstance(config, Mapping) or any(
        config.get(field) != value for field, value in expected.items()
    ):
        raise TargetRefreshCommonAnchorResultError("run manifest config differs")
    if (
        preflight.get("schema_version") != "nmm.generalist-preflight.v1"
        or preflight.get("verdict") != "needs_decision"
        or preflight.get("errors") != []
        or preflight.get("unresolved_decisions")
        != [PRODUCT_AUTHORIZATION_DECISION]
        or preflight.get("resume_config_sha256") != plan.resume_config_sha256
    ):
        raise TargetRefreshCommonAnchorResultError(
            "readiness preflight content differs"
        )
    policy = manifest.get("checkpoint_policy")
    if not isinstance(policy, Mapping):
        raise TargetRefreshCommonAnchorResultError(
            "run checkpoint policy is absent"
        )
    mif = policy.get("mifSuite")
    ruleset = policy.get("ruleset")
    runtime = contract["rules_and_runtime"]
    if (
        not isinstance(mif, Mapping)
        or not isinstance(ruleset, Mapping)
        or mif.get("tag") != runtime["mif_tag"]
        or mif.get("releaseCommit") != runtime["mif_release_commit"]
        or mif.get("suiteJcsSha256")
        != "sha256:" + runtime["mif_suite_jcs_sha256"]
        or ruleset.get("semanticDigest") != runtime["rules_semantic_digest"]
    ):
        raise TargetRefreshCommonAnchorResultError(
            "run protocol identity differs"
        )


def _analyze_arm(
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    readiness: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
) -> dict[str, Any]:
    control_dir = _repository_path(root, arm["control_dir"], field="control_dir")
    plan_path = control_dir / "plan.json"
    authorization_path = control_dir / "authorization.json"
    plan = load_managed_plan(plan_path)
    _assert_plan_semantics(
        plan,
        root=root,
        contract=contract,
        arm=arm,
        paths_config=paths_config,
        source_commit=source_commit,
    )
    ready_arm = _readiness_arm(readiness, str(arm["arm_id"]))
    if ready_arm.get("plan_sha256") != plan.plan_sha256:
        raise TargetRefreshCommonAnchorResultError("readiness plan hash differs")
    preflight_record = ready_arm.get("preflight")
    if not isinstance(preflight_record, Mapping):
        raise TargetRefreshCommonAnchorResultError("arm preflight is absent")
    preflight_path = Path(str(preflight_record.get("path", "")))
    if (
        not preflight_path.is_file()
        or _sha256_file(preflight_path) != preflight_record.get("sha256")
    ):
        raise TargetRefreshCommonAnchorResultError("arm preflight changed")
    preflight = _strict_json(preflight_path)
    authorization = load_managed_authorization(authorization_path)
    _validate_authorization(authorization, plan)
    completed_details, checkpoint = _validate_optimizer_controller_completion(
        plan,
        expected_updates=int(arm["optimizer_update_bound"]),
    )
    if completed_details.get("completed_updates") != arm["optimizer_update_bound"]:
        raise TargetRefreshCommonAnchorResultError(
            "controller optimizer completion differs"
        )
    health = _validate_policy_health(
        plan,
        details=completed_details,
        checkpoint=checkpoint,
    )
    segment = control_dir / "segments" / "segment-0001"
    manifest_path = segment / "run-manifest.json"
    train_log_path = segment / "train_log.jsonl"
    update_log_path = segment / "update_log.jsonl"
    measurement_log_path = segment / "development_measurement_log.jsonl"
    run_events_path = segment / managed.RUN_EVENT_LEDGER_NAME
    manifest = _strict_json(manifest_path)
    _validate_manifest(
        manifest,
        plan=plan,
        arm=arm,
        contract=contract,
        preflight=preflight,
    )
    run_events = managed.load_run_events(run_events_path)
    if not run_events or run_events[-1].status != "completed":
        raise TargetRefreshCommonAnchorResultError("trainer lifecycle is incomplete")
    game_rows = _strict_jsonl(train_log_path)
    update_rows = _strict_jsonl(update_log_path)
    measurement_rows = _strict_jsonl(measurement_log_path)
    completed_games = int(completed_details["completed_games"])
    if not 50 < completed_games <= 150:
        raise TargetRefreshCommonAnchorResultError("training game ceiling differs")
    if len(update_rows) != arm["optimizer_update_bound"]:
        raise TargetRefreshCommonAnchorResultError("optimizer update log differs")
    anchor_updates = sum(1 for row in update_rows if int(row["game"]) <= 50)
    if anchor_updates != arm["anchor_expected_update_count"]:
        raise TargetRefreshCommonAnchorResultError(
            "pre-boundary optimizer update count differs"
        )
    metrics = {
        "training": _validate_training_rows(
            game_rows,
            completed_games=completed_games,
        ),
        "updates": summarize_update_rows(
            update_rows,
            expected_games=completed_games,
        ),
        "measurement": _validate_measurement_rows(
            measurement_rows,
            arm=arm,
            contract=contract,
            segment=segment,
        ),
        "optimizer_exposure": {
            "anchor_updates": anchor_updates,
            "post_anchor_updates": len(update_rows) - anchor_updates,
            "total_updates": len(update_rows),
            "total_training_steps": sum(int(row["batch_steps"]) for row in update_rows),
            "post_anchor_training_steps": sum(
                int(row["batch_steps"])
                for row in update_rows[anchor_updates:]
            ),
        },
        "supervised_curves": {
            "available": False,
            "reason": (
                "this is an online RL diagnostic; supervised train/validation "
                "curves do not exist"
            ),
        },
    }
    _validate_finite_tree(metrics, field=f"arm[{arm['arm_id']}].metrics")
    specialist_db = _repository_path(
        root, arm["specialist_db"], field="specialist_db"
    )
    return {
        "arm_id": arm["arm_id"],
        "seed": arm["seed"],
        "condition": arm["condition"],
        "target_refresh_every_games": arm["target_refresh_every_games"],
        "source_commit": source_commit,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": _sha256_file(authorization_path),
        "completed_games": completed_games,
        "optimizer_updates": len(update_rows),
        "policy_health": health,
        "metrics": metrics,
        "game_rows": game_rows,
        "artifacts": {
            "plan": _artifact_record(root, plan_path),
            "authorization": _artifact_record(root, authorization_path),
            "preflight": _artifact_record(root, preflight_path),
            "run_manifest": _artifact_record(root, manifest_path),
            "run_events": _artifact_record(root, run_events_path),
            "train_log": _artifact_record(root, train_log_path),
            "update_log": _artifact_record(root, update_log_path),
            "measurement_log": _artifact_record(root, measurement_log_path),
            "checkpoint": _artifact_record(root, checkpoint),
            "specialist_db": _artifact_record(root, specialist_db),
        },
    }


def _inspect_analysis_source(root: Path, expected_commit: str) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise TargetRefreshCommonAnchorResultError("Git audit failed")
        return result.stdout.strip()

    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    origin_dev = git("rev-parse", "origin/dev")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if branch != "dev" or head != origin_dev or head != expected_commit or status:
        raise TargetRefreshCommonAnchorResultError(
            "result analysis requires the clean published training source"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin_dev,
        "training_source_commit": expected_commit,
        "worktree_clean": True,
    }


def analyze_target_refresh_common_anchor_result(
    *,
    root: Path,
    contract_path: Path,
    readiness_path: Path,
    paths_config: Path,
) -> dict[str, Any]:
    """Analyze one authorized four-arm sequence without changing inputs."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    readiness_path = readiness_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    contract = load_target_refresh_common_anchor_contract(contract_path)
    readiness = _validate_readiness(
        readiness_path,
        contract=contract,
        contract_path=contract_path,
    )
    source_commit = str(readiness["source"]["head"])
    source = _inspect_analysis_source(root, source_commit)
    arm_records = [
        _analyze_arm(
            root=root,
            contract=contract,
            arm=arm,
            readiness=readiness,
            paths_config=paths_config,
            source_commit=source_commit,
        )
        for arm in _ordered_arms(contract)
    ]
    paired = validate_paired_anchor(arm_records)
    decision_arms = [
        {key: value for key, value in arm.items() if key != "game_rows"}
        for arm in arm_records
    ]
    decision = decide_common_anchor_result(
        decision_arms,
        threshold=float(
            contract["analysis"]["decision_rule"][
                "minimum_material_score_effect"
            ]
        ),
    )
    body = {
        "schema_version": RESULT_SCHEMA,
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "plan_identity": contract["plan_identity"],
            "sha256": _sha256_file(contract_path),
        },
        "readiness": {
            "path": readiness_path.relative_to(root).as_posix(),
            "readiness_identity": readiness["readiness_identity"],
            "sha256": _sha256_file(readiness_path),
        },
        "analysis_source": source,
        "hyperparameters": contract["common_training_contract"],
        "measurement_contract": contract["measurement_contract"],
        "data_and_runtime_versions": {
            "data_contract": contract["data_contract"],
            "rules_and_runtime": contract["rules_and_runtime"],
        },
        "arms": decision_arms,
        "paired_anchor_validation": paired,
        "decision": decision,
        "claim_boundary": contract["claim_boundary"],
        "interpretation": {
            "observed_facts": (
                "completed logs, exact optimizer-step counts, fixed-anchor and "
                "fixed-node measurements, per-class outcomes, curves, snapshots, "
                "and policy-health results"
            ),
            "hypothesis": contract["hypothesis"],
            "supporting_evidence": (
                "same-seed arms must have byte-identical first-50 ledgers and "
                "identical anchor model tensors before the treatment"
            ),
            "counterevidence": (
                "batch_steps, training-game counts, fixed-node Sanmill results, "
                "and seed/horizon disagreement are reported separately and can "
                "invalidate a causal interpretation"
            ),
            "next_validation_experiment": (
                "only a supported, stable mechanism result may justify a new "
                "held-out design; this report cannot authorize one"
            ),
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}


def publish_result(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise TargetRefreshCommonAnchorResultError(
            f"result already exists: {path}"
        ) from exc


__all__ = [
    "DEFAULT_RESULT",
    "TargetRefreshCommonAnchorResultError",
    "analyze_target_refresh_common_anchor_result",
    "decide_common_anchor_result",
    "publish_result",
    "validate_paired_anchor",
]

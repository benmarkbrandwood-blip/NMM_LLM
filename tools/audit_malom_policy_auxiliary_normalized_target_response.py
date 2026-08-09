"""Audit normalized auxiliary target response without games or saved updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet  # noqa: E402
from learned_ai.training.checkpoint_envelope import load_checkpoint  # noqa: E402
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402
from learned_ai.validation.malom_policy_auxiliary_gradient_interaction import (  # noqa: E402
    audit_malom_policy_auxiliary_normalized_target_response,
)


SCHEMA_VERSION = (
    "nmm.sanmill-malom-policy-auxiliary-normalized-target-response-result.v1"
)
PLAN_SCHEMA = "nmm.sanmill-malom-policy-auxiliary-normalized-target-response-plan.v1"
DEFAULT_PLAN = ROOT / (
    "docs/experiments/"
    "sanmill-malom-policy-auxiliary-normalized-target-response-audit-v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError("audit input must stay inside the repository") from exc


def _strict_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _last_jsonl_row(path: Path) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"JSONL row must be an object at {path}:{line_number}"
                )
            last = value
    if last is None:
        raise RuntimeError(f"JSONL file is empty: {path}")
    return last


def _git_commit(expected: str) -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise RuntimeError("tracked worktree must be clean")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if commit != expected:
        raise RuntimeError(
            f"source commit differs: expected {expected}, observed {commit}"
        )
    return commit


def _checked_file(spec: Mapping[str, Any], *, label: str) -> Path:
    if set(spec) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError(f"{label} identity member set differs")
    path = _resolve(str(spec["path"]))
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    if path.stat().st_size != spec["size_bytes"] or _sha256(path) != spec["sha256"]:
        raise RuntimeError(f"{label} bytes differ")
    return path


def _model(payload: Any, *, device: torch.device) -> ScaffoldedPolicyNet:
    config = payload.trainer_state.get("model_config")
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint model configuration is missing")
    model = ScaffoldedPolicyNet.from_config(config).to(device)
    model.load_state_dict(payload.model_state)
    return model


def _optimizer(model: ScaffoldedPolicyNet, state: Any) -> torch.optim.Adam:
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint optimizer state is missing")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(state)
    return optimizer


def _close(
    actual: Any,
    expected: Any,
    *,
    label: str,
    tolerance: float,
) -> float:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not math.isfinite(actual_value) or not math.isfinite(expected_value):
        raise RuntimeError(f"{label} is non-finite")
    if not math.isclose(
        actual_value,
        expected_value,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise RuntimeError(
            f"{label} differs: replay={actual_value}, logged={expected_value}"
        )
    return actual_value - expected_value


def _audit_arm(
    spec: Mapping[str, Any],
    *,
    result_arm: Mapping[str, Any],
    method: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    required = {
        "arm_id",
        "seed",
        "pre_update_checkpoint",
        "final_checkpoint",
        "update_log",
    }
    if set(spec) != required:
        raise RuntimeError("target-response arm has an invalid member set")
    arm_id = str(spec["arm_id"])
    seed = int(spec["seed"])
    if (
        result_arm.get("arm_id") != arm_id
        or result_arm.get("condition") != "normalized-0.25"
        or result_arm.get("malom_policy_aux_mode") != "policy-head-normalized"
        or int(result_arm.get("seed")) != seed
    ):
        raise RuntimeError(f"calibration result does not bind arm {arm_id}")

    pre_path = _checked_file(
        spec["pre_update_checkpoint"],
        label=f"{arm_id} pre-update checkpoint",
    )
    final_path = _checked_file(
        spec["final_checkpoint"],
        label=f"{arm_id} final checkpoint",
    )
    update_path = _checked_file(
        spec["update_log"],
        label=f"{arm_id} update log",
    )
    pre = load_checkpoint(pre_path, map_location=device)
    final = load_checkpoint(final_path, map_location=device)
    if pre.descriptor.experiment_id != final.descriptor.experiment_id:
        raise RuntimeError(f"{arm_id} checkpoint experiments differ")
    if pre.descriptor.save_reason != "periodic":
        raise RuntimeError(f"{arm_id} pre-update checkpoint is not periodic")
    if final.descriptor.save_reason != "final":
        raise RuntimeError(f"{arm_id} final checkpoint is not final")

    pre_state = pre.payload.trainer_state
    final_state = final.payload.trainer_state
    steps = pre_state["recovery_state"]["pending_steps"]
    if (
        pre_state.get("game_count") != 100
        or final_state.get("game_count") != 100
        or final_state.get("update_count") != pre_state.get("update_count") + 1
        or final_state["recovery_state"]["pending_steps"]
    ):
        raise RuntimeError(f"{arm_id} final-flush checkpoint relationship differs")

    last_update = _last_jsonl_row(update_path)
    targets = tuple(float(value) for value in method["target_policy_head_ratios"])
    coefficient_cap = float(method["coefficient_cap"])
    denominator_floor = float(method["denominator_floor"])
    expected_target = float(method["persisted_replay_target_ratio"])
    if (
        last_update.get("reason") != "final_flush"
        or last_update.get("game") != 100
        or last_update.get("batch_steps") != len(steps)
        or last_update.get("malom_policy_aux_mode") != "policy-head-normalized"
    ):
        raise RuntimeError(f"{arm_id} final update row differs")

    model = _model(pre.payload, device=device)
    optimizer = _optimizer(model, pre.payload.optimizer_state)
    expected_model = _model(final.payload, device=device)
    audit = audit_malom_policy_auxiliary_normalized_target_response(
        model,
        optimizer,
        steps,
        target_policy_head_ratios=targets,
        coefficient_cap=coefficient_cap,
        denominator_floor=denominator_floor,
        device=device,
        gamma=float(method["gamma"]),
        entropy_coef=float(method["entropy_coefficient"]),
        value_coef=float(method["value_coefficient"]),
        grad_clip=float(method["gradient_clip"]),
        expected_treatment_model=expected_model,
        expected_treatment_target_ratio=expected_target,
    )
    if (
        audit.get("original_model_unchanged") is not True
        or audit.get("original_optimizer_unchanged") is not True
    ):
        raise RuntimeError(f"{arm_id} audit mutated its source state")
    indexed = {
        float(response["target_policy_head_ratio"]): response
        for response in audit["responses"]
    }
    replay = indexed[expected_target]
    scalar_tolerance = float(method["logged_scalar_tolerance"])
    replay_losses = replay["audit"]["adam_step"]["treatment_reported_losses"]
    residuals = {
        "policy_loss": _close(
            replay_losses[0],
            last_update.get("policy_loss"),
            label=f"{arm_id} policy loss",
            tolerance=scalar_tolerance,
        ),
        "value_loss": _close(
            replay_losses[1],
            last_update.get("value_loss"),
            label=f"{arm_id} value loss",
            tolerance=scalar_tolerance,
        ),
        "entropy": _close(
            replay_losses[2],
            last_update.get("entropy"),
            label=f"{arm_id} entropy",
            tolerance=scalar_tolerance,
        ),
        "auxiliary_loss": _close(
            replay["audit"]["objectives"]["auxiliary"]["objective_value"],
            last_update.get("malom_policy_aux_loss"),
            label=f"{arm_id} auxiliary loss",
            tolerance=scalar_tolerance,
        ),
        "effective_coefficient": _close(
            replay["effective_coefficient"],
            last_update.get("malom_policy_aux_effective_coef"),
            label=f"{arm_id} effective coefficient",
            tolerance=scalar_tolerance,
        ),
        "realized_policy_head_ratio": _close(
            replay["realized_policy_head_ratio"],
            last_update.get("malom_policy_aux_applied_to_ordinary_policy_head_ratio"),
            label=f"{arm_id} realized policy-head ratio",
            tolerance=scalar_tolerance,
        ),
        "ordinary_policy_head_gradient_l2": _close(
            audit["measurement"]["ordinary_policy_head_gradient_l2"],
            last_update.get("malom_policy_aux_ordinary_policy_head_gradient_l2"),
            label=f"{arm_id} ordinary policy-head gradient",
            tolerance=scalar_tolerance,
        ),
        "raw_auxiliary_gradient_l2": _close(
            audit["measurement"]["raw_auxiliary_gradient_l2"],
            last_update.get("malom_policy_aux_raw_auxiliary_gradient_l2"),
            label=f"{arm_id} raw auxiliary gradient",
            tolerance=scalar_tolerance,
        ),
    }
    replay_difference = replay["audit"]["adam_step"].get(
        "persisted_treatment_replay_difference"
    )
    parameter_tolerance = float(method["production_replay_parameter_tolerance"])
    if (
        not isinstance(replay_difference, dict)
        or not isinstance(replay_difference.get("functionally_relevant"), dict)
        or float(replay_difference["functionally_relevant"].get("max_abs", math.inf))
        > parameter_tolerance
    ):
        raise RuntimeError(f"{arm_id} Adam replay differs from final checkpoint")

    return {
        "arm_id": arm_id,
        "seed": seed,
        "inputs": {
            "pre_update_checkpoint": dict(spec["pre_update_checkpoint"]),
            "final_checkpoint": dict(spec["final_checkpoint"]),
            "update_log": dict(spec["update_log"]),
        },
        "checkpoint_relationship": {
            "game": 100,
            "pre_update_count": pre_state["update_count"],
            "final_update_count": final_state["update_count"],
            "pending_steps": len(steps),
        },
        "persisted_target_replay_residuals": residuals,
        "audit": audit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_commit = _git_commit(args.expected_source_commit)
    plan_path = _resolve(args.plan)
    output_path = _resolve(args.output)
    if output_path.exists():
        raise RuntimeError("target-response output already exists")
    plan = _strict_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise RuntimeError("target-response plan schema differs")
    if plan.get("status") != "frozen_unrun":
        raise RuntimeError("target-response plan status differs")
    if plan.get("claim_boundary") != {
        "candidate_model_loaded": True,
        "new_games": 0,
        "optimizer_or_model_mutation": False,
        "strength_or_promotion_claim": False,
        "training_updates": 0,
    }:
        raise RuntimeError("target-response claim boundary differs")

    implementation = plan.get("implementation")
    if not isinstance(implementation, dict):
        raise RuntimeError("target-response implementation identity is missing")
    if (
        subprocess.check_output(
            ["git", "cat-file", "-t", str(implementation.get("commit"))],
            cwd=ROOT,
            text=True,
        ).strip()
        != "commit"
    ):
        raise RuntimeError("target-response implementation commit is missing")
    for name in ("module", "tool"):
        _checked_file(implementation[name], label=f"target-response {name}")

    result_spec = plan.get("calibration_result")
    if not isinstance(result_spec, dict):
        raise RuntimeError("calibration result identity is missing")
    result_file_spec = {
        key: result_spec[key] for key in ("path", "sha256", "size_bytes")
    }
    result_path = _checked_file(result_file_spec, label="calibration result")
    result = _strict_json(result_path)
    if (
        result.get("result_identity") != result_spec.get("result_identity")
        or result.get("decision", {}).get("verdict") != result_spec.get("verdict")
        or result.get("decision", {}).get("eligible") is not False
    ):
        raise RuntimeError("calibration result decision differs")
    result_arms = {
        arm.get("arm_id"): arm
        for arm in result.get("arms", [])
        if isinstance(arm, dict)
    }

    method = plan.get("method")
    if not isinstance(method, dict):
        raise RuntimeError("target-response method is missing")
    if method.get("target_policy_head_ratios") != [0.25, 0.5, 1.0]:
        raise RuntimeError("target-response target order differs")
    arms = plan.get("arms")
    expected_order = [
        "seed55-normalized-r025",
        "seed56-normalized-r025",
        "seed57-normalized-r025",
    ]
    if (
        not isinstance(arms, list)
        or [arm.get("arm_id") for arm in arms] != expected_order
    ):
        raise RuntimeError("target-response arm order differs")

    input_hashes_before = {
        _portable(_resolve(spec[role]["path"])): _sha256(_resolve(spec[role]["path"]))
        for spec in arms
        for role in ("pre_update_checkpoint", "final_checkpoint", "update_log")
    }
    input_hashes_before[_portable(result_path)] = _sha256(result_path)
    device = torch.device("cpu")
    reports = [
        _audit_arm(
            spec,
            result_arm=result_arms[str(spec["arm_id"])],
            method=method,
            device=device,
        )
        for spec in arms
    ]
    input_hashes_after = {path: _sha256(_resolve(path)) for path in input_hashes_before}
    status_after = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identities": {
            "source_commit": source_commit,
            "implementation_commit": implementation["commit"],
            "plan_path": _portable(plan_path),
            "plan_sha256": _sha256(plan_path),
            "calibration_result_identity": result_spec["result_identity"],
            "calibration_result_sha256": result_spec["sha256"],
        },
        "scope": dict(plan["claim_boundary"]),
        "method": dict(method),
        "arms": reports,
        "mutation_checks": {
            "input_files_unchanged": input_hashes_before == input_hashes_after,
            "tracked_worktree_clean_after": not status_after.strip(),
        },
        "interpretation": {
            "observed_fact": (
                "one disposable Adam update at each target from three persisted "
                "production final-flush batches"
            ),
            "claim_boundary": (
                "optimizer target-response evidence only; no new games, saved "
                "update, target selection, strength claim, or launch authority"
            ),
        },
    }
    if not all(report["mutation_checks"].values()):
        raise RuntimeError("target-response mutation check failed")
    report["audit_identity"] = canonical_sha256(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "state": "audit_published",
                "output": _portable(output_path),
                "audit_identity": report["audit_identity"],
                "sha256": _sha256(output_path),
                "arms": len(reports),
                "targets_per_arm": len(method["target_policy_head_ratios"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

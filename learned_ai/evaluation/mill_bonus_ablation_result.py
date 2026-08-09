"""Fail-closed analysis for the paired Sanmill mill-bonus ablation smoke."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from learned_ai.training import managed_generalist as managed
from learned_ai.training.managed_generalist import (
    ManagedAuthorization,
    ManagedPlan,
    load_managed_authorization,
    load_managed_plan,
)
from learned_ai.training.run_contract import (
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.validation.mill_bonus_ablation_readiness import (
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_REPORT as DEFAULT_READINESS_REPORT,
    PRODUCT_AUTHORIZATION_DECISION,
    READINESS_SCHEMA,
    _assert_plan_semantics,
    _ordered_arms,
    _repository_path,
    load_ablation_contract,
)
from scripts import train_s_gen_v2 as trainer


RESULT_SCHEMA = "nmm.sanmill-mill-bonus-ablation-result.v1"
DEFAULT_RESULT = Path("out/mill-bonus-ablation-smoke-v1/result.json")
ROLLING_WINDOW = 50
TAIL_FIRST_GAME = 301
TAIL_LAST_GAME = 500
PHASES = ("place", "move", "fly")

_POST_TRAINING_ANALYSIS_PATHS = frozenset(
    {
        "learned_ai/evaluation/mill_bonus_ablation_result.py",
        "tests/test_mill_bonus_ablation_result.py",
    }
)

_REQUIRED_GAME_FIELDS = {
    "game_id",
    "game",
    "difficulty",
    "learner_color",
    "temperature",
    "outcome",
    "ply",
    "steps",
    "update_policy_loss",
    "update_value_loss",
    "update_entropy",
    "reward_total_mean",
    "reward_mill_bonus_mean",
    "mill_bonus_awarded_total",
    "chosen_prob_mean",
    "entropy_mean",
    "policy_top1_rate",
    "heuristic_top1_rate",
    "malom_preserving_move_rate",
    "malom_downgrade_move_rate",
    "game_type",
    "phase_bucket",
    "is_branch",
    "termination_reason",
    "opponent_node_budget",
    "formed_mill_count",
    "formed_mill_move_count",
    "formed_mill_malom_unknown_count",
    "formed_mill_malom_downgrade_count",
    "formed_mill_malom_downgrade_rate",
    "formed_mill_malom_known_place",
    "formed_mill_malom_known_move",
    "formed_mill_malom_known_fly",
    "formed_mill_malom_downgrade_place",
    "formed_mill_malom_downgrade_move",
    "formed_mill_malom_downgrade_fly",
}

_RAW_CURVE_FIELDS = (
    "temperature",
    "ply",
    "outcome",
    "reward_total_mean",
    "reward_mill_bonus_mean",
    "mill_bonus_awarded_total",
    "chosen_prob_mean",
    "entropy_mean",
    "policy_top1_rate",
    "heuristic_top1_rate",
    "malom_preserving_move_rate",
    "malom_downgrade_move_rate",
)

_REQUIRED_UPDATE_FIELDS = {
    "game",
    "policy_loss",
    "value_loss",
    "entropy",
    "lr",
    "batch_steps",
    "reason",
}


class MillBonusAblationResultError(RuntimeError):
    """An input, identity, metric, or evidence relationship is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _strict_loads(text: str, *, source: Path, line: int | None = None) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                suffix = "" if line is None else f" line {line}"
                raise MillBonusAblationResultError(
                    f"duplicate JSON key {key!r}: {source}{suffix}"
                )
            value[key] = item
        return value

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        suffix = "" if line is None else f" line {line}"
        raise MillBonusAblationResultError(
            f"invalid JSON: {source}{suffix}"
        ) from exc


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MillBonusAblationResultError(f"cannot read JSON: {path}") from exc
    value = _strict_loads(text, source=path)
    if not isinstance(value, dict):
        raise MillBonusAblationResultError(f"JSON root is not an object: {path}")
    return value


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MillBonusAblationResultError(f"cannot read JSONL: {path}") from exc
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise MillBonusAblationResultError(
                f"blank JSONL record: {path} line {line_number}"
            )
        value = _strict_loads(line, source=path, line=line_number)
        if not isinstance(value, dict):
            raise MillBonusAblationResultError(
                f"JSONL record is not an object: {path} line {line_number}"
            )
        values.append(value)
    if not values:
        raise MillBonusAblationResultError(f"JSONL input is empty: {path}")
    return values


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MillBonusAblationResultError(f"{field} must be an integer >= {minimum}")
    return value


def _require_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MillBonusAblationResultError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MillBonusAblationResultError(f"{field} must be finite")
    return result


def _validate_finite_tree(value: Any, *, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_tree(item, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_tree(item, field=f"{field}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise MillBonusAblationResultError(f"{field} must be finite")


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _outcome_label(value: Any) -> str:
    outcome = _require_finite(value, field="outcome")
    if math.isclose(outcome, trainer.WIN_REWARD):
        return "win"
    if math.isclose(outcome, trainer.LOSS_REWARD):
        return "loss"
    if math.isclose(outcome, trainer.DRAW_SHORT) or math.isclose(
        outcome, trainer.DRAW_LONG
    ):
        return "draw"
    raise MillBonusAblationResultError(f"unsupported trainer outcome {outcome}")


def _primary_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numerator = sum(int(row["formed_mill_malom_downgrade_count"]) for row in rows)
    denominator = sum(
        int(row["formed_mill_move_count"])
        - int(row["formed_mill_malom_unknown_count"])
        for row in rows
    )
    return {
        "downgrading_known_mill_actions": numerator,
        "known_mill_actions": denominator,
        "rate": _rate(numerator, denominator),
    }


def _phase_primary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        phase: {
            "downgrading_known_mill_actions": sum(
                int(row[f"formed_mill_malom_downgrade_{phase}"])
                for row in rows
            ),
            "known_mill_actions": sum(
                int(row[f"formed_mill_malom_known_{phase}"])
                for row in rows
            ),
        }
        for phase in PHASES
    }


def _finish_phase_rates(
    value: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    for metrics in value.values():
        metrics["rate"] = _rate(
            metrics["downgrading_known_mill_actions"],
            metrics["known_mill_actions"],
        )
    return value


def _wdl(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(_outcome_label(row["outcome"]) for row in rows)
    total = len(rows)
    return {
        "games": total,
        "wins": counts["win"],
        "draws": counts["draw"],
        "losses": counts["loss"],
        "score_rate": (
            None
            if total == 0
            else (counts["win"] + 0.5 * counts["draw"]) / total
        ),
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    key,
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return dict(sorted(groups.items()))


def _group_wdl(
    rows: Sequence[Mapping[str, Any]],
    key,
) -> dict[str, dict[str, Any]]:
    return {name: _wdl(group) for name, group in _group_rows(rows, key).items()}


def _group_primary(
    rows: Sequence[Mapping[str, Any]],
    key,
) -> dict[str, dict[str, Any]]:
    return {
        name: _primary_counts(group)
        for name, group in _group_rows(rows, key).items()
    }


def _raw_curve(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for row in rows:
        primary = _primary_counts((row,))
        values.append(
            {
                "game": row["game"],
                "primary_numerator": primary[
                    "downgrading_known_mill_actions"
                ],
                "primary_denominator": primary["known_mill_actions"],
                "primary_rate": primary["rate"],
                **{field: row[field] for field in _RAW_CURVE_FIELDS},
            }
        )
    return values


def _rolling_curve(
    rows: Sequence[Mapping[str, Any]], window: int = ROLLING_WINDOW
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for end in range(window, len(rows) + 1):
        sample = rows[end - window : end]
        primary = _primary_counts(sample)
        point: dict[str, Any] = {
            "game": sample[-1]["game"],
            "window_games": window,
            "primary_numerator": primary[
                "downgrading_known_mill_actions"
            ],
            "primary_denominator": primary["known_mill_actions"],
            "primary_rate": primary["rate"],
        }
        for field in _RAW_CURVE_FIELDS:
            point[field] = sum(float(row[field]) for row in sample) / window
        values.append(point)
    return values


def _validate_game_row(row: Mapping[str, Any], *, expected_game: int) -> None:
    missing = _REQUIRED_GAME_FIELDS - set(row)
    if missing:
        raise MillBonusAblationResultError(
            f"game {expected_game} lacks fields: {sorted(missing)}"
        )
    _validate_finite_tree(row, field=f"game[{expected_game}]")
    if _require_int(row["game"], field="game", minimum=1) != expected_game:
        raise MillBonusAblationResultError("training games are not exactly 1..500")
    if row["learner_color"] not in {"W", "B"}:
        raise MillBonusAblationResultError("learner_color is invalid")
    if row["phase_bucket"] != "main" or row["is_branch"] != 0:
        raise MillBonusAblationResultError("ablation log contains a branch rollout")
    if row["difficulty"] != 1:
        raise MillBonusAblationResultError("ablation left Sanmill node level 1")
    if row["game_type"] not in {"vs_frozen", "vs_sanmill"}:
        raise MillBonusAblationResultError("opponent source is invalid")
    if row["game_type"] == "vs_sanmill":
        if row["opponent_node_budget"] != 1000:
            raise MillBonusAblationResultError("Sanmill node budget is not 1,000")
    elif row["opponent_node_budget"] is not None:
        raise MillBonusAblationResultError("frozen opponent has a node budget")

    mill_moves = _require_int(
        row["formed_mill_move_count"], field="formed_mill_move_count"
    )
    mills_formed = _require_int(
        row["formed_mill_count"], field="formed_mill_count"
    )
    if mills_formed < mill_moves:
        raise MillBonusAblationResultError(
            "formed Mill count is below Mill-forming action count"
        )
    unknown = _require_int(
        row["formed_mill_malom_unknown_count"],
        field="formed_mill_malom_unknown_count",
    )
    downgrade = _require_int(
        row["formed_mill_malom_downgrade_count"],
        field="formed_mill_malom_downgrade_count",
    )
    if unknown > mill_moves:
        raise MillBonusAblationResultError("unknown Mill count exceeds support")
    known = mill_moves - unknown
    if downgrade > known:
        raise MillBonusAblationResultError("downgrade Mill count exceeds support")
    phase_known = sum(
        _require_int(
            row[f"formed_mill_malom_known_{phase}"],
            field=f"formed_mill_malom_known_{phase}",
        )
        for phase in PHASES
    )
    phase_downgrade = sum(
        _require_int(
            row[f"formed_mill_malom_downgrade_{phase}"],
            field=f"formed_mill_malom_downgrade_{phase}",
        )
        for phase in PHASES
    )
    if phase_known != known or phase_downgrade != downgrade:
        raise MillBonusAblationResultError(
            "phase Mill counts do not reconcile with the whole game"
        )
    for phase in PHASES:
        if (
            row[f"formed_mill_malom_downgrade_{phase}"]
            > row[f"formed_mill_malom_known_{phase}"]
        ):
            raise MillBonusAblationResultError(
                f"{phase} downgrade count exceeds phase support"
            )
    expected_rate = _rate(downgrade, known) or 0.0
    logged_rate = _require_finite(
        row["formed_mill_malom_downgrade_rate"],
        field="formed_mill_malom_downgrade_rate",
    )
    if not math.isclose(logged_rate, expected_rate, abs_tol=1e-12):
        raise MillBonusAblationResultError("logged Mill downgrade rate differs")
    _outcome_label(row["outcome"])


def _validate_schedule_counts(
    rows: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]
) -> None:
    observed: Counter[str] = Counter()
    for row in rows:
        source = "frozen" if row["game_type"] == "vs_frozen" else "sanmill"
        colour = "white" if row["learner_color"] == "W" else "black"
        observed[f"{source}_{colour}"] += 1
    wanted = Counter({key: int(value) for key, value in expected.items()})
    if observed != wanted:
        raise MillBonusAblationResultError(
            f"scheduled opponent/colour counts differ: {dict(observed)}"
        )


def summarize_game_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_games: int,
    expected_schedule_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one arm's game ledger and return exact stratified metrics."""
    if len(rows) != expected_games:
        raise MillBonusAblationResultError(
            f"expected {expected_games} game rows, observed {len(rows)}"
        )
    game_ids: set[str] = set()
    for expected_game, row in enumerate(rows, start=1):
        _validate_game_row(row, expected_game=expected_game)
        game_id = row["game_id"]
        if not isinstance(game_id, str) or not game_id or game_id in game_ids:
            raise MillBonusAblationResultError("game identities are invalid or repeated")
        game_ids.add(game_id)
    _validate_schedule_counts(rows, expected_schedule_counts)
    tail = [
        row
        for row in rows
        if TAIL_FIRST_GAME <= int(row["game"]) <= TAIL_LAST_GAME
    ]
    if len(tail) != TAIL_LAST_GAME - TAIL_FIRST_GAME + 1:
        raise MillBonusAblationResultError("tail window is incomplete")

    def opponent_key(row: Mapping[str, Any]) -> Any:
        return row["game_type"]

    def colour_key(row: Mapping[str, Any]) -> Any:
        return row["learner_color"]

    def node_key(row: Mapping[str, Any]) -> Any:
        return (
            "frozen"
            if row["opponent_node_budget"] is None
            else row["opponent_node_budget"]
        )

    def termination_key(row: Mapping[str, Any]) -> Any:
        return row["termination_reason"]

    def opponent_colour_key(row: Mapping[str, Any]) -> Any:
        return f"{row['game_type']}:{row['learner_color']}"
    return {
        "games": len(rows),
        "tail_games": len(tail),
        "primary": {
            "whole_run": _primary_counts(rows),
            "tail_301_500": _primary_counts(tail),
            "whole_run_by_phase": _finish_phase_rates(_phase_primary(rows)),
            "tail_by_phase": _finish_phase_rates(_phase_primary(tail)),
            "tail_by_opponent_source": _group_primary(tail, opponent_key),
            "tail_by_learner_colour": _group_primary(tail, colour_key),
            "tail_by_node_level": _group_primary(tail, node_key),
            "tail_by_termination_reason": _group_primary(
                tail, termination_key
            ),
        },
        "wdl": {
            "all": _wdl(rows),
            "by_opponent_source": _group_wdl(rows, opponent_key),
            "by_learner_colour": _group_wdl(rows, colour_key),
            "by_opponent_and_colour": _group_wdl(
                rows, opponent_colour_key
            ),
            "by_node_level": _group_wdl(rows, node_key),
            "by_termination_reason": _group_wdl(rows, termination_key),
        },
        "curves": {
            "interpretation": (
                "observed training diagnostics only; no forecast or held-out "
                "strength curve"
            ),
            "raw": _raw_curve(rows),
            "rolling_50_complete_windows_only": _rolling_curve(rows),
            "validation": {
                "available": False,
                "reason": "ordinary RL run has no supervised validation curve",
            },
        },
    }


def summarize_update_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_games: int
) -> dict[str, Any]:
    """Validate and preserve optimizer-update curves separately from games."""
    previous_game = 0
    raw: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        missing = _REQUIRED_UPDATE_FIELDS - set(row)
        if missing:
            raise MillBonusAblationResultError(
                f"update {index} lacks fields: {sorted(missing)}"
            )
        _validate_finite_tree(row, field=f"update[{index}]")
        game = _require_int(row["game"], field="update.game", minimum=1)
        if game < previous_game or game > expected_games:
            raise MillBonusAblationResultError("update games are not monotonic")
        previous_game = game
        for field in ("policy_loss", "value_loss", "entropy", "lr"):
            _require_finite(row[field], field=f"update.{field}")
        _require_int(row["batch_steps"], field="update.batch_steps", minimum=1)
        if row["reason"] not in {"periodic", "final_flush"}:
            raise MillBonusAblationResultError("update reason is invalid")
        raw.append({field: row[field] for field in sorted(_REQUIRED_UPDATE_FIELDS)})
    return {
        "updates": len(raw),
        "raw": raw,
        "validation": {
            "available": False,
            "reason": "ordinary RL run has no supervised validation updates",
        },
    }


def decide_paired_result(
    arm_summaries: Sequence[Mapping[str, Any]],
    *,
    material_reduction: float,
) -> dict[str, Any]:
    """Apply the preregistered three-seed decision without pooling arms."""
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for arm in arm_summaries:
        by_seed[int(arm["seed"])][str(arm["mill_bonus_mode"])] = arm
    paired: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        modes = by_seed[seed]
        if set(modes) != {"legacy-unconditional", "malom-preserving-only"}:
            raise MillBonusAblationResultError(f"seed {seed} pair is incomplete")
        legacy = modes["legacy-unconditional"]["metrics"]["primary"][
            "tail_301_500"
        ]
        corrected = modes["malom-preserving-only"]["metrics"]["primary"][
            "tail_301_500"
        ]
        legacy_rate = legacy["rate"]
        corrected_rate = corrected["rate"]
        reduction = (
            None
            if legacy_rate is None or corrected_rate is None
            else legacy_rate - corrected_rate
        )
        paired.append(
            {
                "seed": seed,
                "legacy": legacy,
                "malom_preserving_only": corrected,
                "corrected_minus_legacy": (
                    None if reduction is None else -reduction
                ),
                "legacy_minus_corrected_rate": reduction,
            }
        )
    finite_reductions = [
        pair["legacy_minus_corrected_rate"]
        for pair in paired
        if pair["legacy_minus_corrected_rate"] is not None
    ]
    corrected_safe = all(
        bool(arm["policy_health"]["passed"])
        for arm in arm_summaries
        if arm["mill_bonus_mode"] == "malom-preserving-only"
    )
    negative_differences = sum(value > 0 for value in finite_reductions)
    median_reduction = (
        None
        if len(finite_reductions) != len(paired)
        else median(finite_reductions)
    )
    supports = (
        corrected_safe
        and len(finite_reductions) == len(paired) == 3
        and negative_differences >= 2
        and median_reduction is not None
        and median_reduction >= material_reduction
    )
    return {
        "verdict": (
            "supports_malom_preserving_only" if supports else "inconclusive"
        ),
        "paired_seed_results": paired,
        "corrected_arms_pass_safety": corrected_safe,
        "pairs_favouring_corrected": negative_differences,
        "median_legacy_minus_corrected_rate": median_reduction,
        "required_material_reduction": material_reduction,
    }


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MillBonusAblationResultError("Git evidence audit failed") from exc
    return result.stdout.strip()


def _inspect_analysis_source(root: Path, expected_commit: str) -> dict[str, Any]:
    branch = _git_output(root, "branch", "--show-current")
    head = _git_output(root, "rev-parse", "HEAD")
    upstream = _git_output(root, "rev-parse", "origin/dev")
    dirty = _git_output(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if branch != "dev" or head != upstream or dirty:
        raise MillBonusAblationResultError(
            "result analysis requires the clean published plan source on dev"
        )
    changed_paths: list[str] = []
    if head != expected_commit:
        merge_base = _git_output(root, "merge-base", expected_commit, head)
        if merge_base != expected_commit:
            raise MillBonusAblationResultError(
                "result analysis source does not descend from the training source"
            )
        changed_paths = sorted(
            path
            for path in _git_output(
                root,
                "diff",
                "--name-only",
                f"{expected_commit}..{head}",
                "--",
            ).splitlines()
            if path
        )
        if not changed_paths or not set(changed_paths).issubset(
            _POST_TRAINING_ANALYSIS_PATHS
        ):
            raise MillBonusAblationResultError(
                "post-training source changes are not analysis-only"
            )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": upstream,
        "training_source_commit": expected_commit,
        "post_training_analysis_paths": changed_paths,
        "worktree_clean": True,
    }


def _validate_readiness(
    path: Path,
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    readiness = _strict_json(path)
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise MillBonusAblationResultError("readiness schema differs")
    identity = readiness.get("readiness_identity")
    body = dict(readiness)
    body.pop("readiness_identity", None)
    if identity != canonical_sha256(body):
        raise MillBonusAblationResultError("readiness identity is invalid")
    if (
        readiness.get("state") != "ready_for_product_authorization"
        or readiness.get("launch_authorized") is not False
    ):
        raise MillBonusAblationResultError("readiness state differs")
    contract_record = readiness.get("contract")
    if not isinstance(contract_record, Mapping):
        raise MillBonusAblationResultError("readiness contract record is absent")
    if (
        contract_record.get("plan_identity") != contract["plan_identity"]
        or contract_record.get("file_sha256") != _sha256_file(contract_path)
    ):
        raise MillBonusAblationResultError("readiness binds another contract")
    if len(readiness.get("arms", ())) != 6:
        raise MillBonusAblationResultError("readiness does not bind six arms")
    return readiness


def _readiness_arm(
    readiness: Mapping[str, Any], arm_id: str
) -> Mapping[str, Any]:
    matches = [item for item in readiness["arms"] if item.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise MillBonusAblationResultError(f"readiness arm differs: {arm_id}")
    return matches[0]


def _validate_authorization(
    authorization: ManagedAuthorization, plan: ManagedPlan
) -> None:
    if (
        authorization.plan_id != plan.plan_id
        or authorization.plan_sha256 != plan.plan_sha256
        or authorization.allow_safe_exact_resume != plan.allow_safe_exact_resume
    ):
        raise MillBonusAblationResultError("authorization does not bind the plan")


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: ManagedPlan,
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
        raise MillBonusAblationResultError("run manifest source identity differs")
    config = manifest.get("resolved_config")
    if not isinstance(config, Mapping):
        raise MillBonusAblationResultError("run manifest config is absent")
    expected = {
        "seed": arm["seed"],
        "mill_bonus_mode": arm["mill_bonus_mode"],
        "max_games": contract["common_training_contract"][
            "max_games_schedule"
        ],
        "segment_games": contract["common_training_contract"][
            "one_segment_games"
        ],
        "segment_stop_game": contract["resources"]["completed_games_per_arm"],
        "start_mode": "fresh",
        "referee_engine": "sanmill",
        "opponent_engine": "sanmill",
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise MillBonusAblationResultError(
                f"run manifest config differs: {field}"
            )
    policy = manifest.get("checkpoint_policy")
    if not isinstance(policy, Mapping):
        raise MillBonusAblationResultError("run checkpoint policy is absent")
    mif = policy.get("mifSuite")
    ruleset = policy.get("ruleset")
    if not isinstance(mif, Mapping) or not isinstance(ruleset, Mapping):
        raise MillBonusAblationResultError("run protocol identity is absent")
    runtime_contract = contract["rules_and_runtime"]
    expected_protocol = {
        "tag": runtime_contract["mif_tag"],
        "releaseCommit": runtime_contract["mif_release_commit"],
        "suiteJcsSha256": "sha256:"
        + runtime_contract["mif_suite_jcs_sha256"],
    }
    for field, value in expected_protocol.items():
        if mif.get(field) != value:
            raise MillBonusAblationResultError(
                f"run MIF identity differs: {field}"
            )
    if ruleset.get("semanticDigest") != runtime_contract[
        "rules_semantic_digest"
    ]:
        raise MillBonusAblationResultError("run ruleset identity differs")
    if (
        preflight.get("schema_version") != "nmm.generalist-preflight.v1"
        or preflight.get("verdict") != "needs_decision"
        or preflight.get("errors") != []
        or preflight.get("unresolved_decisions")
        != [PRODUCT_AUTHORIZATION_DECISION]
        or preflight.get("resume_config_sha256") != plan.resume_config_sha256
    ):
        raise MillBonusAblationResultError("readiness preflight content differs")
    preflight_mif = preflight.get("mifSuite")
    preflight_ruleset = preflight.get("ruleset")
    if (
        not isinstance(preflight_mif, Mapping)
        or not isinstance(preflight_ruleset, Mapping)
        or dict(preflight_mif) != dict(mif)
        or preflight_ruleset.get("semanticDigest")
        != ruleset.get("semanticDigest")
    ):
        raise MillBonusAblationResultError(
            "run protocol identity differs from readiness"
        )
    asset_by_name = {
        asset.get("logical_name"): asset
        for asset in manifest.get("assets", ())
        if isinstance(asset, Mapping)
    }
    checks = preflight.get("checks")
    if not isinstance(checks, Mapping):
        raise MillBonusAblationResultError("readiness preflight checks are absent")

    def check_identity(name: str) -> str | None:
        record = checks.get(name)
        return record.get("identity") if isinstance(record, Mapping) else None

    specialist_check = checks.get("specialist_db")
    if (
        not isinstance(specialist_check, Mapping)
        or specialist_check.get("content_sha256")
        != contract["data_contract"]["specialist_db_initial_template"][
            "sha256"
        ]
    ):
        raise MillBonusAblationResultError(
            "readiness SpecialistDB content differs from the empty template"
        )
    if check_identity("malom") != contract["data_contract"][
        "malom_manifest_identity"
    ]:
        raise MillBonusAblationResultError("readiness Malom identity differs")
    if check_identity("human_db") != contract["data_contract"][
        "human_db_identity"
    ]:
        raise MillBonusAblationResultError("readiness HumanDB identity differs")
    expected_assets = {
        "mif_suite_1_0": preflight_mif.get("releaseManifestSha256"),
        "training_ruleset": ruleset.get("semanticDigest"),
        "malom_tablebase": check_identity("malom"),
        "specialist_db": check_identity("specialist_db"),
        "human_db": check_identity("human_db"),
        "sanmill_training_runtime": check_identity("sanmill_training"),
    }
    for name, identity in expected_assets.items():
        if not isinstance(identity, str) or (
            asset_by_name.get(name, {}).get("identity") != identity
        ):
            raise MillBonusAblationResultError(f"run asset differs: {name}")


def _validate_controller_completion(
    plan: ManagedPlan,
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
    if any(event.event_type in forbidden for event in events):
        raise MillBonusAblationResultError("controller ledger contains a safety stop")
    completed = [
        event for event in events if event.event_type == "managed_segment_completed"
    ]
    if (
        len(completed) != 1
        or events[-1].event_type != "managed_plan_completed"
        or completed[0].details.get("segment_index") != 1
        or completed[0].details.get("completed_games") != plan.game_bound
    ):
        raise MillBonusAblationResultError("controller completion evidence differs")
    completed_games, checkpoint = managed._inspect_completed_segment(
        plan,
        segment_index=1,
        previous_completed_games=0,
    )
    if completed_games != plan.game_bound:
        raise MillBonusAblationResultError("checkpoint completion bound differs")
    return completed[0].details, checkpoint


def _validate_policy_health(
    plan: ManagedPlan,
    *,
    details: Mapping[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    recorded = details.get("policy_health")
    if not isinstance(recorded, Mapping) or recorded.get("passed") is not True:
        raise MillBonusAblationResultError("policy-health gate did not pass")
    report_path = Path(str(recorded.get("report", ""))).resolve(strict=False)
    specialist_db = Path(
        managed._trainer_arg_value(plan, "--specialist-db")
    ).resolve(strict=False)
    validated = managed._validate_policy_health_report(
        plan,
        segment_index=1,
        report_path=report_path,
        checkpoint=checkpoint,
        specialist_db=specialist_db,
        completed_games=plan.game_bound,
        runtime_commit=plan.git_commit,
    )
    if (
        canonical_json_bytes(validated) != canonical_json_bytes(recorded)
        or not validated["passed"]
    ):
        raise MillBonusAblationResultError("policy-health evidence changed")
    return validated


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(path.resolve())
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _analyze_arm(
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    readiness: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
) -> dict[str, Any]:
    control_dir = _repository_path(
        root, arm["control_dir"], field="control_dir"
    )
    plan_path = control_dir / "plan.json"
    authorization_path = control_dir / "authorization.json"
    plan = load_managed_plan(plan_path)
    args = _assert_plan_semantics(
        plan,
        root=root,
        contract=contract,
        arm=arm,
        paths_config=paths_config,
        source_commit=source_commit,
    )
    ready_arm = _readiness_arm(readiness, str(arm["arm_id"]))
    if ready_arm.get("plan_sha256") != plan.plan_sha256:
        raise MillBonusAblationResultError("readiness plan hash differs")
    preflight = ready_arm.get("preflight")
    if not isinstance(preflight, Mapping):
        raise MillBonusAblationResultError("arm preflight evidence is absent")
    preflight_path = Path(str(preflight.get("path", "")))
    if (
        not preflight_path.is_file()
        or _sha256_file(preflight_path) != preflight.get("sha256")
    ):
        raise MillBonusAblationResultError("arm preflight evidence changed")
    preflight_report = _strict_json(preflight_path)
    authorization = load_managed_authorization(authorization_path)
    _validate_authorization(authorization, plan)
    completed_details, checkpoint = _validate_controller_completion(plan)
    health = _validate_policy_health(
        plan,
        details=completed_details,
        checkpoint=checkpoint,
    )

    segment = control_dir / "segments" / "segment-0001"
    manifest_path = segment / "run-manifest.json"
    train_log_path = segment / "train_log.jsonl"
    update_log_path = segment / "update_log.jsonl"
    run_events_path = segment / managed.RUN_EVENT_LEDGER_NAME
    manifest = _strict_json(manifest_path)
    _validate_manifest(
        manifest,
        plan=plan,
        arm=arm,
        contract=contract,
        preflight=preflight_report,
    )
    run_events = managed.load_run_events(run_events_path)
    if not run_events or run_events[-1].event_type != "training_completed":
        raise MillBonusAblationResultError("trainer lifecycle is incomplete")
    game_rows = _strict_jsonl(train_log_path)
    update_rows = _strict_jsonl(update_log_path)
    metrics = summarize_game_rows(
        game_rows,
        expected_games=contract["resources"]["completed_games_per_arm"],
        expected_schedule_counts=contract["resources"][
            "schedule_counts_per_arm"
        ][str(arm["seed"])],
    )
    update_metrics = summarize_update_rows(
        update_rows,
        expected_games=contract["resources"]["completed_games_per_arm"],
    )
    specialist_db = Path(args.specialist_db).resolve(strict=True)
    artifacts = {
        "plan": _artifact_record(root, plan_path),
        "authorization": _artifact_record(root, authorization_path),
        "controller_events": _artifact_record(
            root, control_dir / managed.CONTROLLER_LEDGER_NAME
        ),
        "preflight": _artifact_record(root, preflight_path),
        "run_manifest": _artifact_record(root, manifest_path),
        "run_events": _artifact_record(root, run_events_path),
        "train_log": _artifact_record(root, train_log_path),
        "update_log": _artifact_record(root, update_log_path),
        "checkpoint": _artifact_record(root, checkpoint),
        "specialist_db": _artifact_record(root, specialist_db),
        "policy_health": _artifact_record(root, Path(health["report"])),
    }
    return {
        "arm_id": arm["arm_id"],
        "seed": arm["seed"],
        "mill_bonus_mode": arm["mill_bonus_mode"],
        "plan_sha256": plan.plan_sha256,
        "authorization_file_sha256": _sha256_file(authorization_path),
        "experiment_id": plan.experiment_id,
        "source_commit": plan.git_commit,
        "schedule_max_games": plan.max_games,
        "completion_game_bound": plan.game_bound,
        "policy_health": health,
        "metrics": metrics,
        "optimizer_updates": update_metrics,
        "runtime_identities": {
            "mif": manifest["checkpoint_policy"]["mifSuite"],
            "ruleset": manifest["checkpoint_policy"]["ruleset"],
            "assets": manifest["assets"],
            "experiment_digest": manifest["checkpoint_policy"][
                "experimentDigest"
            ],
            "resume_config_sha256": plan.resume_config_sha256,
        },
        "artifacts": artifacts,
    }


def analyze_ablation_result(
    *,
    root: Path,
    contract_path: Path,
    readiness_path: Path,
    paths_config: Path,
) -> dict[str, Any]:
    """Validate six completed arms and produce one deterministic result."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    readiness_path = readiness_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    contract = load_ablation_contract(contract_path)
    readiness = _validate_readiness(
        readiness_path,
        contract=contract,
        contract_path=contract_path,
    )
    source_commit = str(readiness["source"]["head"])
    source = _inspect_analysis_source(root, source_commit)
    arm_summaries = [
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
    decision = decide_paired_result(
        arm_summaries,
        material_reduction=contract["analysis"]["decision_rule"][
            "material_absolute_reduction"
        ],
    )
    report_body = {
        "schema_version": RESULT_SCHEMA,
        "claim_boundary": contract["claim_boundary"],
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "plan_identity": contract["plan_identity"],
            "file_sha256": _sha256_file(contract_path),
        },
        "readiness": {
            "path": readiness_path.relative_to(root).as_posix(),
            "readiness_identity": readiness["readiness_identity"],
            "file_sha256": _sha256_file(readiness_path),
        },
        "analysis_source": source,
        "data_and_runtime_versions": {
            "data_contract": contract["data_contract"],
            "rules_and_runtime": contract["rules_and_runtime"],
        },
        "hyperparameters": contract["common_training_contract"],
        "baseline": {
            "type": "matched fresh legacy-unconditional reward arm",
            "pairing": contract["pairing"],
        },
        "arms": arm_summaries,
        "decision": decision,
        "interpretation": {
            "observation_facts": [
                "All six fresh 500-game arms completed with finite logs, exact "
                "identities, and passing fixed-state safety gates.",
                "The primary result is computed from exact support counts in "
                "games 301 through 500 for each seed and arm.",
                "Training and optimizer curves are observed values; no curve "
                "in this report is a forecast.",
            ],
            "hypothesis": contract["hypothesis"],
            "supporting_evidence": decision["paired_seed_results"],
            "counter_evidence_and_limits": [
                "This is a bounded reward-shaping ablation, not a playing-strength "
                "evaluation.",
                "The 29-state policy-health corpus is an inspected development "
                "diagnostic, not held-out validation.",
                "Ordinary RL provides no supervised validation curve in this run.",
                "W/D/L, loss, entropy, and top-1 metrics are secondary diagnostics "
                "and cannot replace the preregistered primary count ratio.",
            ],
            "next_verification_experiment": (
                "Only a supporting result may justify a separately frozen, "
                "multi-seed longer successor followed by a newly independent "
                "held-out strength evaluation. An inconclusive result requires "
                "diagnosis or redesign rather than automatic continuation."
            ),
        },
    }
    return {
        **report_body,
        "result_identity": canonical_sha256(report_body),
    }


def publish_result(path: Path, report: Mapping[str, Any]) -> None:
    """Publish one immutable canonical result after all validation passes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise MillBonusAblationResultError(
            f"result output already exists: {path}"
        ) from exc


__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_PATHS_CONFIG",
    "DEFAULT_READINESS_REPORT",
    "DEFAULT_RESULT",
    "MillBonusAblationResultError",
    "RESULT_SCHEMA",
    "analyze_ablation_result",
    "decide_paired_result",
    "publish_result",
    "summarize_game_rows",
    "summarize_update_rows",
]

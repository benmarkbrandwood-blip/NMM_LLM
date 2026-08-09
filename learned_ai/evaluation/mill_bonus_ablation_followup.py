"""Compact post-hoc audit of the completed mill-bonus ablation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from scripts import train_s_gen_v2 as trainer


FOLLOWUP_AUDIT_SCHEMA = "nmm.mill-bonus-ablation-followup-audit.v1"
EXPECTED_RESULT_SHA256 = (
    "8d7557483f102dcea548d1568f53523f51e0faa782bc2abaca6ff175055548fe"
)
EXPECTED_RESULT_IDENTITY = (
    "4c030be00932306c2270d5407c79d939991f65f24653f6af1a84e8308c5d134d"
)
CONTROL_MODE = "legacy-unconditional"
CORRECTED_MODE = "malom-preserving-only"
TAIL_FIRST_GAME = 301
TAIL_LAST_GAME = 500
SCHEDULE_FIELDS = (
    "game",
    "game_id",
    "difficulty",
    "learner_color",
    "temperature",
    "game_type",
    "opponent_node_budget",
)


class MillBonusAblationFollowupError(RuntimeError):
    """A published result, raw log, or paired schedule differs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_loads(text: str, *, source: Path, line: int | None = None) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                suffix = "" if line is None else f" line {line}"
                raise MillBonusAblationFollowupError(
                    f"duplicate JSON key {key!r}: {source}{suffix}"
                )
            value[key] = item
        return value

    try:
        return json.loads(text, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        suffix = "" if line is None else f" line {line}"
        raise MillBonusAblationFollowupError(
            f"invalid JSON: {source}{suffix}"
        ) from exc


def _load_result(
    path: Path,
    *,
    expected_sha256: str,
    expected_identity: str,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MillBonusAblationFollowupError(
            f"cannot read published result: {path}"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise MillBonusAblationFollowupError("published result hash differs")
    value = _strict_loads(raw.decode("utf-8"), source=path)
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise MillBonusAblationFollowupError(
            "published result is not a canonical JSON object"
        )
    identity = value.pop("result_identity", None)
    if identity != expected_identity or canonical_sha256(value) != identity:
        raise MillBonusAblationFollowupError("published result identity differs")
    if value.get("decision", {}).get("verdict") != "inconclusive":
        raise MillBonusAblationFollowupError(
            "published preregistered verdict is not inconclusive"
        )
    return {**value, "result_identity": identity}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MillBonusAblationFollowupError(f"cannot read raw log: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise MillBonusAblationFollowupError(
                f"blank raw-log line: {path} line {line_number}"
            )
        value = _strict_loads(line, source=path, line=line_number)
        if not isinstance(value, dict):
            raise MillBonusAblationFollowupError(
                f"raw-log row is not an object: {path} line {line_number}"
            )
        rows.append(value)
    return rows


def _integer_from_rate(rate: Any, denominator: int, *, field: str) -> int:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise MillBonusAblationFollowupError(f"{field} is not numeric")
    value = float(rate)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise MillBonusAblationFollowupError(f"{field} is outside [0, 1]")
    count = round(value * denominator)
    if not math.isclose(value * denominator, count, abs_tol=1e-8):
        raise MillBonusAblationFollowupError(
            f"{field} does not reconstruct an integer count"
        )
    return int(count)


def _row_counts(row: Mapping[str, Any]) -> dict[str, int]:
    steps = row.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise MillBonusAblationFollowupError("raw-log steps are invalid")
    known = _integer_from_rate(
        row.get("malom_known_move_rate"),
        steps,
        field="malom_known_move_rate",
    )
    downgrades = _integer_from_rate(
        row.get("malom_downgrade_move_rate"),
        known,
        field="malom_downgrade_move_rate",
    )
    known_mills = sum(
        int(row.get(f"formed_mill_malom_known_{phase}", -1))
        for phase in ("place", "move", "fly")
    )
    downgrade_mills = sum(
        int(row.get(f"formed_mill_malom_downgrade_{phase}", -1))
        for phase in ("place", "move", "fly")
    )
    if min(known_mills, downgrade_mills) < 0 or downgrade_mills > known_mills:
        raise MillBonusAblationFollowupError("raw-log Mill counts are invalid")
    if int(row.get("formed_mill_malom_downgrade_count", -1)) != downgrade_mills:
        raise MillBonusAblationFollowupError(
            "raw-log Mill downgrade cross-tab differs"
        )
    return {
        "learner_actions": steps,
        "known_actions": known,
        "downgrade_actions": downgrades,
        "known_mill_actions": known_mills,
        "downgrade_mill_actions": downgrade_mills,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        field: 0
        for field in (
            "learner_actions",
            "known_actions",
            "downgrade_actions",
            "known_mill_actions",
            "downgrade_mill_actions",
            "formed_mills",
        )
    }
    bonus_total = 0.0
    games_with_mill_downgrade = 0
    for row in rows:
        row_counts = _row_counts(row)
        for field, value in row_counts.items():
            counts[field] += value
        bonus = row.get("mill_bonus_awarded_total")
        if isinstance(bonus, bool) or not isinstance(bonus, (int, float)):
            raise MillBonusAblationFollowupError(
                "raw-log mill_bonus_awarded_total is invalid"
            )
        bonus_total += float(bonus)
        formed_mills = row.get("formed_mill_count")
        if (
            isinstance(formed_mills, bool)
            or not isinstance(formed_mills, int)
            or formed_mills < 0
        ):
            raise MillBonusAblationFollowupError(
                "raw-log formed_mill_count is invalid"
            )
        counts["formed_mills"] += formed_mills
        games_with_mill_downgrade += int(
            row_counts["downgrade_mill_actions"] > 0
        )
    return {
        **counts,
        "all_action_downgrade_rate": (
            counts["downgrade_actions"] / counts["known_actions"]
            if counts["known_actions"]
            else None
        ),
        "mill_action_downgrade_rate": (
            counts["downgrade_mill_actions"] / counts["known_mill_actions"]
            if counts["known_mill_actions"]
            else None
        ),
        "mill_bonus_awarded_total": bonus_total,
        "games_with_mill_downgrade": games_with_mill_downgrade,
    }


def _split_aggregate(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field))].append(row)
    return {key: _aggregate(grouped[key]) for key in sorted(grouped)}


def _two_proportion_independent_actions(
    control_rate: float,
    treatment_rate: float,
) -> int:
    """Optimistic normal-approximation actions/arm at alpha=.05, power=.80."""
    if not 0.0 < treatment_rate < control_rate < 1.0:
        raise MillBonusAblationFollowupError("power scenario rates are invalid")
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143
    pooled = (control_rate + treatment_rate) / 2.0
    numerator = (
        z_alpha * math.sqrt(2.0 * pooled * (1.0 - pooled))
        + z_power
        * math.sqrt(
            control_rate * (1.0 - control_rate)
            + treatment_rate * (1.0 - treatment_rate)
        )
    ) ** 2
    return math.ceil(numerator / ((control_rate - treatment_rate) ** 2))


def build_followup_audit(
    *,
    root: Path,
    result_path: Path,
    auditor: Mapping[str, Any],
    expected_result_sha256: str = EXPECTED_RESULT_SHA256,
    expected_result_identity: str = EXPECTED_RESULT_IDENTITY,
) -> dict[str, Any]:
    """Validate six raw arms and emit compact successor-design evidence."""
    result = _load_result(
        result_path,
        expected_sha256=expected_result_sha256,
        expected_identity=expected_result_identity,
    )
    arms = result.get("arms")
    if not isinstance(arms, list) or len(arms) != 6:
        raise MillBonusAblationFollowupError("published result lacks six arms")

    audited_arms: list[dict[str, Any]] = []
    rows_by_seed: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for arm in arms:
        seed = int(arm["seed"])
        mode = str(arm["mill_bonus_mode"])
        if mode not in (CONTROL_MODE, CORRECTED_MODE):
            raise MillBonusAblationFollowupError("published arm mode differs")
        artifact = arm["artifacts"]["train_log"]
        log_path = root / artifact["path"]
        if _sha256_file(log_path) != artifact["sha256"]:
            raise MillBonusAblationFollowupError("raw train-log hash differs")
        rows = _load_jsonl(log_path)
        if [row.get("game") for row in rows] != list(range(1, 501)):
            raise MillBonusAblationFollowupError("raw train-log games differ")
        rows_by_seed[seed][mode] = rows
        tail = [
            row
            for row in rows
            if TAIL_FIRST_GAME <= int(row["game"]) <= TAIL_LAST_GAME
        ]
        blocks = {
            f"{start}-{start + 99}": _aggregate(rows[start - 1 : start + 99])
            for start in range(1, 501, 100)
        }
        whole = _aggregate(rows)
        legacy_bonus_ceiling = trainer.MILL_BONUS * whole["formed_mills"]
        actual_bonus = whole["mill_bonus_awarded_total"]
        bonus_units = actual_bonus / trainer.MILL_BONUS
        if (
            actual_bonus < 0.0
            or actual_bonus > legacy_bonus_ceiling
            or not math.isclose(bonus_units, round(bonus_units), abs_tol=1e-9)
            or (
                mode == CONTROL_MODE
                and not math.isclose(
                    actual_bonus, legacy_bonus_ceiling, abs_tol=1e-9
                )
            )
        ):
            raise MillBonusAblationFollowupError(
                "raw reward exposure does not match the selected mode"
            )
        audited_arms.append(
            {
                "arm_id": arm["arm_id"],
                "seed": seed,
                "mode": mode,
                "train_log": {
                    "path": artifact["path"],
                    "sha256": artifact["sha256"],
                },
                "whole_run": whole,
                "tail_301_500": _aggregate(tail),
                "tail_by_opponent": _split_aggregate(tail, "game_type"),
                "tail_by_colour": _split_aggregate(tail, "learner_color"),
                "blocks_100_games": blocks,
                "counterfactual_removed_mill_reward": (
                    trainer.MILL_BONUS * whole["formed_mills"]
                    - whole["mill_bonus_awarded_total"]
                    if mode == CORRECTED_MODE
                    else 0.0
                ),
            }
        )

    paired: list[dict[str, Any]] = []
    for seed in sorted(rows_by_seed):
        pair = rows_by_seed[seed]
        if set(pair) != {CONTROL_MODE, CORRECTED_MODE}:
            raise MillBonusAblationFollowupError("paired seed modes differ")
        control_rows = pair[CONTROL_MODE]
        corrected_rows = pair[CORRECTED_MODE]
        for control, corrected in zip(control_rows, corrected_rows, strict=True):
            if any(control.get(field) != corrected.get(field) for field in SCHEDULE_FIELDS):
                raise MillBonusAblationFollowupError(
                    f"same-seed exogenous schedule differs: seed {seed}"
                )
        control_whole = _aggregate(control_rows)
        corrected_whole = _aggregate(corrected_rows)
        control_tail = _aggregate(control_rows[300:500])
        corrected_tail = _aggregate(corrected_rows[300:500])
        paired.append(
            {
                "seed": seed,
                "schedule_rows_equal": 500,
                "whole_run_control_minus_corrected": (
                    control_whole["all_action_downgrade_rate"]
                    - corrected_whole["all_action_downgrade_rate"]
                ),
                "tail_control_minus_corrected": (
                    control_tail["all_action_downgrade_rate"]
                    - corrected_tail["all_action_downgrade_rate"]
                ),
                "published_primary_control_minus_corrected": next(
                    item["legacy_minus_corrected_rate"]
                    for item in result["decision"]["paired_seed_results"]
                    if int(item["seed"]) == seed
                ),
            }
        )

    tail_differences = [row["tail_control_minus_corrected"] for row in paired]
    whole_differences = [row["whole_run_control_minus_corrected"] for row in paired]
    body = {
        "schema_version": FOLLOWUP_AUDIT_SCHEMA,
        "source_result": {
            "path": str(result_path.relative_to(root)).replace("\\", "/"),
            "sha256": expected_result_sha256,
            "result_identity": result["result_identity"],
            "preregistered_verdict": result["decision"]["verdict"],
        },
        "auditor": dict(auditor),
        "arms": audited_arms,
        "paired_seed_sensitivity": paired,
        "summary": {
            "primary_median_control_minus_corrected": result["decision"][
                "median_legacy_minus_corrected_rate"
            ],
            "all_action_tail_median_control_minus_corrected": median(
                tail_differences
            ),
            "all_action_tail_pairs_favouring_corrected": sum(
                value > 0.0 for value in tail_differences
            ),
            "all_action_whole_median_control_minus_corrected": median(
                whole_differences
            ),
            "all_action_whole_pairs_favouring_corrected": sum(
                value > 0.0 for value in whole_differences
            ),
            "corrected_mill_downgrade_events_affected": sum(
                arm["whole_run"]["downgrade_mill_actions"]
                for arm in audited_arms
                if arm["mode"] == CORRECTED_MODE
            ),
            "corrected_all_downgrade_events_potentially_affected": sum(
                arm["whole_run"]["downgrade_actions"]
                for arm in audited_arms
                if arm["mode"] == CORRECTED_MODE
            ),
            "corrected_games": sum(
                500 for arm in audited_arms if arm["mode"] == CORRECTED_MODE
            ),
        },
        "optimistic_independent_action_power": {
            "interpretation": (
                "normal-approximation actions per arm at two-sided alpha 0.05 "
                "and 80% power; this is a lower bound because actions cluster "
                "within games, policies and seeds"
            ),
            "scenarios": [
                {
                    "control_rate": control,
                    "treatment_rate": treatment,
                    "actions_per_arm": _two_proportion_independent_actions(
                        control, treatment
                    ),
                }
                for control, treatment in (
                    (0.09, 0.04),
                    (0.08, 0.06),
                    (0.08, 0.05),
                    (0.08, 0.04),
                )
            ],
        },
        "interpretation": {
            "observed_facts": [
                "The preregistered six-arm result is inconclusive and remains so.",
                "The old corrected mode changed reward only on Mill-forming exact-WDL downgrades.",
                "The old primary and all-action sensitivity results vary materially by seed.",
            ],
            "hypothesis": (
                "A direct asymmetric penalty on every exact-WDL downgrade may "
                "provide a denser and more targeted learning signal than merely "
                "withholding the contradictory Mill bonus."
            ),
            "supporting_evidence": (
                "Exact Malom quality was known for the learner actions in the "
                "audited logs, while the old treatment affected only the sparse "
                "Mill-downgrade subset."
            ),
            "counterevidence": [
                "The audit is post-hoc and cannot make the old result causal or positive.",
                "Action-level power calculations ignore within-game and within-seed clustering.",
                "A denser reward signal may still distort exploration or W/D/L learning.",
            ],
            "next_validation_experiment": (
                "Use fresh paired seeds and change only malom-preserving-only "
                "versus malom-preserving-plus-downgrade-penalty; keep the old "
                "five-percentage-point Mill-only gate immutable and use an "
                "all-action downgrade endpoint with explicit support and safety gates."
            ),
        },
        "claim_boundary": {
            "old_verdict_changed": False,
            "post_hoc_audit": True,
            "ordinary_supervised_validation": False,
            "playing_strength": False,
            "promotion": False,
        },
    }
    return {**body, "audit_identity": canonical_sha256(body)}

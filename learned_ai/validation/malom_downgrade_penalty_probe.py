"""No-update mechanism probe for exact-WDL downgrade penalties."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation.mill_bonus_no_update_probe import (
    MILL_BONUS_NO_UPDATE_PROBE_SCHEMA,
)
from scripts.train_s_gen_v2 import (
    _malom_downgrade_reward,
    _mill_formation_reward,
)


MALOM_DOWNGRADE_PENALTY_PROBE_SCHEMA = (
    "nmm.malom-downgrade-penalty-no-update-probe.v1"
)
EXPECTED_SOURCE_PROBE_SHA256 = (
    "0560c3fe3b89f32e4a9f59778c214167496e404be10ba24b03622fdc5a618f37"
)
EXPECTED_SOURCE_PROBE_IDENTITY = (
    "8f554f113ca65f05b8733f7e28b1e26177f58283c10b1c6f7d97abd603ef2186"
)
CONTROL_MODE = "malom-preserving-only"
TREATMENT_MODE = "malom-preserving-plus-downgrade-penalty"


class MalomDowngradePenaltyProbeError(RuntimeError):
    """A frozen source or no-update invariant differs."""


def _load_source_probe(
    path: Path,
    *,
    expected_sha256: str,
    expected_identity: str,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MalomDowngradePenaltyProbeError(
            f"cannot read frozen source probe: {path}"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise MalomDowngradePenaltyProbeError("frozen source probe hash differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalomDowngradePenaltyProbeError(
            "frozen source probe is invalid JSON"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise MalomDowngradePenaltyProbeError(
            "frozen source probe is not a canonical JSON object"
        )
    identity = value.pop("probe_identity", None)
    if (
        value.get("schema_version") != MILL_BONUS_NO_UPDATE_PROBE_SCHEMA
        or identity != expected_identity
        or canonical_sha256(value) != identity
    ):
        raise MalomDowngradePenaltyProbeError(
            "frozen source probe identity differs"
        )
    return {**value, "probe_identity": identity}


def _validate_auditor(auditor: Mapping[str, Any]) -> None:
    required = {
        "implementation_commit",
        "implementation_tree",
        "module_sha256",
        "script_sha256",
        "tracked_worktree_clean",
    }
    if set(auditor) != required or auditor.get("tracked_worktree_clean") is not True:
        raise MalomDowngradePenaltyProbeError(
            "probe implementation identity differs"
        )
    for field in required - {"tracked_worktree_clean"}:
        value = auditor.get(field)
        if not isinstance(value, str) or not value:
            raise MalomDowngradePenaltyProbeError(
                "probe implementation identity is incomplete"
            )


def _reward_components(row: Mapping[str, Any], mode: str) -> dict[str, float]:
    mills_formed = row.get("mills_formed")
    quality = row.get("malom_quality")
    if isinstance(mills_formed, bool) or not isinstance(mills_formed, int):
        raise MalomDowngradePenaltyProbeError("source mill count is invalid")
    if isinstance(quality, bool) or not isinstance(quality, (int, float)):
        raise MalomDowngradePenaltyProbeError("source Malom quality is invalid")
    quality_float = float(quality)
    if quality_float not in (-1.0, -2.0):
        raise MalomDowngradePenaltyProbeError(
            "source cohort contains a non-downgrade action"
        )
    mill = _mill_formation_reward(
        mills_formed=mills_formed,
        malom_quality=quality_float,
        mode=mode,
    )
    malom = _malom_downgrade_reward(
        malom_quality=quality_float,
        mode=mode,
    )
    return {
        "malom_downgrade": malom,
        "mill_formation": mill,
        "total": mill + malom,
    }


def build_malom_downgrade_penalty_probe(
    *,
    source_probe_path: Path,
    auditor: Mapping[str, Any],
    expected_source_sha256: str = EXPECTED_SOURCE_PROBE_SHA256,
    expected_source_identity: str = EXPECTED_SOURCE_PROBE_IDENTITY,
) -> dict[str, Any]:
    """Compare control and treatment rewards on frozen downgrade turns only."""
    _validate_auditor(auditor)
    source = _load_source_probe(
        source_probe_path,
        expected_sha256=expected_source_sha256,
        expected_identity=expected_source_identity,
    )
    source_rows = source.get("per_state")
    if not isinstance(source_rows, list) or not source_rows:
        raise MalomDowngradePenaltyProbeError("source probe has no states")

    per_state: list[dict[str, Any]] = []
    seen_ordinals: set[int] = set()
    for row in source_rows:
        if not isinstance(row, Mapping):
            raise MalomDowngradePenaltyProbeError("source state is not an object")
        ordinal = row.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise MalomDowngradePenaltyProbeError("source ordinal is invalid")
        if ordinal in seen_ordinals:
            raise MalomDowngradePenaltyProbeError("source ordinal is duplicated")
        seen_ordinals.add(ordinal)
        required_strings = (
            "after_fen",
            "before_fen",
            "game_id",
            "move_identity",
            "phase",
            "stratum",
            "turn_identity",
        )
        if any(not isinstance(row.get(field), str) or not row[field] for field in required_strings):
            raise MalomDowngradePenaltyProbeError(
                "source state identity is incomplete"
            )
        control = _reward_components(row, CONTROL_MODE)
        treatment = _reward_components(row, TREATMENT_MODE)
        per_state.append(
            {
                "after_fen": row["after_fen"],
                "before_fen": row["before_fen"],
                "game_id": row["game_id"],
                "malom_quality": float(row["malom_quality"]),
                "mills_formed": int(row["mills_formed"]),
                "move": row["move"],
                "move_identity": row["move_identity"],
                "ordinal": ordinal,
                "phase": row["phase"],
                "rewards": {
                    "control": control,
                    "treatment": treatment,
                },
                "stratum": row["stratum"],
                "turn_identity": row["turn_identity"],
            }
        )

    control_total = sum(row["rewards"]["control"]["total"] for row in per_state)
    treatment_total = sum(
        row["rewards"]["treatment"]["total"] for row in per_state
    )
    body = {
        "schema_version": MALOM_DOWNGRADE_PENALTY_PROBE_SCHEMA,
        "source_probe": {
            "probe_identity": source["probe_identity"],
            "sha256": expected_source_sha256,
        },
        "auditor": dict(auditor),
        "modes": {
            "control": CONTROL_MODE,
            "treatment": TREATMENT_MODE,
        },
        "summary": {
            "states": len(per_state),
            "affected_states": sum(
                row["rewards"]["treatment"]["total"]
                != row["rewards"]["control"]["total"]
                for row in per_state
            ),
            "mill_forming_states": sum(
                int(row["mills_formed"] > 0) for row in per_state
            ),
            "non_mill_states": sum(
                int(row["mills_formed"] == 0) for row in per_state
            ),
            "quality_rank_counts": dict(
                sorted(
                    Counter(
                        str(int(-row["malom_quality"])) for row in per_state
                    ).items()
                )
            ),
            "phase_counts": dict(
                sorted(Counter(row["phase"] for row in per_state).items())
            ),
            "stratum_counts": dict(
                sorted(Counter(row["stratum"] for row in per_state).items())
            ),
            "control_reward_total": control_total,
            "treatment_reward_total": treatment_total,
            "treatment_minus_control": treatment_total - control_total,
        },
        "per_state": per_state,
        "claim_boundary": {
            "candidate_policy_loaded": False,
            "new_games": False,
            "optimizer_created": False,
            "weights_updated": False,
            "actions_changed_between_modes": False,
            "states_changed_between_modes": False,
            "reward_component_only": True,
            "causal_training_effect_proven": False,
        },
    }
    return {**body, "probe_identity": canonical_sha256(body)}

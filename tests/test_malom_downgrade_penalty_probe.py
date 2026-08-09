"""Focused tests for the exact-WDL downgrade-penalty no-update probe."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation.malom_downgrade_penalty_probe import (
    CONTROL_MODE,
    TREATMENT_MODE,
    MalomDowngradePenaltyProbeError,
    build_malom_downgrade_penalty_probe,
)
from learned_ai.validation.mill_bonus_no_update_probe import (
    MILL_BONUS_NO_UPDATE_PROBE_SCHEMA,
)
from scripts import train_s_gen_v2 as trainer


def _auditor() -> dict[str, object]:
    return {
        "implementation_commit": "a" * 40,
        "implementation_tree": "b" * 40,
        "module_sha256": "c" * 64,
        "script_sha256": "d" * 64,
        "tracked_worktree_clean": True,
    }


def _source_row(
    *,
    ordinal: int,
    quality: float,
    mills_formed: int,
    phase: str,
) -> dict[str, object]:
    return {
        "after_fen": f"after-{ordinal}",
        "before_fen": f"before-{ordinal}",
        "game_id": f"game-{ordinal}",
        "malom_quality": quality,
        "mills_formed": mills_formed,
        "move": {"from": None, "to": "a7", "capture": None},
        "move_identity": f"move-{ordinal}",
        "ordinal": ordinal,
        "phase": phase,
        "stratum": "book" if ordinal == 1 else "perfect_db",
        "turn_identity": f"turn-{ordinal}",
    }


def _write_source(path: Path) -> tuple[str, str]:
    body = {
        "schema_version": MILL_BONUS_NO_UPDATE_PROBE_SCHEMA,
        "per_state": [
            _source_row(
                ordinal=1,
                quality=-1.0,
                mills_formed=1,
                phase="place",
            ),
            _source_row(
                ordinal=2,
                quality=-2.0,
                mills_formed=0,
                phase="move",
            ),
        ],
    }
    source = {**body, "probe_identity": canonical_sha256(body)}
    raw = canonical_json_bytes(source)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), source["probe_identity"]


def test_probe_penalises_mill_and_non_mill_downgrades(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source_sha256, source_identity = _write_source(source)

    probe = build_malom_downgrade_penalty_probe(
        source_probe_path=source,
        auditor=_auditor(),
        expected_source_sha256=source_sha256,
        expected_source_identity=source_identity,
    )

    assert probe["modes"] == {
        "control": CONTROL_MODE,
        "treatment": TREATMENT_MODE,
    }
    first, second = probe["per_state"]
    assert first["rewards"]["control"]["total"] == 0.0
    assert first["rewards"]["treatment"]["mill_formation"] == 0.0
    assert first["rewards"]["treatment"]["malom_downgrade"] == pytest.approx(
        -trainer.MALOM_DOWNGRADE_PENALTY
    )
    assert second["rewards"]["treatment"]["total"] == pytest.approx(
        -2 * trainer.MALOM_DOWNGRADE_PENALTY
    )
    assert probe["summary"] == {
        "states": 2,
        "affected_states": 2,
        "mill_forming_states": 1,
        "non_mill_states": 1,
        "quality_rank_counts": {"1": 1, "2": 1},
        "phase_counts": {"move": 1, "place": 1},
        "stratum_counts": {"book": 1, "perfect_db": 1},
        "control_reward_total": 0.0,
        "treatment_reward_total": pytest.approx(
            -3 * trainer.MALOM_DOWNGRADE_PENALTY
        ),
        "treatment_minus_control": pytest.approx(
            -3 * trainer.MALOM_DOWNGRADE_PENALTY
        ),
    }
    identity = probe.pop("probe_identity")
    assert identity == canonical_sha256(probe)


def test_probe_rejects_source_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _, source_identity = _write_source(source)

    with pytest.raises(MalomDowngradePenaltyProbeError, match="hash differs"):
        build_malom_downgrade_penalty_probe(
            source_probe_path=source,
            auditor=_auditor(),
            expected_source_sha256="0" * 64,
            expected_source_identity=source_identity,
        )


def test_probe_rejects_non_downgrade_source_state(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    body = {
        "schema_version": MILL_BONUS_NO_UPDATE_PROBE_SCHEMA,
        "per_state": [
            _source_row(
                ordinal=1,
                quality=0.0,
                mills_formed=1,
                phase="place",
            )
        ],
    }
    value = {**body, "probe_identity": canonical_sha256(body)}
    raw = canonical_json_bytes(value)
    source.write_bytes(raw)

    with pytest.raises(
        MalomDowngradePenaltyProbeError,
        match="non-downgrade",
    ):
        build_malom_downgrade_penalty_probe(
            source_probe_path=source,
            auditor=_auditor(),
            expected_source_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_identity=value["probe_identity"],
        )

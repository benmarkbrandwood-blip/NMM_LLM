from __future__ import annotations

from pathlib import Path

import pytest

from learned_ai.validation.classical_a_pos_distillation_readiness import (
    FROZEN_TRAINING_PROFILE,
    DistillationReadinessError,
    build_supervised_run_manifest,
    run_distillation_preflight,
    validate_training_profile,
)


def test_profile_is_pure_supervised_and_freezes_all_update_semantics() -> None:
    profile = validate_training_profile(FROZEN_TRAINING_PROFILE)
    manifest = build_supervised_run_manifest(
        profile,
        corpus_identity="a" * 64,
        split_identity="b" * 64,
    )

    assert manifest["training_semantics"] == (
        "offline-supervised-hard-a-pos-masked-one-hot-cross-entropy"
    )
    assert manifest["optimizer"]["parameter_scope"] == "policy_mlp-only"
    assert manifest["optimizer_updates_per_seed"] == 2240
    assert manifest["rl"] == {
        "enabled": False,
        "a2c": False,
        "ppo": False,
        "value_loss": False,
        "entropy": False,
    }


def test_forbidden_asset_is_rejected_before_corpus_is_read(tmp_path: Path) -> None:
    profile = {**FROZEN_TRAINING_PROFILE, "human_db": str(tmp_path / "human.db")}
    calls: list[Path] = []

    def unexpected_loader(path: Path):
        calls.append(path)
        raise AssertionError("forbidden profile must fail before corpus loading")

    with pytest.raises(DistillationReadinessError, match="unknown.*human_db"):
        run_distillation_preflight(
            profile,
            corpus_manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "output",
            corpus_loader=unexpected_loader,
        )

    assert calls == []


def test_online_or_rl_semantics_cannot_be_enabled() -> None:
    profile = {
        **FROZEN_TRAINING_PROFILE,
        "forbidden": {
            **FROZEN_TRAINING_PROFILE["forbidden"],
            "online_teacher_queries": True,
        },
    }

    with pytest.raises(DistillationReadinessError, match="profile differs"):
        validate_training_profile(profile)

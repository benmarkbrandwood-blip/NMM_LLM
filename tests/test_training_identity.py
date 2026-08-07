"""Focused tests for MIF, ruleset, and experiment training identities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.training.training_identity import (
    MIF_RELEASE_COMMIT,
    MIF_RELEASE_MANIFEST_SHA256,
    MIF_SUITE_JCS_SHA256,
    MIF_SUITE_TAG,
    TRAINER_RULESET_SEMANTIC_DIGEST,
    TrainingIdentityError,
    experiment_digest,
    load_trainer_ruleset,
    mif_release_identity,
)


ROOT = Path(__file__).resolve().parents[1]
RULESET = ROOT / "data" / "rulesets" / "nmm-training-core@2.json"


def test_final_mif_release_identity_is_frozen() -> None:
    assert mif_release_identity() == {
        "tag": MIF_SUITE_TAG,
        "releaseCommit": MIF_RELEASE_COMMIT,
        "suiteJcsSha256": MIF_SUITE_JCS_SHA256,
        "finalEvidenceSha256": (
            "sha256:2c23983281858386bc66e3adfce52f365c712d9e63a31c53f6a68bd6b2de08e1"
        ),
        "releaseManifestSha256": MIF_RELEASE_MANIFEST_SHA256,
        "claim": "exact-for-tested-domain",
    }


def test_tracked_ruleset_matches_implemented_training_semantics() -> None:
    identity = load_trainer_ruleset(RULESET)

    assert identity.ruleset_id == "nmm-training-core"
    assert identity.version == 2
    assert identity.semantic_digest == TRAINER_RULESET_SEMANTIC_DIGEST


def test_semantic_ruleset_change_fails_closed(tmp_path: Path) -> None:
    value = json.loads(RULESET.read_text(encoding="utf-8"))
    value["draw"]["repetition"] = {
        "count": 3,
        "mode": "claim",
        "observation": "stable-primary-decision-v1",
        "projection": "repetition-observation-v1",
        "resetEvents": ["board-remove"],
        "summary": "reset-count-smt-v1",
    }
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(TrainingIdentityError, match="does not match"):
        load_trainer_ruleset(changed)


def test_experiment_digest_is_stable_and_sensitive_to_semantic_inputs() -> None:
    ruleset = load_trainer_ruleset(RULESET)
    values = {
        "experiment_id": "experiment",
        "git_commit": "a" * 40,
        "resume_config_sha256": "b" * 64,
        "immutable_assets": {"malom": "c" * 64, "human": "d" * 64},
        "ruleset": ruleset,
    }

    first = experiment_digest(**values)
    assert experiment_digest(**values) == first
    assert first.startswith("sha256:") and len(first) == 71

    changed = dict(values)
    changed["resume_config_sha256"] = "e" * 64
    assert experiment_digest(**changed) != first

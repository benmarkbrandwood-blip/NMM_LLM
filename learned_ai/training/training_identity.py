"""Frozen protocol, ruleset, and experiment identities for training runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learned_ai.interop.mif_v1.model import resolve_manifest
from learned_ai.training.run_contract import canonical_sha256


MIF_SUITE_TAG = "mif-suite-1.0"
MIF_RELEASE_COMMIT = "a0a0f21cff5d6fbde045cd1482e416b92e0dc45a"
MIF_SUITE_JCS_SHA256 = (
    "sha256:81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f"
)
MIF_FINAL_EVIDENCE_SHA256 = (
    "sha256:2c23983281858386bc66e3adfce52f365c712d9e63a31c53f6a68bd6b2de08e1"
)
MIF_RELEASE_MANIFEST_SHA256 = (
    "sha256:dde89416bf5251cdc445ebdb9b92a899f58ec3930d1d8077ae26f1cb1a084499"
)

TRAINER_RULESET_ID = "nmm-training-core"
TRAINER_RULESET_VERSION = 1
TRAINER_RULESET_SEMANTIC_DIGEST = (
    "sha256:8f3ae8ac476b672418295d30e1a725d759ca424f9e670b7335154a934c7f2979"
)


class TrainingIdentityError(ValueError):
    """Raised when a protocol or ruleset identity is missing or inconsistent."""


@dataclass(frozen=True)
class RulesetIdentity:
    path: Path
    ruleset_id: str
    version: int
    semantic_digest: str
    document_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.ruleset_id,
            "version": self.version,
            "semanticDigest": self.semantic_digest,
            "documentDigest": self.document_digest,
        }


def mif_release_identity() -> dict[str, str]:
    """Return the immutable MIF Suite release identity required by every run."""
    return {
        "tag": MIF_SUITE_TAG,
        "releaseCommit": MIF_RELEASE_COMMIT,
        "suiteJcsSha256": MIF_SUITE_JCS_SHA256,
        "finalEvidenceSha256": MIF_FINAL_EVIDENCE_SHA256,
        "releaseManifestSha256": MIF_RELEASE_MANIFEST_SHA256,
        "claim": "exact-for-tested-domain",
    }


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TrainingIdentityError(
                    f"duplicate JSON key {key!r} in ruleset manifest"
                )
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingIdentityError(f"cannot read ruleset manifest: {path}") from exc
    if not isinstance(value, dict):
        raise TrainingIdentityError("ruleset manifest must be a JSON object")
    return value


def load_trainer_ruleset(path: str | Path) -> RulesetIdentity:
    """Resolve and verify the only rules semantics implemented by this trainer."""
    source = Path(path)
    manifest = _strict_json_object(source)
    try:
        resolved = resolve_manifest(manifest)
    except Exception as exc:
        raise TrainingIdentityError(f"ruleset manifest is not supported: {exc}") from exc
    identity = RulesetIdentity(
        path=source,
        ruleset_id=str(resolved.manifest["id"]),
        version=int(resolved.manifest["version"]),
        semantic_digest=resolved.semantic_digest,
        document_digest=resolved.document_digest,
    )
    expected = (
        TRAINER_RULESET_ID,
        TRAINER_RULESET_VERSION,
        TRAINER_RULESET_SEMANTIC_DIGEST,
    )
    observed = (identity.ruleset_id, identity.version, identity.semantic_digest)
    if observed != expected:
        raise TrainingIdentityError(
            "ruleset semantic identity does not match train_s_gen_v2 behavior"
        )
    return identity


def experiment_digest(
    *,
    experiment_id: str,
    git_commit: str,
    resume_config_sha256: str,
    immutable_assets: Mapping[str, str],
    ruleset: RulesetIdentity,
) -> str:
    """Hash the stable semantic inputs shared by all segments of an experiment."""
    value = {
        "format": "NMM-EXPERIMENT-IDENTITY/1",
        "experimentId": experiment_id,
        "gitCommit": git_commit,
        "resumeConfigSha256": resume_config_sha256,
        "mifSuite": mif_release_identity(),
        "ruleset": {
            "id": ruleset.ruleset_id,
            "version": ruleset.version,
            "semanticDigest": ruleset.semantic_digest,
        },
        "immutableAssets": dict(sorted(immutable_assets.items())),
    }
    return "sha256:" + canonical_sha256(value)

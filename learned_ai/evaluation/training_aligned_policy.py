"""Fail-closed policy adapter for the exact Generalist v2 training route."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ai.human_db import HumanDB
from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.delivery.training_route_bundle import (
    load_training_route_models,
)
from learned_ai.models.lookahead_advisor import LookaheadAdvisor
from learned_ai.models.scaffolded_encoder import (
    encode_position_with_lookahead,
)
from learned_ai.models.training_rollout_heuristic import (
    training_rollout_evaluate,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.training.generalist_preflight import (
    _probe_human_db,
    _probe_malom,
    _probe_specialist_db,
)


class TrainingAlignedPolicyError(RuntimeError):
    """Raised when the frozen route cannot be reconstructed exactly."""


def _require_identity(
    name: str,
    report: dict[str, Any],
    *,
    field: str,
    expected: str,
) -> None:
    error = report.get("error")
    if error:
        raise TrainingAlignedPolicyError(f"{name} verification failed: {error}")
    if report.get(field) != expected:
        raise TrainingAlignedPolicyError(f"{name} identity mismatch")


def _verify_resource_paths(
    manifest: dict[str, Any],
    *,
    human_db_path: Path,
    specialist_db_path: Path,
    malom_path: Path,
    malom_manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """Repeat the producer probes and bind them to bundle identities."""
    resources = manifest["resources"]

    human_report = _probe_human_db(human_db_path)
    _require_identity(
        "HumanDB",
        human_report,
        field="identity",
        expected=resources["human_db"]["identity"],
    )
    if human_report.get("malom_columns_policy") != (
        "masked_historical_labels"
    ):
        raise TrainingAlignedPolicyError(
            "HumanDB historical Malom columns are not masked"
        )

    specialist_report = _probe_specialist_db(specialist_db_path)
    _require_identity(
        "SpecialistDB",
        specialist_report,
        field="content_sha256",
        expected=resources["specialist_db"]["identity"],
    )
    if specialist_report.get("label_version") != "sector-corrected-v1":
        raise TrainingAlignedPolicyError(
            "SpecialistDB label version is not sector-corrected-v1"
        )

    malom_report = _probe_malom(malom_path, malom_manifest_path)
    _require_identity(
        "Malom",
        malom_report,
        field="identity",
        expected=resources["malom_tablebase"]["identity"],
    )
    if malom_report.get("available") is not True:
        raise TrainingAlignedPolicyError("Malom decoder is unavailable")

    return {
        "human_db": human_report,
        "specialist_db": specialist_report,
        "malom_tablebase": malom_report,
    }


class TrainingAlignedPolicy:
    """Argmax policy whose feature route matches the frozen training route."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        target: torch.nn.Module,
        manifest: dict[str, Any],
        human_db: HumanDB,
        specialist_db: SpecialistDB,
        malom: ExternalSolvedDB,
        resource_reports: dict[str, dict[str, Any]],
        device: str,
    ) -> None:
        self.policy = policy
        self.target = target
        self.manifest = manifest
        self.human_db = human_db
        self.specialist_db = specialist_db
        self.malom = malom
        self.resource_reports = resource_reports
        self.device = device
        route = manifest["route"]
        self.lookahead_advisor = LookaheadAdvisor(
            sentinel=None,
            evaluate_fn=training_rollout_evaluate,
            value_net=None,
            gap_net=None,
            human_db=human_db,
            use_sentinel=False,
            endgame_db=malom,
            ply_depth=int(route["ply_depth"]),
            frozen_model=target,
            frozen_device=device,
            sim_ply_depth=int(route["sim_ply_depth"]),
            strict=True,
        )
        self._closed = False

    @property
    def bundle_identity(self) -> str:
        return str(self.manifest["bundle_identity"])

    def choose_move(self, board) -> dict[str, Any]:
        """Choose the final policy argmax or fail on route degradation."""
        encoded = encode_position_with_lookahead(
            board,
            board.turn,
            sentinel_advisor=None,
            db=None,
            value_net=None,
            lookahead_advisor=self.lookahead_advisor,
            specialist_db=self.specialist_db,
            sdb_min_samples=3,
            strict=True,
        )
        if encoded is None or not encoded.legal_moves:
            return {}
        features = np.asarray(encoded.feat_matrix, dtype=np.float32)
        expected_width = int(self.manifest["route"]["feature_width"])
        if features.ndim != 2 or features.shape != (
            len(encoded.legal_moves),
            expected_width,
        ):
            raise TrainingAlignedPolicyError(
                "training-route feature matrix has an incompatible shape"
            )
        if not np.isfinite(features).all():
            raise TrainingAlignedPolicyError(
                "training-route features contain non-finite values"
            )
        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            logits = self.policy.policy_logits(feature_tensor)
        if logits.ndim != 1 or logits.shape[0] != len(encoded.legal_moves):
            raise TrainingAlignedPolicyError(
                "training-route policy returned an incompatible logit shape"
            )
        if not torch.isfinite(logits).all():
            raise TrainingAlignedPolicyError(
                "training-route policy returned non-finite logits"
            )
        return encoded.legal_moves[int(torch.argmax(logits).item())]

    def close(self) -> None:
        if self._closed:
            return
        self.specialist_db.close()
        self.human_db.close()
        self._closed = True

    def __enter__(self) -> "TrainingAlignedPolicy":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def load_training_aligned_policy(
    bundle_path: str | Path,
    *,
    human_db_path: str | Path,
    specialist_db_path: str | Path,
    malom_path: str | Path,
    malom_manifest_path: str | Path,
    device: str = "cpu",
) -> TrainingAlignedPolicy:
    """Verify a route bundle and its machine-local dependencies, then load it."""
    policy, target, manifest = load_training_route_models(
        bundle_path,
        device=device,
    )
    human_path = Path(human_db_path)
    specialist_path = Path(specialist_db_path)
    tablebase_path = Path(malom_path)
    tablebase_manifest = Path(malom_manifest_path)
    reports = _verify_resource_paths(
        manifest,
        human_db_path=human_path,
        specialist_db_path=specialist_path,
        malom_path=tablebase_path,
        malom_manifest_path=tablebase_manifest,
    )

    human_db = None
    specialist_db = None
    try:
        human_db = HumanDB(human_path, read_only=True)
        if not human_db.is_available():
            raise TrainingAlignedPolicyError("HumanDB read-only open failed")
        specialist_db = SpecialistDB(specialist_path, read_only=True)
        specialist_db.require_trusted_malom_labels()
        malom = ExternalSolvedDB(
            str(tablebase_path),
            strict=True,
        )
        if not malom.is_available():
            raise TrainingAlignedPolicyError("Malom strict decoder is unavailable")
        return TrainingAlignedPolicy(
            policy=policy,
            target=target,
            manifest=manifest,
            human_db=human_db,
            specialist_db=specialist_db,
            malom=malom,
            resource_reports=reports,
            device=device,
        )
    except Exception as exc:
        if specialist_db is not None:
            specialist_db.close()
        if human_db is not None:
            human_db.close()
        if isinstance(exc, TrainingAlignedPolicyError):
            raise
        raise TrainingAlignedPolicyError(
            "cannot open verified training-route dependencies"
        ) from exc

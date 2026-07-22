from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import learned_ai.evaluation.training_aligned_policy as route_policy
from game.board import BoardState


class _PolicyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def policy_logits(self, features):
        return torch.arange(
            features.shape[0], dtype=torch.float32, device=features.device
        )


class _HumanDB:
    def __init__(self, path, *, read_only):
        self.path = path
        self.read_only = read_only
        self.closed = False

    def is_available(self):
        return True

    def close(self):
        self.closed = True


class _SpecialistDB:
    def __init__(self, path, *, read_only):
        self.path = path
        self.read_only = read_only
        self.closed = False
        self.trusted = False

    def require_trusted_malom_labels(self):
        self.trusted = True

    def close(self):
        self.closed = True


class _Malom:
    def __init__(self, path, *, strict):
        self.path = path
        self.strict = strict

    def is_available(self):
        return True


def _manifest() -> dict:
    return {
        "bundle_identity": "b" * 64,
        "route": {
            "feature_width": 134,
            "ply_depth": 12,
            "sim_ply_depth": 5,
        },
        "resources": {
            "human_db": {"identity": "1" * 64},
            "specialist_db": {"identity": "2" * 64},
            "malom_tablebase": {"identity": "3" * 64},
        },
    }


def _patch_verified_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        route_policy,
        "load_training_route_models",
        lambda _path, device: (_PolicyModel(), _PolicyModel(), _manifest()),
    )
    monkeypatch.setattr(
        route_policy,
        "_probe_human_db",
        lambda _path: {
            "identity": "1" * 64,
            "malom_columns_policy": "masked_historical_labels",
        },
    )
    monkeypatch.setattr(
        route_policy,
        "_probe_specialist_db",
        lambda _path: {
            "content_sha256": "2" * 64,
            "label_version": "sector-corrected-v1",
        },
    )
    monkeypatch.setattr(
        route_policy,
        "_probe_malom",
        lambda _path, _manifest_path: {
            "identity": "3" * 64,
            "available": True,
        },
    )
    monkeypatch.setattr(route_policy, "HumanDB", _HumanDB)
    monkeypatch.setattr(route_policy, "SpecialistDB", _SpecialistDB)
    monkeypatch.setattr(route_policy, "ExternalSolvedDB", _Malom)


def test_loader_binds_read_only_resources_and_exact_route(monkeypatch) -> None:
    _patch_verified_dependencies(monkeypatch)
    observed = {}

    def _encode(board, player, **kwargs):
        observed.update(kwargs)
        assert player == board.turn
        return SimpleNamespace(
            feat_matrix=np.zeros((2, 134), dtype=np.float32),
            legal_moves=[{"to": "a1"}, {"to": "a4"}],
        )

    monkeypatch.setattr(route_policy, "encode_position_with_lookahead", _encode)
    loaded = route_policy.load_training_aligned_policy(
        "route-bundle",
        human_db_path="human.sqlite",
        specialist_db_path="specialist.sqlite",
        malom_path="malom",
        malom_manifest_path="malom.json",
        device="cpu",
    )

    assert loaded.choose_move(BoardState.new_game()) == {"to": "a4"}
    assert loaded.bundle_identity == "b" * 64
    assert loaded.human_db.read_only is True
    assert loaded.specialist_db.read_only is True
    assert loaded.specialist_db.trusted is True
    assert loaded.malom.strict is True
    assert observed["strict"] is True
    assert observed["sdb_min_samples"] == 3
    assert observed["specialist_db"] is loaded.specialist_db
    assert observed["lookahead_advisor"]._frozen_model is loaded.target
    assert observed["lookahead_advisor"]._human_db is loaded.human_db
    assert observed["lookahead_advisor"]._endgame_db is loaded.malom

    loaded.close()
    assert loaded.human_db.closed is True
    assert loaded.specialist_db.closed is True


def test_loader_rejects_human_db_identity_mismatch(monkeypatch) -> None:
    _patch_verified_dependencies(monkeypatch)
    monkeypatch.setattr(
        route_policy,
        "_probe_human_db",
        lambda _path: {"identity": "f" * 64},
    )

    with pytest.raises(route_policy.TrainingAlignedPolicyError, match="HumanDB"):
        route_policy.load_training_aligned_policy(
            "route-bundle",
            human_db_path="human.sqlite",
            specialist_db_path="specialist.sqlite",
            malom_path="malom",
            malom_manifest_path="malom.json",
        )


def test_loader_rejects_specialist_db_content_mismatch(monkeypatch) -> None:
    _patch_verified_dependencies(monkeypatch)
    monkeypatch.setattr(
        route_policy,
        "_probe_specialist_db",
        lambda _path: {
            "content_sha256": "f" * 64,
            "label_version": "sector-corrected-v1",
        },
    )

    with pytest.raises(
        route_policy.TrainingAlignedPolicyError,
        match="SpecialistDB",
    ):
        route_policy.load_training_aligned_policy(
            "route-bundle",
            human_db_path="human.sqlite",
            specialist_db_path="specialist.sqlite",
            malom_path="malom",
            malom_manifest_path="malom.json",
        )


def test_loader_rejects_unavailable_malom(monkeypatch) -> None:
    _patch_verified_dependencies(monkeypatch)
    monkeypatch.setattr(
        route_policy,
        "_probe_malom",
        lambda _path, _manifest_path: {
            "identity": "3" * 64,
            "available": False,
            "error": "decoder unavailable",
        },
    )

    with pytest.raises(route_policy.TrainingAlignedPolicyError, match="Malom"):
        route_policy.load_training_aligned_policy(
            "route-bundle",
            human_db_path="human.sqlite",
            specialist_db_path="specialist.sqlite",
            malom_path="malom",
            malom_manifest_path="malom.json",
        )


def test_policy_rejects_nonfinite_logits(monkeypatch) -> None:
    _patch_verified_dependencies(monkeypatch)
    loaded = route_policy.load_training_aligned_policy(
        "route-bundle",
        human_db_path="human.sqlite",
        specialist_db_path="specialist.sqlite",
        malom_path="malom",
        malom_manifest_path="malom.json",
    )
    monkeypatch.setattr(
        route_policy,
        "encode_position_with_lookahead",
        lambda *_args, **_kwargs: SimpleNamespace(
            feat_matrix=np.zeros((1, 134), dtype=np.float32),
            legal_moves=[{"to": "a1"}],
        ),
    )
    loaded.policy.policy_logits = lambda _features: torch.tensor([float("nan")])

    with pytest.raises(
        route_policy.TrainingAlignedPolicyError,
        match="non-finite",
    ):
        loaded.choose_move(BoardState.new_game())

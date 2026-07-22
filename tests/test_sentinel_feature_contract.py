from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from learned_ai.sentinel.feature_builder import FEATURE_DIM
from learned_ai.sentinel.feature_contract import (
    DB_FEATURE_SLOTS,
    db_free_numpy_mask,
    db_free_torch_mask,
)
from scripts import eval_sentinel, train_sentinel, train_sentinel2


EXPECTED_DB_FEATURE_SLOTS = tuple(range(41, 46)) + tuple(range(48, 58))


def test_db_feature_slots_match_move_feature_layout() -> None:
    assert FEATURE_DIM == 58
    assert DB_FEATURE_SLOTS == EXPECTED_DB_FEATURE_SLOTS


def test_numpy_mask_preserves_only_runtime_available_features() -> None:
    mask = db_free_numpy_mask()

    assert mask.shape == (FEATURE_DIM,)
    assert mask.dtype == np.float32
    assert np.all(mask[list(DB_FEATURE_SLOTS)] == 0.0)

    structural_slots = sorted(set(range(FEATURE_DIM)) - set(DB_FEATURE_SLOTS))
    assert np.all(mask[structural_slots] == 1.0)


def test_torch_mask_matches_numpy_contract() -> None:
    mask = db_free_torch_mask(device="cpu")

    assert mask.shape == (FEATURE_DIM,)
    assert mask.dtype == torch.float32
    assert np.array_equal(mask.cpu().numpy(), db_free_numpy_mask())


def test_trainers_build_the_required_mask_without_an_opt_in() -> None:
    expected = db_free_torch_mask(device="cpu")

    assert torch.equal(
        train_sentinel._required_db_feature_mask("cpu"),
        expected,
    )
    assert torch.equal(
        train_sentinel2._required_db_feature_mask("cpu"),
        expected,
    )


@pytest.mark.parametrize("trainer", [train_sentinel, train_sentinel2])
def test_trainers_activate_contract_when_legacy_flag_is_omitted(
    trainer,
    monkeypatch,
) -> None:
    calls = []
    expected = db_free_torch_mask(device="cpu")

    class EmptyDataset:
        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(
        trainer,
        "load_config",
        lambda path: SimpleNamespace(
            seed=42,
            epochs=1,
            checkpoint_dir="unused-checkpoints",
            log_dir="unused-logs",
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_required_db_feature_mask",
        lambda device: calls.append(device) or expected,
    )
    monkeypatch.setattr(trainer.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trainer.os.path,
        "exists",
        lambda path: path == "empty.npz",
    )
    monkeypatch.setattr(
        trainer.SentinelDataset,
        "load_from_disk",
        staticmethod(lambda path: EmptyDataset()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [trainer.__name__, "--dataset", "empty.npz"],
    )

    assert trainer.main() == 1
    assert len(calls) == 1


def test_evaluator_uses_the_shared_runtime_mask() -> None:
    assert np.array_equal(eval_sentinel._build_db_mask(), db_free_numpy_mask())

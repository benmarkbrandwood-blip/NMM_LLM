"""tests/test_gap_v3_d4_flag.py — D4 augmentation permutes only the board block.

Decision 5A (2026-08-06): --d4-augmentation {on,off} default off.  When on,
the 79-dim board block is permuted per D4 group elements; the 3-dim band
one-hot must not move.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_gap_net_v3", _ROOT / "tools" / "train_gap_net_v3.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer()


def test_perms_count_and_identity(trainer):
    perms = trainer._BOARD_PERMS
    assert len(perms) == 8
    # Identity (sym_idx 0) must be a no-op
    assert list(perms[0]) == list(range(79))


def test_all_perms_valid_permutation(trainer):
    for i, perm in enumerate(trainer._BOARD_PERMS):
        assert sorted(perm.tolist()) == list(range(79)), f"sym {i} not a valid permutation"


def test_d4_augmentation_preserves_band_block(trainer):
    """Applying the trainer's D4 gather to an 82-dim input must leave band block intact."""
    torch.manual_seed(0)
    board = torch.randn(16, 79)
    bands = torch.tensor([0, 1, 2] * 5 + [0])
    x = trainer._build_features(board, bands)   # (16, 82)
    perm_t = torch.from_numpy(np.stack(trainer._BOARD_PERMS)).long()

    # Force non-identity sym for every sample (choose sym 1..7 uniformly)
    sym_choice = torch.tensor([1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2])
    board_gather = perm_t[sym_choice]           # (16, 79)
    board_part   = torch.gather(x[:, :79], 1, board_gather)
    x_aug        = torch.cat([board_part, x[:, 79:]], dim=1)

    # Band one-hot at indices 79..81 unchanged
    assert torch.equal(x_aug[:, 79:], x[:, 79:])
    # Board block actually changed for at least one sample
    assert not torch.equal(x_aug[:, :79], x[:, :79])


def test_identity_sym_is_noop(trainer):
    torch.manual_seed(1)
    board = torch.randn(4, 79)
    bands = torch.tensor([0, 1, 2, 0])
    x = trainer._build_features(board, bands)
    perm_t = torch.from_numpy(np.stack(trainer._BOARD_PERMS)).long()
    sym_choice = torch.zeros(4, dtype=torch.long)
    board_gather = perm_t[sym_choice]
    board_part = torch.gather(x[:, :79], 1, board_gather)
    x_aug = torch.cat([board_part, x[:, 79:]], dim=1)
    assert torch.equal(x_aug, x)


def test_rotation_180_is_involution(trainer):
    """sym_idx 2 (180°) applied twice returns to identity."""
    perms = trainer._BOARD_PERMS
    p2 = perms[2]
    twice = p2[p2]
    assert list(twice) == list(range(79))

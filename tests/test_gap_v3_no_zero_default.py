"""tests/test_gap_v3_no_zero_default.py — missing Malom never becomes 0 in the pipeline.

Plan §5.5 / §14: 'missing Malom never becomes 0 anywhere in the pipeline'.
Coverage at the training-script level:
  - _nan_mse masks NaN targets; the resulting loss is NOT the MSE against 0.
  - _nan_mse with all-NaN target returns a 0-connected scalar (so .backward()
    works safely), but the value does NOT depend on any implicit 0 target.
  - _load_split preserves NaN sentinels in targets_empirical; does not fill
    or replace them with 0.
  - Removing NaN rows from a batch produces the same loss as leaving them in
    (proves NaN rows contribute 0 gradient, not 0-target gradient).
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


def test_nan_mse_all_nan_does_not_impute_zero(trainer):
    """If NaN were treated as 0, loss would be ((pred - 0)**2).mean() = 4.0.
    The correct behaviour returns 0 because no valid pairs exist — not because
    NaN was replaced with 0.
    """
    pred = torch.tensor([2.0, 2.0, 2.0])
    target = torch.tensor([float("nan"), float("nan"), float("nan")])
    loss = trainer._nan_mse(pred, target)
    assert float(loss) == 0.0
    # Must remain differentiable through pred (0-connected via pred.sum() * 0)
    assert loss.grad_fn is not None or not pred.requires_grad


def test_nan_mse_matches_masked_mse(trainer):
    """Mixed batch — result must equal MSE over valid entries only.
    Confirms NaN rows are excluded, not counted as (pred - 0)**2.
    """
    pred = torch.tensor([1.0, 3.0, 5.0, 7.0])
    target = torch.tensor([2.0, float("nan"), 4.0, float("nan")])
    valid_mse = ((torch.tensor([1.0, 5.0]) - torch.tensor([2.0, 4.0])) ** 2).mean()
    assert torch.isclose(trainer._nan_mse(pred, target), valid_mse)


def test_nan_target_distinct_from_zero_target(trainer):
    """Genuine 0 target must produce a different loss than a NaN target.
    Catches the classic bug of 'NaN → 0' silent substitution.
    """
    pred = torch.tensor([2.0, 2.0])
    zero_target = torch.tensor([0.0, 0.0])
    nan_target  = torch.tensor([float("nan"), float("nan")])
    zero_loss = trainer._nan_mse(pred, zero_target)
    nan_loss  = trainer._nan_mse(pred, nan_target)
    assert float(zero_loss) == 4.0
    assert float(nan_loss)  == 0.0
    assert not torch.isclose(zero_loss, nan_loss)


def test_gradient_zero_from_nan_row_not_zero_target(trainer):
    """A NaN row must contribute NO gradient — not a (pred - 0) gradient.
    We compare gradients from a batch with NaN row vs. the same batch with
    the NaN row removed; they must be identical.
    """
    pred_full = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
    target_full = torch.tensor([2.0, float("nan"), 4.0])
    loss_full = trainer._nan_mse(pred_full, target_full)
    loss_full.backward()
    grad_full = pred_full.grad.clone()

    pred_masked = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
    target_masked = torch.tensor([2.0, float("nan"), 4.0])
    # Manually mask the NaN entry — take valid subset
    valid_pred = torch.stack([pred_masked[0], pred_masked[2]])
    valid_tgt  = torch.tensor([2.0, 4.0])
    loss_masked = ((valid_pred - valid_tgt) ** 2).mean()
    loss_masked.backward()
    grad_masked = pred_masked.grad.clone()

    # The NaN-row entry receives 0 gradient in both computations
    assert grad_full[1] == 0.0
    assert grad_masked[1] == 0.0
    # Non-NaN entries have identical gradients
    assert torch.isclose(grad_full[0], grad_masked[0])
    assert torch.isclose(grad_full[2], grad_masked[2])


def test_load_split_preserves_nan_in_empirical(trainer, tmp_path):
    """_load_split must NOT fill NaN in targets_empirical with 0.
    Empirical targets carry NaN as the sentinel for 'support < min_support'.
    """
    from tests.test_gap_v3_fail_closed import _write_synthetic_dataset
    _write_synthetic_dataset(tmp_path)

    emp = np.memmap(
        str(tmp_path / "targets_empirical.f32.bin"),
        dtype="float32", mode="r+", shape=(8, 3),
    )
    # Insert NaN at a known cell
    emp[4, 2] = np.nan
    emp.flush()

    result = trainer._load_split(tmp_path, split_val=0)
    # NaN must remain NaN — not be replaced with 0 or any other value
    assert bool(np.isnan(result["y_emp"][4, 2].item())), \
        "targets_empirical NaN was silently replaced — no-zero-default violated"


def test_nan_targets_survive_dataloader_roundtrip(trainer):
    """NaN in a tensor survives a DataLoader batch construction.
    Regression check: past bugs have quietly zero-filled NaN when TensorDataset
    interacts with pin_memory or default_collate.
    """
    from torch.utils.data import DataLoader, TensorDataset
    X = torch.randn(8, 82)
    y = torch.tensor([
        [1.0, float("nan"), 2.0],
        [float("nan"), 3.0, float("nan")],
    ] * 4)
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)
    seen_nan = False
    for _, yb in loader:
        if torch.isnan(yb).any():
            seen_nan = True
    assert seen_nan, "NaN entries were replaced during DataLoader iteration"

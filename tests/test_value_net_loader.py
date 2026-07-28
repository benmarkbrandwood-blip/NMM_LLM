"""Regression tests for the ValueNet loader (§V1 of discussion_plan.md).

The previous eval_value_net_v2.py code path did

    net = ValueNet()
    net.load(str(net_path))

which silently discarded the return value of ValueNet.load — a
@classmethod that constructs a new instance rather than mutating self —
and evaluated a random-init model for BOTH v1 and v2.  These tests guard
against that class of bug reappearing.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ai.value_net import ValueNet


def _make_two_checkpoints(tmp_dir: Path) -> tuple[Path, Path]:
    """Emit two clearly-different ValueNet .npz files."""
    p_zero = tmp_dir / "zero.npz"
    p_one  = tmp_dir / "one.npz"

    zero_net = ValueNet()
    zero_net.W1 = np.zeros_like(zero_net.W1)
    zero_net.b1 = np.zeros_like(zero_net.b1)
    zero_net.W2 = np.zeros_like(zero_net.W2)
    zero_net.b2 = np.zeros_like(zero_net.b2)
    zero_net.W3 = np.zeros_like(zero_net.W3)
    zero_net.b3 = np.zeros_like(zero_net.b3)
    zero_net.save(p_zero)

    one_net = ValueNet()
    one_net.W1 = np.ones_like(one_net.W1)
    one_net.b1 = np.ones_like(one_net.b1)
    one_net.W2 = np.ones_like(one_net.W2)
    one_net.b2 = np.ones_like(one_net.b2)
    one_net.W3 = np.ones_like(one_net.W3)
    one_net.b3 = np.ones_like(one_net.b3)
    one_net.save(p_one)

    return p_zero, p_one


def test_load_return_value_is_used():
    """§V1 — ValueNet.load returns the new instance; ignoring it evaluates
    a random-init model."""
    with tempfile.TemporaryDirectory() as td:
        p_zero, p_one = _make_two_checkpoints(Path(td))

        # Correct usage — assign the return value.
        net_zero = ValueNet.load(str(p_zero))
        net_one  = ValueNet.load(str(p_one))

        # Zero-weight net returns tanh(bias) = tanh(0) = 0 for any input.
        X = np.random.rand(4, ValueNet.INPUT_DIM if hasattr(ValueNet, "INPUT_DIM") else net_zero.W1.shape[1]).astype(np.float32)
        pred_zero = net_zero.predict_batch(X).ravel()
        pred_one  = net_one.predict_batch(X).ravel()

        # Distinctly different predictions.
        assert not np.allclose(pred_zero, pred_one), (
            "Two obviously-different checkpoints produced the same predictions — "
            "the loader is not being respected (§V1 bug pattern)."
        )


def test_load_wrong_pattern_returns_random_init():
    """Explicit demonstration of the bug: assigning ValueNet() and then
    calling .load(...) as an instance method does NOT mutate the assigned
    object.  This test would fail if someone ever "fixed" load to mutate.
    """
    with tempfile.TemporaryDirectory() as td:
        p_zero, _ = _make_two_checkpoints(Path(td))
        net = ValueNet()
        result = net.load(str(p_zero))    # ← returns a new object; net not mutated
        assert result is not net, (
            "ValueNet.load should return a fresh object, not mutate self.  "
            "If this changed, update eval_value_net_v2.py to match."
        )

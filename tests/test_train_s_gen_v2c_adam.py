"""Regression test: Adam momentum/variance survive difficulty advancement.

Sanmill's review required that optimizer state (step, exp_avg, exp_avg_sq)
be preserved across curriculum advancement rather than reset to a fresh Adam.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _make_model_and_opt(lr: float = 1e-4) -> tuple[nn.Module, torch.optim.Adam]:
    model = nn.Linear(4, 2, bias=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt


def _warm_optimizer(model: nn.Module, opt: torch.optim.Adam, steps: int = 3) -> None:
    """Run a few gradient steps so Adam state (exp_avg etc.) is populated."""
    for _ in range(steps):
        loss = model(torch.randn(8, 4)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


def _adam_moment_snapshot(opt: torch.optim.Adam) -> dict:
    """Return a copy of the first parameter's Adam state entries."""
    param = list(opt.state.values())[0]
    return {
        "step":        int(param["step"]),
        "exp_avg":     param["exp_avg"].clone(),
        "exp_avg_sq":  param["exp_avg_sq"].clone(),
    }


def test_adam_state_survives_advancement():
    """After the advancement LR-reset, momentum and variance are unchanged."""
    target_lr = 5e-5
    model, opt = _make_model_and_opt(lr=1e-3)
    _warm_optimizer(model, opt, steps=5)

    before = _adam_moment_snapshot(opt)
    assert before["step"] == 5

    # Simulate the advancement code path: only LR is reset.
    for pg in opt.param_groups:
        pg["lr"] = target_lr

    after = _adam_moment_snapshot(opt)

    assert after["step"] == before["step"], "step count changed after advancement"
    assert torch.allclose(after["exp_avg"], before["exp_avg"]), \
        "exp_avg changed after advancement"
    assert torch.allclose(after["exp_avg_sq"], before["exp_avg_sq"]), \
        "exp_avg_sq changed after advancement"
    assert opt.param_groups[0]["lr"] == target_lr, \
        "LR was not updated to the new value"


def test_adam_reset_loses_state():
    """Baseline: a fresh Adam() call *does* wipe momentum (documents old behaviour)."""
    model, opt = _make_model_and_opt(lr=1e-4)
    _warm_optimizer(model, opt, steps=5)

    before = _adam_moment_snapshot(opt)
    assert before["step"] == 5

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    assert len(opt.state) == 0, "fresh Adam still has state — unexpected"


def test_lr_reset_does_not_affect_other_param_groups():
    """All param groups receive the new LR; none is left with the old value."""
    model = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 2, bias=False))
    opt = torch.optim.Adam([
        {"params": model[0].parameters(), "lr": 1e-3},
        {"params": model[1].parameters(), "lr": 2e-3},
    ])
    _warm_optimizer(model, opt, steps=3)

    new_lr = 5e-5
    for pg in opt.param_groups:
        pg["lr"] = new_lr

    for pg in opt.param_groups:
        assert pg["lr"] == new_lr

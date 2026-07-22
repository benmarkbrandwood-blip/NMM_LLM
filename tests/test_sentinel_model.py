"""Tests for the current move-quality SentinelNet and its loss."""

from __future__ import annotations

import torch

from learned_ai.sentinel.feature_builder import FEATURE_DIM
from learned_ai.sentinel.model import SentinelNet, sentinel_loss


def test_forward_returns_one_quality_per_candidate():
    net = SentinelNet(
        input_dim=FEATURE_DIM,
        hidden_dims=[64, 32],
        dropout=0.0,
    )
    output = net(torch.randn(8, FEATURE_DIM))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (8,)


def test_forward_single_vector_keeps_batch_dimension():
    net = SentinelNet(
        input_dim=FEATURE_DIM,
        hidden_dims=[64, 32],
        dropout=0.0,
    )
    output = net(torch.randn(FEATURE_DIM))

    assert output.shape == (1,)


def test_quality_output_is_bounded():
    net = SentinelNet(
        input_dim=FEATURE_DIM,
        hidden_dims=[64, 32],
        dropout=0.0,
    )
    output = net(torch.randn(32, FEATURE_DIM) * 5.0)

    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_default_architecture_builds():
    net = SentinelNet()
    output = net(torch.randn(4, FEATURE_DIM))

    assert net.hidden_dims == [128, 64, 32]
    assert output.shape == (4,)


def test_weighted_quality_loss_backpropagates():
    net = SentinelNet(
        input_dim=FEATURE_DIM,
        hidden_dims=[64, 32],
        dropout=0.0,
    )
    output = net(torch.randn(16, FEATURE_DIM))
    targets = torch.rand(16)
    weights = torch.rand(16) + 0.1

    losses = sentinel_loss(output, targets, sample_weight=weights)

    assert set(losses) == {"total", "bce"}
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["bce"])
    losses["total"].backward()
    assert any(
        parameter.grad is not None
        for parameter in net.trunk.parameters()
    )


def test_unweighted_quality_loss_is_finite():
    net = SentinelNet(
        input_dim=FEATURE_DIM,
        hidden_dims=[32],
        dropout=0.0,
    )
    output = net(torch.randn(5, FEATURE_DIM))
    losses = sentinel_loss(output, torch.rand(5))

    assert torch.isfinite(losses["total"])
    assert losses["total"].shape == ()


def test_auxiliary_wdl_head_shapes_and_loss():
    net = SentinelNet(
        input_dim=FEATURE_DIM,
        hidden_dims=[32],
        dropout=0.0,
        aux_wdl=True,
    )
    quality, wdl_logits = net(
        torch.randn(6, FEATURE_DIM),
        return_aux=True,
    )

    assert quality.shape == (6,)
    assert wdl_logits.shape == (6, 3)

    losses = sentinel_loss(
        quality,
        torch.rand(6),
        wdl_logits=wdl_logits,
        wdl_targets=torch.tensor([0, 1, 2, -1, 1, 0]),
    )
    assert set(losses) == {"total", "bce", "wdl"}
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["wdl"])


def test_auxiliary_wdl_loss_ignores_fully_masked_targets():
    net = SentinelNet(
        input_dim=FEATURE_DIM,
        hidden_dims=[32],
        dropout=0.0,
        aux_wdl=True,
    )
    quality, wdl_logits = net(
        torch.randn(3, FEATURE_DIM),
        return_aux=True,
    )

    losses = sentinel_loss(
        quality,
        torch.rand(3),
        wdl_logits=wdl_logits,
        wdl_targets=torch.full((3,), -1, dtype=torch.long),
    )

    assert set(losses) == {"total", "bce"}
    assert torch.equal(losses["total"].detach(), losses["bce"])

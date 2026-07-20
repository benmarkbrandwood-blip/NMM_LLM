"""Tests for stable serial game identities and RNG substreams."""

from __future__ import annotations

import torch

from scripts import train_s_gen_v2 as trainer


def test_game_identity_is_stable_and_role_sensitive() -> None:
    first = trainer._derive_game_identity(42, 7, "primary")

    assert trainer._derive_game_identity(42, 7, "primary") == first
    assert trainer._derive_game_identity(42, 8, "primary") != first
    assert trainer._derive_game_identity(42, 7, "retry") != first
    assert first[0].startswith("game:")


def test_game_torch_substream_is_independent_of_global_rng() -> None:
    _, seed = trainer._derive_game_identity(42, 7, "primary")
    first = torch.rand(8, generator=trainer._game_torch_generator(seed))
    torch.manual_seed(999)
    torch.rand(100)

    repeated = torch.rand(8, generator=trainer._game_torch_generator(seed))

    assert torch.equal(first, repeated)

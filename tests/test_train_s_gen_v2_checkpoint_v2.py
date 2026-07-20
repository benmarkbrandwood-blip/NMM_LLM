"""Tests for Generalist v2 CheckpointEnvelope integration."""

from __future__ import annotations

import copy
import random
from collections import deque
from pathlib import Path

import pytest
import torch

from learned_ai.agents.specialist_router import _load_spec_model
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.checkpoint_envelope import (
    CheckpointDescriptor,
    save_checkpoint,
)
from scripts import train_s_gen_v2 as trainer


def _model() -> ScaffoldedPolicyNet:
    return ScaffoldedPolicyNet(
        move_feat_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        policy_hidden=(8,),
    )


def _payload(model: ScaffoldedPolicyNet, *, game_count: int = 7):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return trainer._make_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        game_rng=random.Random(42),
        game_count=game_count,
        batch_count=game_count,
        update_count=2,
        difficulty=3,
        temperature=0.7,
        win_history=deque([1.0, 0.5], maxlen=40),
        win_history_heuristic=deque([1.0], maxlen=40),
        diag_buffer=[],
        games_at_level=4,
        best_win_rate=0.6,
        best_win_rate_at_diff=0.5,
        branch_bucket_history=deque(["opening"], maxlen=300),
        frozen_model=copy.deepcopy(model),
        games_since_target_update=3,
        recovery_grace=0,
        pending_steps=[],
        last_update_losses=(0.1, 0.2, 0.3),
        source_checkpoint="scratch",
        checkpoint_sequence=1,
        specialist_db_identity={"sha256": "specialist-identity"},
    )


def _descriptor() -> CheckpointDescriptor:
    return CheckpointDescriptor(
        checkpoint_id="run-001:checkpoint:00000001",
        run_id="run-001",
        experiment_id="dev-v4-corrected",
        parent_checkpoint_id=None,
        role="latest",
        save_reason="test",
        created_at_utc="2026-07-20T11:00:00Z",
        config_sha256="a" * 64,
        feature_schema_version=trainer.FEATURE_SCHEMA_VERSION,
        label_schema_version=trainer.LABEL_SCHEMA_VERSION,
        database_schema_versions={"specialist_db": "sector-corrected-v1"},
        asset_identities={"malom": "malom-identity"},
        implementation={"trainer": trainer.STAGE_TAG, "framework": "pytorch"},
    )


def test_generalist_payload_captures_mutable_training_state() -> None:
    model = _model()
    payload = _payload(model)

    assert payload.trainer_state["game_count"] == 7
    assert payload.trainer_state["rolling_metrics"]["best_win_rate"] == 0.6
    assert payload.trainer_state["target_network"]["games_since_update"] == 3
    assert payload.trainer_state["model_config"] == model.get_config()
    assert payload.data_state["cursor"] == {"completed_games": 7}
    assert payload.data_state["mutable_assets"]["specialist_db"] == {
        "sha256": "specialist-identity"
    }
    assert set(payload.rng_state["components"]) == {"game"}


def test_generalist_model_loader_reads_v2_state_and_counters(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "latest.pt"
    save_checkpoint(path, _descriptor(), _payload(model))

    loaded, start_game, best_win_rate, difficulty, source = trainer._load_model(
        torch.device("cpu"), path, (8,)
    )

    assert start_game == 7
    assert best_win_rate == 0.6
    assert difficulty == 3
    assert source == str(path)
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)


def test_weights_only_model_import_resets_all_training_counters(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "latest.pt"
    save_checkpoint(path, _descriptor(), _payload(model))

    _, start_game, best_win_rate, difficulty, source = trainer._load_model(
        torch.device("cpu"), path, (8,), start_mode="weights-only"
    )

    assert start_game == 0
    assert best_win_rate == 0.0
    assert difficulty == trainer.DIFF_START
    assert source == str(path)


def test_exact_resume_restores_complete_mutable_state() -> None:
    model = _model()
    payload = _payload(model, game_count=11)
    payload.trainer_state["batch_count"] = 6
    payload.trainer_state["update_count"] = 4
    payload.trainer_state["temperature"] = 0.42
    payload.trainer_state["curriculum"]["games_at_level"] = 9
    payload.trainer_state["target_network"]["games_since_update"] = 5
    payload.trainer_state["recovery_state"]["grace"] = 2
    payload.trainer_state["recovery_state"]["last_update_losses"] = (
        0.4,
        0.5,
        0.6,
    )
    payload.data_state["buckets"]["branch_history"] = ["midgame", "endgame"]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    game_rng = random.Random(99)
    frozen = copy.deepcopy(model)
    for parameter in frozen.parameters():
        parameter.data.zero_()
    random.seed(999)
    torch.manual_seed(999)

    restored = trainer._restore_exact_resume_payload(
        payload,
        optimizer=optimizer,
        game_rng=game_rng,
        frozen_model=frozen,
        rolling_win=40,
        bucket_window=300,
    )

    assert restored["game_count"] == 11
    assert restored["batch_count"] == 6
    assert restored["update_count"] == 4
    assert restored["temperature"] == 0.42
    assert restored["games_at_level"] == 9
    assert restored["games_since_target_update"] == 5
    assert restored["recovery_grace"] == 2
    assert restored["last_update_losses"] == (0.4, 0.5, 0.6)
    assert list(restored["branch_bucket_history"]) == ["midgame", "endgame"]
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert game_rng.getstate() == payload.rng_state["components"]["game"]
    assert random.getstate() == payload.rng_state["python"]
    assert torch.equal(torch.get_rng_state(), payload.rng_state["torch_cpu"])
    for name, tensor in model.state_dict().items():
        assert torch.equal(frozen.state_dict()[name], tensor)


def test_exact_resume_rejects_inconsistent_data_cursor() -> None:
    model = _model()
    payload = _payload(model, game_count=7)
    payload.data_state["cursor"]["completed_games"] = 6

    with pytest.raises(RuntimeError, match="cursor disagrees"):
        trainer._restore_exact_resume_payload(
            payload,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.5),
            game_rng=random.Random(99),
            frozen_model=copy.deepcopy(model),
            rolling_win=40,
            bucket_window=300,
        )


def test_inference_loader_reads_v2_model_config_and_weights(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "best.pt"
    save_checkpoint(path, _descriptor(), _payload(model))

    loaded, config = _load_spec_model(path)

    assert loaded is not None
    assert config == model.get_config()
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)

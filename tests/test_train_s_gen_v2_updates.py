"""Focused tests for truthful Generalist update evidence."""

from __future__ import annotations

import copy
import random

import pytest
import torch

from learned_ai.training.generalist_preflight import PreflightConfigurationError
from scripts import train_s_gen_v2 as trainer


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor)
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left.detach().cpu(), right.detach().cpu())
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict)
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in sorted(left, key=repr):
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def test_update_if_ready_does_not_claim_or_consume_small_batch() -> None:
    steps = [object()] * (trainer.MIN_UPDATE_STEPS - 1)

    def unexpected_update(*args, **kwargs):
        raise AssertionError("small batch must not call the optimizer")

    result = trainer._update_if_ready(
        update_fn=unexpected_update,
        model=object(),
        optimizer=object(),
        steps=steps,
        device=torch.device("cpu"),
        gamma=0.99,
        entropy_coef=0.01,
    )

    assert result is None
    assert len(steps) == trainer.MIN_UPDATE_STEPS - 1


def test_update_if_ready_reports_a_real_minimum_batch() -> None:
    steps = [object()] * trainer.MIN_UPDATE_STEPS
    calls = []

    def update(*args, **kwargs):
        calls.append((args, kwargs))
        return 1.0, 2.0, 3.0

    result = trainer._update_if_ready(
        update_fn=update,
        model=object(),
        optimizer=object(),
        steps=steps,
        device=torch.device("cpu"),
        gamma=0.99,
        entropy_coef=0.01,
    )

    assert result == (1.0, 2.0, 3.0)
    assert len(calls) == 1


def test_exact_transition_queue_consumes_64_and_retains_six() -> None:
    original = [bytes([index]) for index in range(70)]
    pending = list(original)

    batch = trainer._take_exact_transition_batch(pending, batch_size=64)

    assert batch == original[:64]
    assert pending == original[64:]
    assert pending[0] is original[64]


def test_exact_transition_queue_consumes_two_ordered_batches() -> None:
    original = list(range(140))
    pending = list(original)

    first = trainer._take_exact_transition_batch(pending, batch_size=64)
    second = trainer._take_exact_transition_batch(pending, batch_size=64)
    incomplete = trainer._take_exact_transition_batch(pending, batch_size=64)

    assert first == original[:64]
    assert second == original[64:128]
    assert incomplete is None
    assert pending == original[128:]


def test_exact_transition_queue_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        trainer._take_exact_transition_batch([], batch_size=0)


def test_exact_transition_mode_never_runs_final_undersized_flush() -> None:
    assert not trainer._should_run_final_transition_flush(
        pending_count=63,
        exact_transition_batches=True,
        optimizer_update_bound=None,
        update_count=8,
    )
    assert trainer._should_run_final_transition_flush(
        pending_count=63,
        exact_transition_batches=False,
        optimizer_update_bound=None,
        update_count=8,
    )
    assert not trainer._should_run_final_transition_flush(
        pending_count=63,
        exact_transition_batches=False,
        optimizer_update_bound=8,
        update_count=8,
    )


def test_configuration_rejects_update_cadence_below_real_batch(tmp_path) -> None:
    args = trainer._build_argument_parser().parse_args(
        [
            "--preflight",
            "smoke",
            "--out-dir",
            str(tmp_path / "out"),
            "--update-every",
            str(trainer.MIN_UPDATE_STEPS - 1),
        ]
    )

    with pytest.raises(PreflightConfigurationError, match="update_every must be at least"):
        trainer.validate_generalist_configuration(args)


def test_exact_transition_mode_requires_isolated_optimizer_path(tmp_path) -> None:
    args = trainer._build_argument_parser().parse_args(
        [
            "--preflight",
            "smoke",
            "--out-dir",
            str(tmp_path / "out"),
            "--exact-transition-batches",
        ]
    )

    with pytest.raises(
        PreflightConfigurationError,
        match="exact_transition_batches requires",
    ):
        trainer.validate_generalist_configuration(args)


def test_exact_transition_mode_accepts_frozen_diagnostic_controls(tmp_path) -> None:
    args = trainer._build_argument_parser().parse_args(
        [
            "--preflight",
            "smoke",
            "--out-dir",
            str(tmp_path / "out"),
            "--exact-transition-batches",
            "--no-s1a-warmstart",
            "--no-imitation-mix",
            "--no-s1b-refresher",
            "--no-recovery",
            "--no-opening-forcing",
            "--max-branches-per-game",
            "0",
        ]
    )

    trainer.validate_generalist_configuration(args)


def test_no_refresh_fork_preserves_target_and_age() -> None:
    state = trainer._new_target_refresh_fork_state(50)
    state["captured"] = True
    refresh_calls: list[bool] = []

    treated, age, applied = trainer._apply_target_refresh_fork_treatment(
        state=state,
        treatment="no-refresh",
        optimizer_consumed_transition_count=640,
        games_since_target_update=50,
        refresh_target=lambda: refresh_calls.append(True),
    )

    assert applied is True
    assert age == 50
    assert refresh_calls == []
    assert treated["treatment"] == "no-refresh"
    assert treated["post_fork_transition_origin"] == 640


def test_refresh_once_fork_changes_only_target_age_and_fork_identity() -> None:
    torch.manual_seed(20260810)
    state = trainer._new_target_refresh_fork_state(50)
    state["captured"] = True
    candidate = torch.nn.Linear(3, 2)
    target = copy.deepcopy(candidate)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.add_(1.0)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-4)
    pending = list(range(17))
    before = {
        "candidate": copy.deepcopy(candidate.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state().clone(),
        "pending": list(pending),
    }
    target_before = copy.deepcopy(target.state_dict())

    treated, age, applied = trainer._apply_target_refresh_fork_treatment(
        state=state,
        treatment="refresh-once",
        optimizer_consumed_transition_count=640,
        games_since_target_update=50,
        refresh_target=lambda: target.load_state_dict(candidate.state_dict()),
    )

    assert applied is True
    assert age == 0
    _assert_nested_equal(target.state_dict(), candidate.state_dict())
    with pytest.raises(AssertionError):
        _assert_nested_equal(target.state_dict(), target_before)
    after = {
        "candidate": candidate.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "pending": pending,
    }
    _assert_nested_equal(after, before)
    assert treated == {
        "schema_version": trainer.TARGET_REFRESH_FORK_STATE_SCHEMA,
        "fork_game": 50,
        "captured": True,
        "treatment": "refresh-once",
        "post_fork_transition_origin": 640,
    }


def test_refresh_once_fork_is_idempotent_on_exact_resume() -> None:
    state = trainer._new_target_refresh_fork_state(50)
    state.update(
        captured=True,
        treatment="refresh-once",
        post_fork_transition_origin=640,
    )
    refresh_calls: list[bool] = []

    treated, age, applied = trainer._apply_target_refresh_fork_treatment(
        state=state,
        treatment="refresh-once",
        optimizer_consumed_transition_count=1152,
        games_since_target_update=8,
        refresh_target=lambda: refresh_calls.append(True),
    )

    assert treated == state
    assert age == 8
    assert applied is False
    assert refresh_calls == []


def test_duplicate_no_refresh_forks_match_after_512_consumed_transitions() -> None:
    torch.manual_seed(94731)
    random.seed(94731)
    source_model = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-4)
    source_model_state = copy.deepcopy(source_model.state_dict())
    source_optimizer_state = copy.deepcopy(source_optimizer.state_dict())
    source_python_rng = random.getstate()
    source_torch_rng = torch.get_rng_state().clone()
    source_pending = [
        (float(index % 17) / 16.0, float((index * 7) % 19) / 18.0)
        for index in range(523)
    ]

    def run_control():
        random.setstate(source_python_rng)
        torch.set_rng_state(source_torch_rng.clone())
        model = torch.nn.Linear(2, 1)
        model.load_state_dict(copy.deepcopy(source_model_state))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(copy.deepcopy(source_optimizer_state))
        target = copy.deepcopy(model)
        pending = list(source_pending)
        fork_state = trainer._new_target_refresh_fork_state(50)
        fork_state["captured"] = True
        fork_state, target_age, applied = (
            trainer._apply_target_refresh_fork_treatment(
                state=fork_state,
                treatment="no-refresh",
                optimizer_consumed_transition_count=0,
                games_since_target_update=50,
                refresh_target=lambda: target.load_state_dict(model.state_dict()),
            )
        )
        assert applied is True
        logs = []
        consumed = 0
        for update_index in range(8):
            batch = trainer._take_exact_transition_batch(
                pending,
                batch_size=64,
            )
            assert batch is not None
            features = torch.tensor(batch, dtype=torch.float32)
            target_values = (features[:, :1] * 0.25) - (features[:, 1:] * 0.5)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(features), target_values)
            loss.backward()
            optimizer.step()
            consumed += len(batch)
            logs.append(
                {
                    "update": update_index + 1,
                    "consumed": consumed,
                    "loss": float(loss.detach()),
                    "python_probe": random.random(),
                    "torch_probe": float(torch.rand(())),
                }
            )
        checkpoint_state = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "target": copy.deepcopy(target.state_dict()),
            "target_age": target_age,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state().clone(),
            "pending": list(pending),
            "optimizer_consumed_transition_count": consumed,
            "fork_state": fork_state,
            "logs": logs,
        }
        assert consumed == 512
        assert len(pending) == 11
        return checkpoint_state

    first = run_control()
    second = run_control()

    _assert_nested_equal(first, second)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            ["--target-refresh-fork-game", "50"],
            "game and treatment must be specified together",
        ),
        (
            [
                "--target-refresh-fork-game",
                "50",
                "--target-refresh-fork-treatment",
                "capture",
            ],
            "requires exact_transition_batches",
        ),
    ],
)
def test_configuration_rejects_incomplete_target_refresh_fork(
    tmp_path, extra, message
) -> None:
    args = trainer._build_argument_parser().parse_args(
        [
            "--preflight",
            "smoke",
            "--out-dir",
            str(tmp_path / "out"),
            *extra,
        ]
    )

    with pytest.raises(PreflightConfigurationError, match=message):
        trainer.validate_generalist_configuration(args)


def _equal_transition_fork_args(tmp_path, *, treatment: str):
    command = [
        "--preflight",
        "long-run",
        "--out-dir",
        str(tmp_path / treatment),
        "--experiment-id",
        "sanmill-target-refresh-equal-transition-v1",
        "--max-games",
        "5000",
        "--self-play-ratio",
        "0.60",
        "--update-target-every",
        "50",
        "--max-ply",
        "120",
        "--max-ply-branch",
        "120",
        "--max-branches-per-game",
        "0",
        "--sim-ply-depth",
        "5",
        "--minimal-rollouts",
        "--no-recovery",
        "--no-sentinel",
        "--no-value-net",
        "--no-gap-net",
        "--no-s1a-warmstart",
        "--no-imitation-mix",
        "--no-s1b-refresher",
        "--no-opening-forcing",
        "--referee-engine",
        "sanmill",
        "--opponent-engine",
        "sanmill",
        "--sanmill-runtime",
        str(tmp_path / "sanmill-runtime"),
        "--sanmill-node-ladder",
        "1000,5000,25000,100000,500000",
        "--sanmill-stage-games",
        "500,500,500,1000,2500",
        "--curriculum-advance-policy",
        "fixed-resource",
        "--diff-start",
        "1",
        "--diff-max",
        "5",
        "--specialist-read-mode",
        "theoretical-only",
        "--mill-bonus-mode",
        "malom-preserving-only",
        "--lr-adaptation-mode",
        "fixed",
        "--exact-transition-batches",
        "--target-refresh-fork-game",
        "50",
        "--target-refresh-fork-treatment",
        treatment,
    ]
    if treatment == "capture":
        command.extend(("--start-mode", "fresh"))
    else:
        command.extend(
            (
                "--start-mode",
                "exact-resume",
                "--resume",
                str(tmp_path / "target-refresh-fork.pt"),
                "--post-fork-transition-bound",
                "8192",
            )
        )
    return trainer._build_argument_parser().parse_args(command)


def test_fork_treatments_preserve_resume_semantics_but_bind_run_config(
    tmp_path,
) -> None:
    capture = _equal_transition_fork_args(tmp_path, treatment="capture")
    refresh = _equal_transition_fork_args(tmp_path, treatment="refresh-once")
    no_refresh = _equal_transition_fork_args(tmp_path, treatment="no-refresh")

    for args in (capture, refresh, no_refresh):
        trainer.validate_generalist_configuration(args)

    assert trainer.resume_config_sha256(capture) == trainer.resume_config_sha256(
        refresh
    )
    assert trainer.resume_config_sha256(capture) == trainer.resume_config_sha256(
        no_refresh
    )
    assert vars(refresh) != vars(no_refresh)


@pytest.mark.parametrize(
    "outcome",
    [trainer.WIN_REWARD, trainer.LOSS_REWARD, trainer.DRAW_SHORT, trainer.DRAW_LONG],
)
def test_minimal_rollouts_keep_every_primary_outcome(outcome) -> None:
    assert trainer._keep_primary_trajectory(
        outcome,
        minimal_rollouts=True,
        confirmed=False,
    )


def test_nonminimal_loss_still_requires_confirmation() -> None:
    assert not trainer._keep_primary_trajectory(
        trainer.LOSS_REWARD,
        minimal_rollouts=False,
        confirmed=False,
    )
    assert trainer._keep_primary_trajectory(
        trainer.LOSS_REWARD,
        minimal_rollouts=False,
        confirmed=True,
    )

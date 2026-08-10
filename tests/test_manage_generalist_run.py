"""Focused command-contract tests for managed-plan preparation."""

from __future__ import annotations

import json

import pytest

from scripts import manage_generalist_run as manager


def _required_prepare_args() -> list[str]:
    return [
        "prepare",
        "--control-dir",
        "control/new-run",
        "--max-wall-hours",
        "12",
        "--objective",
        "fresh-rules-corrected-baseline",
        "--experiment-id",
        "dev-v4-rules-corrected-v2",
        "--seed",
        "43",
        "--max-ply",
        "120",
        "--mill-bonus-mode",
        "malom-preserving-only",
    ]


@pytest.mark.parametrize(
    "omitted",
    [
        "--objective",
        "--experiment-id",
        "--seed",
        "--max-ply",
        "--mill-bonus-mode",
    ],
)
def test_prepare_requires_successor_identity_and_truncation(
    omitted: str,
) -> None:
    argv = _required_prepare_args()
    index = argv.index(omitted)
    del argv[index : index + 2]

    with pytest.raises(SystemExit):
        manager._build_parser().parse_args(argv)


def test_common_args_record_explicit_truncation_ceiling(tmp_path) -> None:
    args = manager._build_parser().parse_args(_required_prepare_args())

    common = manager._common_trainer_args(args, tmp_path / "paths.json")

    assert common[common.index("--max-ply") + 1] == "120"
    assert common[common.index("--max-ply-branch") + 1] == "120"
    assert common[common.index("--experiment-id") + 1] == (
        "dev-v4-rules-corrected-v2"
    )
    assert common[common.index("--seed") + 1] == "43"
    assert common[common.index("--mill-bonus-mode") + 1] == (
        "malom-preserving-only"
    )
    assert common[common.index("--malom-policy-aux-coef") + 1] == "0.0"


def test_common_args_bind_explicit_malom_policy_auxiliary(tmp_path) -> None:
    args = manager._build_parser().parse_args(
        _required_prepare_args()
        + ["--malom-policy-aux-coef", "0.1"]
    )

    common = manager._common_trainer_args(args, tmp_path / "paths.json")

    assert common[common.index("--malom-policy-aux-coef") + 1] == "0.1"


def test_common_args_bind_refresh_cadence_and_learning_rate_mode(tmp_path) -> None:
    args = manager._build_parser().parse_args(
        _required_prepare_args()
        + [
            "--target-refresh-every",
            "5001",
            "--lr-adaptation-mode",
            "fixed",
        ]
    )

    common = manager._common_trainer_args(args, tmp_path / "paths.json")

    assert common[common.index("--update-target-every") + 1] == "5001"
    assert common[common.index("--lr-adaptation-mode") + 1] == "fixed"


def test_common_args_bind_optimizer_and_measurement_contract(tmp_path) -> None:
    args = manager._build_parser().parse_args(
        _required_prepare_args()
        + [
            "--optimizer-update-bound",
            "34",
            "--measurement-anchor-game",
            "50",
            "--measurement-anchor-expected-update-count",
            "18",
            "--measurement-every-updates",
            "4",
            "--measurement-games-per-opponent",
            "8",
            "--measurement-sanmill-node-budget",
            "1000",
            "--measurement-temperature",
            "0.2",
            "--no-exact-resume",
        ]
    )

    common = manager._common_trainer_args(args, tmp_path / "paths.json")

    expected = {
        "--optimizer-update-bound": "34",
        "--measurement-anchor-game": "50",
        "--measurement-anchor-expected-update-count": "18",
        "--measurement-every-updates": "4",
        "--measurement-games-per-opponent": "8",
        "--measurement-sanmill-node-budget": "1000",
        "--measurement-temperature": "0.2",
    }
    for option, value in expected.items():
        assert common[common.index(option) + 1] == value
    assert args.no_exact_resume is True


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf"])
def test_prepare_rejects_invalid_malom_policy_auxiliary(value: str) -> None:
    with pytest.raises(SystemExit):
        manager._build_parser().parse_args(
            _required_prepare_args()
            + ["--malom-policy-aux-coef", value]
        )


def test_completion_bound_does_not_shorten_trainer_schedule(tmp_path) -> None:
    args = manager._build_parser().parse_args(
        _required_prepare_args()
        + [
            "--max-games",
            "5000",
            "--completion-game-bound",
            "500",
            "--segment-games",
            "500",
        ]
    )

    common = manager._common_trainer_args(args, tmp_path / "paths.json")

    assert args.completion_game_bound == 500
    assert common[common.index("--max-games") + 1] == "5000"


def test_common_args_disable_unapproved_training_inputs(tmp_path) -> None:
    args = manager._build_parser().parse_args(_required_prepare_args())

    common = manager._common_trainer_args(args, tmp_path / "paths.json")

    assert {
        "--no-sentinel",
        "--no-value-net",
        "--no-gap-net",
        "--no-s1a-warmstart",
        "--no-imitation-mix",
        "--no-s1b-refresher",
        "--no-opening-forcing",
    } <= set(common)


def test_common_args_build_sanmill_fixed_resource_profile(tmp_path) -> None:
    argv = _required_prepare_args() + [
        "--engine-profile",
        "sanmill-fixed-resource",
        "--self-play-ratio",
        "0.60",
        "--sanmill-node-ladder",
        "1000,5000,25000,100000,500000",
        "--sanmill-stage-games",
        "500,500,500,1000,2500",
    ]
    args = manager._build_parser().parse_args(argv)

    common = manager._common_trainer_args(args, tmp_path / "paths.json")

    assert common[common.index("--referee-engine") + 1] == "sanmill"
    assert common[common.index("--opponent-engine") + 1] == "sanmill"
    assert common[common.index("--curriculum-advance-policy") + 1] == (
        "fixed-resource"
    )
    assert common[common.index("--sanmill-node-ladder") + 1] == (
        "1000,5000,25000,100000,500000"
    )
    assert common[common.index("--sanmill-stage-games") + 1] == (
        "500,500,500,1000,2500"
    )
    assert common[common.index("--self-play-ratio") + 1] == "0.6"
    assert common[common.index("--diff-start") + 1] == "1"
    assert common[common.index("--diff-max") + 1] == "5"
    assert "--minimal-rollouts" in common
    assert "--no-recovery" in common
    assert "--heuristic-node-budget" not in common


def test_local_profile_keeps_explicit_game_ai_budget(tmp_path) -> None:
    args = manager._build_parser().parse_args(_required_prepare_args())

    common = manager._common_trainer_args(args, tmp_path / "paths.json")

    assert common[common.index("--heuristic-node-budget") + 1] == "500000"
    assert "--sanmill-node-ladder" not in common


def test_prepare_cli_reserves_stdout_for_one_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def noisy_prepare(_args: object) -> dict[str, object]:
        print("[s_gen_v2] Path config: machine-local.json")
        return {
            "state": "awaiting_product_authorization",
            "needs_product_decision": True,
        }

    monkeypatch.setattr(manager, "_prepare", noisy_prepare)

    assert manager.main(_required_prepare_args()) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "needs_product_decision": True,
        "state": "awaiting_product_authorization",
    }
    assert "[s_gen_v2] Path config" in captured.err

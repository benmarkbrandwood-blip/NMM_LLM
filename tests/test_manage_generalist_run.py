"""Focused command-contract tests for managed-plan preparation."""

from __future__ import annotations

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
        "--max-ply",
        "120",
        "--mill-bonus-mode",
        "malom-preserving-only",
    ]


@pytest.mark.parametrize(
    "omitted",
    ["--objective", "--experiment-id", "--max-ply", "--mill-bonus-mode"],
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
    assert common[common.index("--mill-bonus-mode") + 1] == (
        "malom-preserving-only"
    )


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

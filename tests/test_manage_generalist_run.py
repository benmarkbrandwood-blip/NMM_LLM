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
    ]


@pytest.mark.parametrize(
    "omitted",
    ["--objective", "--experiment-id", "--max-ply"],
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

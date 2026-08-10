"""Focused tests for the common-anchor target-refresh diagnostic."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

import learned_ai.validation.target_refresh_common_anchor_diagnostic as diagnostic
from learned_ai.evaluation.target_refresh_common_anchor_result import (
    TargetRefreshCommonAnchorResultError,
    decide_common_anchor_result,
    validate_paired_anchor,
)
from learned_ai.validation.target_refresh_common_anchor_diagnostic import (
    DEFAULT_CONTRACT,
    EXPECTED_CONDITIONS,
    EXPECTED_SEEDS,
    TargetRefreshCommonAnchorError,
    build_prepare_commands,
    load_target_refresh_common_anchor_contract,
    validate_prepare_commands,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / DEFAULT_CONTRACT
ATTEMPT_002_CONTRACT_PATH = (
    ROOT
    / "docs/experiments/"
    / "sanmill-target-refresh-common-anchor-diagnostic-v1-attempt-002.json"
)
ATTEMPT_003_CONTRACT_PATH = (
    ROOT
    / "docs/experiments/"
    / "sanmill-target-refresh-common-anchor-diagnostic-v1-attempt-003.json"
)


def _measurement_arm(
    seed: int,
    condition: str,
    *,
    anchor_scores: tuple[float, float, float, float],
    sanmill_scores: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> dict:
    by_checkpoint = {}
    for index, delta in enumerate((4, 8, 12, 16)):
        by_checkpoint[str(delta)] = {
            "by_opponent_source": {
                "fixed_model_anchor": {"score": anchor_scores[index]},
                "sanmill_fixed_node": {"score": sanmill_scores[index]},
            }
        }
    return {
        "arm_id": f"seed{seed}-{condition}",
        "seed": seed,
        "condition": condition,
        "metrics": {"measurement": {"by_checkpoint": by_checkpoint}},
    }


def test_decision_requires_consistent_material_update_matched_signal() -> None:
    arms = []
    for seed in EXPECTED_SEEDS:
        arms.extend(
            (
                _measurement_arm(
                    seed,
                    "refresh",
                    anchor_scores=(0.20, 0.25, 0.30, 0.35),
                    sanmill_scores=(0.0, 0.0, 0.0, 0.0),
                ),
                _measurement_arm(
                    seed,
                    "no-refresh",
                    anchor_scores=(0.35, 0.40, 0.45, 0.50),
                    sanmill_scores=(0.0, 0.0, 0.0, 0.0),
                ),
            )
        )

    decision = decide_common_anchor_result(arms)

    assert decision["classification"] == "target_refresh_mechanism_signal"
    assert decision["contrast_definition"] == "no-refresh minus refresh"
    assert decision["mean_contrast_direction"] == "positive"
    assert decision["final_contrast_direction"] == "positive"
    assert decision["material_threshold_met"] is True
    assert decision["supported"] is True


def test_decision_is_inconclusive_when_seeds_disagree() -> None:
    arms = [
        _measurement_arm(64, "refresh", anchor_scores=(0.2, 0.2, 0.2, 0.2)),
        _measurement_arm(64, "no-refresh", anchor_scores=(0.4, 0.4, 0.4, 0.4)),
        _measurement_arm(65, "refresh", anchor_scores=(0.4, 0.4, 0.4, 0.4)),
        _measurement_arm(65, "no-refresh", anchor_scores=(0.2, 0.2, 0.2, 0.2)),
    ]

    decision = decide_common_anchor_result(arms)

    assert decision["classification"] == (
        "inconclusive_seed_or_horizon_disagreement"
    )
    assert decision["supported"] is False


def _paired_arm(seed: int, condition: str, hash_value: str) -> dict:
    return {
        "seed": seed,
        "condition": condition,
        "game_rows": [
            {"game": game, "game_id": f"seed{seed}-game{game}"}
            for game in range(1, 52)
        ],
        "metrics": {
            "measurement": {
                "anchor": {"model_state_sha256": hash_value}
            }
        },
    }


def test_pairing_requires_identical_prefix_and_anchor_model() -> None:
    arms = [
        _paired_arm(seed, condition, str(seed) * 32)
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    ]

    report = validate_paired_anchor(arms)

    assert len(report) == 2
    assert all(item["first_50_games_byte_identical"] for item in report)
    assert all(item["anchor_model_state_identical"] for item in report)

    changed = copy.deepcopy(arms)
    changed[1]["game_rows"][10]["game_id"] = "different"
    with pytest.raises(
        TargetRefreshCommonAnchorResultError,
        match="before the target-refresh intervention",
    ):
        validate_paired_anchor(changed)


def test_contract_and_prepare_commands_are_frozen() -> None:
    contract = load_target_refresh_common_anchor_contract(CONTRACT_PATH)
    commands = build_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=ROOT / "data/training_paths.local.json",
        python_executable="python-under-test",
    )

    validate_prepare_commands(contract, commands)

    assert len(commands) == 4
    for arm, command in zip(contract["arms"], commands, strict=True):
        assert "authorize" not in command
        assert "run-next" not in command
        assert command[command.index("--optimizer-update-bound") + 1] == str(
            arm["optimizer_update_bound"]
        )
        assert command[
            command.index("--measurement-anchor-expected-update-count") + 1
        ] == str(arm["anchor_expected_update_count"])
        assert "--no-exact-resume" in command


def test_attempt_002_is_a_fresh_unauthorized_successor() -> None:
    predecessor = load_target_refresh_common_anchor_contract(CONTRACT_PATH)
    successor = load_target_refresh_common_anchor_contract(
        ATTEMPT_002_CONTRACT_PATH
    )

    assert successor["plan_identity"] != predecessor["plan_identity"]
    assert successor["attempt"] == {
        "attempt_number": 2,
        "predecessor_plan_identity": predecessor["plan_identity"],
        "predecessor_readiness_identity": (
            "d6ed98beebb01d9c3482b9dcc7547656dace88df8afe5ad3a8550f7b86c24547"
        ),
        "predecessor_disposition": "fatal_stop_no_result_no_retry",
        "reason": (
            "fresh successor after the attempt-001 checkpoint-envelope role "
            "failure was fixed and regression-tested"
        ),
        "requires_new_product_authorization": True,
    }
    assert successor["authorization"]["launch_authorized"] is False

    for field in (
        "control_dir",
        "experiment_id",
        "plan_id",
        "specialist_db",
    ):
        predecessor_values = {arm[field] for arm in predecessor["arms"]}
        successor_values = {arm[field] for arm in successor["arms"]}
        assert predecessor_values.isdisjoint(successor_values)


def test_attempt_003_is_a_fresh_analysis_corrected_successor() -> None:
    predecessor = load_target_refresh_common_anchor_contract(
        ATTEMPT_002_CONTRACT_PATH
    )
    successor = load_target_refresh_common_anchor_contract(
        ATTEMPT_003_CONTRACT_PATH
    )

    assert successor["plan_identity"] != predecessor["plan_identity"]
    assert successor["attempt"] == {
        "attempt_number": 3,
        "predecessor_plan_identity": predecessor["plan_identity"],
        "predecessor_readiness_identity": (
            "bcbb625d6d903ce8257a550a2489cf302bbb0fae50558f999c1a62b08a53165c"
        ),
        "predecessor_disposition": "fatal_stop_no_result_no_retry",
        "reason": (
            "fresh successor after the attempt-002 optimizer-bounded "
            "policy-health analysis failure was fixed and regression-tested"
        ),
        "requires_new_product_authorization": True,
    }
    assert successor["authorization"]["launch_authorized"] is False
    assert successor["common_training_contract"] == predecessor[
        "common_training_contract"
    ]
    assert successor["measurement_contract"] == predecessor[
        "measurement_contract"
    ]
    assert successor["pairing"] == predecessor["pairing"]
    assert successor["resources"] == predecessor["resources"]

    for field in (
        "control_dir",
        "experiment_id",
        "plan_id",
        "specialist_db",
    ):
        predecessor_values = {arm[field] for arm in predecessor["arms"]}
        successor_values = {arm[field] for arm in successor["arms"]}
        assert predecessor_values.isdisjoint(successor_values)


def test_source_readiness_audits_requested_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path = tmp_path / "contract.json"
    paths_config = tmp_path / "paths.json"
    contract_path.write_text("{}", encoding="utf-8")
    paths_config.write_text("{}", encoding="utf-8")
    requested_report = tmp_path / "attempt-003" / "readiness.json"
    contract = {
        "plan_identity": "plan-identity",
        "resources": {},
        "claim_boundary": "development only",
    }
    observed: dict[str, Path] = {}

    monkeypatch.setattr(
        diagnostic,
        "load_target_refresh_common_anchor_contract",
        lambda path: contract,
    )
    monkeypatch.setattr(
        diagnostic,
        "_inspect_source",
        lambda root, loaded: {"published": True},
    )
    monkeypatch.setattr(
        diagnostic,
        "_inspect_source_evidence",
        lambda root, loaded: {},
    )
    monkeypatch.setattr(
        diagnostic,
        "inspect_template",
        lambda root, loaded: {},
    )
    monkeypatch.setattr(
        diagnostic,
        "inspect_runtime_identities",
        lambda root, config, loaded: {},
    )
    monkeypatch.setattr(
        diagnostic,
        "inspect_result_implementation",
        lambda root, loaded: {},
    )

    def inspect_targets(root, loaded, *, report_path):
        observed["report_path"] = report_path
        return {"absent": True}

    monkeypatch.setattr(
        diagnostic,
        "inspect_preparation_targets",
        inspect_targets,
    )
    monkeypatch.setattr(
        diagnostic,
        "build_prepare_commands",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_prepare_commands",
        lambda loaded, commands: None,
    )

    report = diagnostic.inspect_source_readiness(
        root=tmp_path,
        contract_path=contract_path,
        paths_config=paths_config,
        report_path=requested_report,
        python_executable="python-under-test",
    )

    assert observed == {"report_path": requested_report.resolve()}
    assert report["state"] == "source_ready_for_local_preparation"


def test_command_validator_rejects_hidden_target_refresh_change() -> None:
    contract = load_target_refresh_common_anchor_contract(CONTRACT_PATH)
    commands = build_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=ROOT / "data/training_paths.local.json",
        python_executable="python-under-test",
    )
    changed = copy.deepcopy(commands)
    index = changed[1].index("--target-refresh-every")
    changed[1][index + 1] = "50"

    with pytest.raises(TargetRefreshCommonAnchorError, match="command factor"):
        validate_prepare_commands(contract, changed)

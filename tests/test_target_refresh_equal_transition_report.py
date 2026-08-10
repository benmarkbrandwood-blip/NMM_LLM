"""Focused contract tests for the equal-transition result publisher."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from learned_ai.training.run_contract import canonical_sha256
from scripts import report_target_refresh_equal_transition_diagnostic as report


def _contract() -> dict:
    body = {
        "schema_version": report.CONTRACT_SCHEMA,
        "status": "designed_unlaunched_needs_publication",
        "prefixes": [
            {
                "seed": seed,
                "control_dir": f"out/prefix-{seed}",
                "experiment_id": f"prefix-{seed}",
            }
            for seed in (64, 65, 66)
        ],
        "arms": [
            {
                "seed": seed,
                "condition": condition,
                "control_dir": f"out/{seed}-{condition}",
                "experiment_id": f"arm-{seed}-{condition}",
                "resume_checkpoint": f"out/prefix-{seed}/fork.pt",
            }
            for seed in (64, 65, 66)
            for condition in ("refresh-once", "no-refresh")
        ],
        "measurement_contract": {
            "transition_boundaries": [1024, 2048, 4096, 8192],
            "fixed_phase_corpus_sha256": report.EXPECTED_CORPUS_SHA256,
            "temperatures": [1.0, 0.2],
            "training_games": 0,
            "optimizer_updates": 0,
        },
    }
    return {**body, "plan_identity": canonical_sha256(body)}


def test_result_contract_accepts_exact_cells_and_identity(tmp_path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(_contract()), encoding="utf-8")

    loaded = report._validate_contract(path)

    assert loaded["plan_identity"] == _contract()["plan_identity"]


def test_result_contract_rejects_missing_arm(tmp_path) -> None:
    contract = _contract()
    contract["arms"].pop()
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    contract["plan_identity"] = canonical_sha256(body)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(report.EqualTransitionReportError, match="arm cells differ"):
        report._validate_contract(path)


def test_result_contract_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "contract.json"
    path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")

    with pytest.raises(report.EqualTransitionReportError, match="duplicate JSON key"):
        report._strict_json(path)


def test_candidate_uses_verified_branch_copy_as_runtime_source(
    tmp_path, monkeypatch
) -> None:
    control_dir = "out/64-refresh-once"
    branch_resume = (
        tmp_path / control_dir / "initial-target-refresh-fork.pt"
    ).resolve()
    recovery = {
        "target_refresh_fork_state": {
            "captured": True,
            "fork_game": 50,
            "treatment": "refresh-once",
            "post_fork_transition_origin": 1_472,
        },
        "optimizer_consumed_transition_count": 2_496,
        "pending_steps": [],
        "source_checkpoint": str(branch_resume),
    }
    envelope = SimpleNamespace(
        descriptor=SimpleNamespace(
            role="transition_diagnostic_candidate",
            experiment_id="arm-64-refresh-once",
            checkpoint_id="candidate-1024",
            asset_identities={
                "human_db": "human",
                "malom_tablebase": "malom",
                "mif_suite_1_0": "mif",
                "sanmill_training_runtime": "sanmill",
                "training_ruleset": "rules",
            },
            implementation={
                "target_refresh_branch_kind": "target-refresh-fork-v1",
                "target_refresh_branch_source_checkpoint_id": "shared-fork",
                "target_refresh_branch_source_payload_sha256": "payload",
                "target_refresh_branch_treatment": "refresh-once",
            },
        ),
        payload=SimpleNamespace(
            trainer_state={"recovery_state": recovery, "model_config": {}},
            model_state={},
        ),
    )
    arms = {
        (64, "refresh-once"): {
            "control_dir": control_dir,
            "experiment_id": "arm-64-refresh-once",
            "resume_checkpoint": "out/prefix-64/target-refresh-fork.pt",
        },
        (64, "no-refresh"): {
            "control_dir": "out/64-no-refresh",
            "experiment_id": "arm-64-no-refresh",
            "resume_checkpoint": "out/prefix-64/target-refresh-fork.pt",
        },
    }

    def fake_load(path, *, map_location):
        condition = "no-refresh" if "no-refresh" in str(path) else "refresh-once"
        candidate = SimpleNamespace(
            descriptor=SimpleNamespace(**vars(envelope.descriptor)),
            payload=SimpleNamespace(
                trainer_state={
                    "recovery_state": {
                        **recovery,
                        "target_refresh_fork_state": {
                            **recovery["target_refresh_fork_state"],
                            "treatment": condition,
                        },
                        "source_checkpoint": str(
                            (
                                tmp_path
                                / arms[(64, condition)]["control_dir"]
                                / "initial-target-refresh-fork.pt"
                            ).resolve()
                        ),
                    },
                    "model_config": {},
                    "game_count": 100,
                    "update_count": 20,
                },
                model_state={},
            ),
        )
        candidate.descriptor.experiment_id = arms[(64, condition)][
            "experiment_id"
        ]
        candidate.descriptor.implementation = {
            **candidate.descriptor.implementation,
            "target_refresh_branch_treatment": condition,
        }
        return candidate

    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(
        report,
        "_segment",
        lambda control: tmp_path / control / "segments" / "segment-0001",
    )
    monkeypatch.setattr(report, "load_checkpoint", fake_load)
    monkeypatch.setattr(report, "_load_policy", lambda value, device: object())
    monkeypatch.setattr(report, "_sha256_file", lambda value: "checkpoint-file")
    monkeypatch.setattr(report, "_state_dict_sha256", lambda value: "model")

    _, records = report._load_candidate_pair(
        arms,
        seed=64,
        boundary=1_024,
        fork_record={"checkpoint_id": "shared-fork", "payload_sha256": "payload"},
        device=report.torch.device("cpu"),
    )

    assert set(records) == {"refresh-once", "no-refresh"}


def test_git_identity_allows_published_analysis_descendant(monkeypatch) -> None:
    values = {
        ("status", "--porcelain", "--untracked-files=no"): "",
        ("branch", "--show-current"): "dev",
        ("rev-parse", "HEAD"): "analysis-head",
        ("rev-parse", "origin/dev"): "analysis-head",
    }
    monkeypatch.setattr(
        report.subprocess,
        "check_output",
        lambda command, cwd, text: values[tuple(command[1:])],
    )
    monkeypatch.setattr(
        report.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    identity = report._git_identity("training-head")

    assert identity["head"] == "training-head"
    assert identity["analysis_head"] == "analysis-head"
    assert identity["training_source_is_ancestor"] is True

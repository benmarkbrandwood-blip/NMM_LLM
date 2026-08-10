"""Focused contract tests for the equal-transition result publisher."""

from __future__ import annotations

import json

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

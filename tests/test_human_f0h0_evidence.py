from __future__ import annotations

from pathlib import Path

from learned_ai.evaluation.human_f0h0_feasibility import (
    load_f0d0_boundary,
    load_plan,
    load_result,
    load_split,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
F0D0 = ROOT / (
    "docs/evidence/"
    "f0-d0-human-raw-reconstructability-manifest-2026-08-14.json"
)
PLAN = ROOT / "docs/experiments/f0-h0-human-feasibility-screen-v1.json"
SPLIT = ROOT / (
    "docs/experiments/f0-h0-human-player-split-membership-v1.json"
)
RESULT = ROOT / (
    "docs/evidence/f0-h0-human-feasibility-screen-manifest-2026-08-14.json"
)


def test_f0h0_split_stop_evidence_is_sealed_and_consistent() -> None:
    boundary = load_f0d0_boundary(F0D0)
    plan, plan_sha = load_plan(PLAN)
    split, split_sha = load_split(SPLIT, boundary=boundary, plan=plan)
    result, _result_sha = load_result(RESULT)

    assert split["component_count"] == 1
    assert split["counts"] == {
        "train": {
            "games": 92_226,
            "logical_plies": 4_394_220,
            "player_keys": 4_994,
        },
        "selection": {"games": 0, "logical_plies": 0, "player_keys": 0},
        "one-time-confirmation": {
            "games": 0,
            "logical_plies": 0,
            "player_keys": 0,
        },
        "final-test": {"games": 0, "logical_plies": 0, "player_keys": 0},
    }
    assert result["decision"] == "触发停止条件"
    assert result["lineage"]["plan_identity"] == plan["plan_identity"]
    assert result["lineage"]["split_identity"] == split["split_identity"]
    assert result["input_files"][1]["sha256"] == plan_sha
    assert result["input_files"][2]["sha256"] == split_sha
    assert result["input_files"][0]["sha256"] == sha256_file(F0D0)
    assert set(result["dimensions"]) == {
        "independent_support",
        "modifiable_state_reachability",
        "concentration",
        "product_effect_upper_bound",
    }
    assert result["access_audit"]["raw_human_game_files_opened"] == 0
    assert result["access_audit"]["final-test_raw_record_reads"] == 0
    assert result["access_audit"]["source_pool_2eb04f54_records_read"] == 0

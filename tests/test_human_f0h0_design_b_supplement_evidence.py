from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from learned_ai.evaluation.human_f0h0_split_retest import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / (
    "docs/experiments/f0-h0-design-b-supplement-measurement-v1.json"
)
RESULT_PATH = ROOT / (
    "docs/evidence/f0-h0-design-b-supplement-manifest-2026-08-15.json"
)
EVIDENCE_PATH = ROOT / (
    "docs/evidence/f0-h0-design-b-supplement-2026-08-15.md"
)
PLAN_IDENTITY = (
    "889ccfcc407def9b7c2b4f3058611566e1bcb541976c42ed286d449dc67d633a"
)
PLAN_FILE_SHA256 = (
    "d96cc6cc13ce3cf44f7394d364db083159423923882f159902ef41a19ddb97e3"
)
RESULT_IDENTITY = (
    "a45fbfa0c472f86f03596b0618c799c4e0fb522bcfaa9b431efc904e838301a2"
)
RESULT_FILE_SHA256 = (
    "2bb06f06f55a86a14bdb30808dd68e617995acf4d663c1a1fefa99186866d850"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_plan_and_result_are_sealed_in_preregistered_order() -> None:
    plan = _load(PLAN_PATH)
    plan_body = dict(plan)
    recorded_plan = plan_body.pop("plan_identity")
    assert hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() == PLAN_FILE_SHA256
    assert recorded_plan == PLAN_IDENTITY
    assert canonical_sha256(plan_body) == recorded_plan

    result = _load(RESULT_PATH)
    result_body = dict(result)
    recorded_result = result_body.pop("result_identity")
    assert hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest() == RESULT_FILE_SHA256
    assert recorded_result == RESULT_IDENTITY
    assert canonical_sha256(result_body) == recorded_result
    assert result["lineage"]["plan_identity"] == PLAN_IDENTITY
    assert result["lineage"]["plan_file_sha256"] == PLAN_FILE_SHA256


def test_support_and_b1_accounts_are_exact() -> None:
    result = _load(RESULT_PATH)
    march = result["support_measurement"]["2026-03-01"]
    may = result["support_measurement"]["2026-05-01"]
    assert (
        march["strong_post"]["games"],
        march["strong_post"]["player_keys"],
        march["strong_post"]["outcome_eligible_games"],
        march["strong_post"]["decisions"],
    ) == (4_577, 1_245, 1_973, 207_044)
    assert (
        may["strong_post"]["games"],
        may["strong_post"]["player_keys"],
        may["strong_post"]["outcome_eligible_games"],
        may["strong_post"]["decisions"],
    ) == (847, 322, 357, 37_353)

    b1 = result["b1_player_resplit"]
    graph = b1["graph"]
    assert graph["connected_components"] == 31
    assert graph["giant_component"]["player_keys"] == 1_178
    assert graph["giant_component"]["games"] == 4_465
    assert sum(row["player_keys"] for row in graph["all_component_rows"]) == 1_245
    assert sum(row["games"] for row in graph["all_component_rows"]) == 4_577

    for row in b1["independent_edge_cut_measurements"]:
        assert (
            row["holdout_internal_games"]
            + row["train_internal_games"]
            + row["cross_cut_discard_games"]
            == 4_577
        )
    three_way = b1["simultaneous_three_way_edge_cut_measurement"]
    assert sum(three_way["internal_games"].values()) + three_way[
        "cross_partition_discard_games"
    ] == 4_577
    assert three_way["cross_partition_discard_games"] == 737


def test_b2_reports_all_three_frozen_candidate_pairs_without_selection() -> None:
    result = _load(RESULT_PATH)
    b2 = result["b2_time_resplit"]
    expected = {
        "early-month-boundaries": (887, 386, 847),
        "equal-calendar-span": (1_686, 773, 22),
        "later-month-boundaries": (2_535, 469, 58),
    }
    for candidate_id, strong_counts in expected.items():
        candidate = b2[candidate_id]
        segments = candidate["segments"]
        assert sum(row["all_segment_games"] for row in segments.values()) == 4_577
        assert tuple(
            segments[partition]["both_players_unseen_before_segment"]["games"]
            for partition in (
                "selection",
                "one-time-confirmation",
                "final-test",
            )
        ) == strong_counts
    assert result["decision"] is None
    assert result["recommendation"] is None
    assert result["scope"]["final_split_selected"] is False
    assert result["scope"]["feasibility_decision_made"] is False


def test_ring16_comparison_includes_b_c_and_random_baseline() -> None:
    result = _load(RESULT_PATH)
    comparison = result["ring16_comparison"]
    coarse = comparison["design_b_profiles"]["design_b_coarse_march"]
    coarse_rows = {
        row["partition"]: row for row in coarse["partition_overlap"]
    }
    random_rows = {
        row["partition"]: row
        for row in comparison["random_baseline"]["overlap"]["partition_overlap"]
    }
    assert coarse_rows["train"]["decision_weighted_overlap_rate"] == pytest.approx(
        0.34319703216683045
    )
    assert coarse_rows["test"]["decision_weighted_overlap_rate"] == pytest.approx(
        0.5359585402136744
    )
    assert random_rows["random-left"][
        "decision_weighted_overlap_rate"
    ] == pytest.approx(0.3755321202197428)
    assert random_rows["random-right"][
        "decision_weighted_overlap_rate"
    ] == pytest.approx(0.5731340161175893)
    assert len(comparison["design_c_previous_result"]["partition_overlap"]) == 4
    assert comparison["interpretation"] is None


def test_access_audit_and_narrative_preserve_the_non_decision() -> None:
    result = _load(RESULT_PATH)
    assert result["raw_replay"] == {
        "raw_bytes_read": 640_865_505,
        "raw_files_opened": 80_719,
        "strict_replayed_decisions": 3_835_847,
        "strict_terminal_games": 33_268,
    }
    assert all(
        value == 0 for value in result["prohibited_operations_observed"].values()
    )
    assert result["access_audit"]["source_pool_2eb04f54_artifact_reads"] == 0
    assert result["access_audit"]["source_pool_records_consumed"] == 0

    narrative = EVIDENCE_PATH.read_text(encoding="utf-8")
    assert RESULT_IDENTITY in narrative
    assert PLAN_IDENTITY in narrative
    assert "completed_measurement_only_no_final_split_selection" in narrative
    assert "does not select or recommend a final split" in narrative
    assert "No candidate membership is frozen" in narrative

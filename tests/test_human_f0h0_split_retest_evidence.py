from __future__ import annotations

import hashlib
import json
from pathlib import Path

from learned_ai.evaluation.human_f0h0_split_retest import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / (
    "docs/evidence/"
    "f0-h0-corrected-split-feasibility-manifest-2026-08-15.json"
)
RESULT_SHA256 = (
    "eb0ed05a458b282a88b6bce12824a9744780238601609f446d7772b886dba77a"
)
RESULT_IDENTITY = (
    "cbfa6d43fa31e9644bae169e6b6d42232aa008e54921c96a46fbdddb73a95931"
)


def _result() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_corrected_result_is_sealed_and_makes_no_decision() -> None:
    raw = RESULT_PATH.read_bytes()
    result = json.loads(raw)
    body = dict(result)
    recorded = body.pop("result_identity")

    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    assert recorded == RESULT_IDENTITY
    assert canonical_sha256(body) == recorded
    assert result["status"] == (
        "completed_measurement_only_no_split_selection"
    )
    assert result["decision"] is None
    assert result["recommendation"] is None
    assert result["scope"]["f0_h0_scientific_dimensions_run"] is False
    assert result["scope"]["final_split_selected"] is False


def test_corrected_result_accounts_for_every_f0d0_unit() -> None:
    result = _result()
    assert result["graph_structure"]["player_keys"] == 4_994
    assert result["graph_structure"]["games"] == 92_226
    assert result["graph_structure"]["connected_components"] == 1
    assert result["graph_structure"]["non_giant_components"] == []
    assert sum(
        row["player_keys"]
        for row in result["community_structure"]["community_rows"]
    ) == 4_994
    assert sum(
        row["games"] for row in result["time_structure"]["weekly"]
    ) == 92_226

    design_a = result["candidate_designs"]["design_a_player_cut"]
    for row in design_a["measurements"]:
        assert (
            row["holdout_internal_games"]
            + row["train_internal_games"]
            + row["cross_cut_discard_games"]
            == 92_226
        )

    design_c = result["candidate_designs"]["design_c_decision_owner"]
    partition_counts = design_c["scale"]["partition_counts"]
    assert sum(row["player_keys"] for row in partition_counts.values()) == 4_994
    assert sum(row["decisions"] for row in partition_counts.values()) == 4_394_220
    assert (
        design_c["scale"]["same_partition_games"]
        + design_c["scale"]["cross_partition_games"]
        == 92_226
    )
    assert design_c["ring16_leakage"]["strict_replayed_games"] == 92_226
    assert design_c["ring16_leakage"]["strict_replayed_decisions"] == 4_394_220


def test_corrected_result_preserves_v1_and_prohibited_boundaries() -> None:
    result = _result()
    expected_hashes = {
        "docs/experiments/f0-h0-human-feasibility-screen-v1.json": (
            "3ec32220b220f019b3a60c8a2e1519eae9933933a28b5ccf16072b06b78e2136"
        ),
        "docs/experiments/f0-h0-human-player-split-membership-v1.json": (
            "1ef901a6776bab15b96fa4ab25273223ae2028568f619fb35f6ecd96094b26c4"
        ),
        (
            "docs/evidence/"
            "f0-h0-human-feasibility-screen-manifest-2026-08-14.json"
        ): "84226cb96e1e7775a896220b3b9cee84b48f3f0562fb68c81ad6bdf28473692e",
    }
    for relative_path, expected in expected_hashes.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == (
            expected
        )

    assert result["supersession"]["v1_status"] == (
        "superseded_by_corrected_split_design"
    )
    assert result["access_audit"]["source_pool_2eb04f54_artifact_reads"] == 0
    assert result["access_audit"]["source_pool_records_consumed"] == 0
    assert result["access_audit"]["malom_queries"] == 0
    assert all(
        count == 0 for count in result["prohibited_operations_observed"].values()
    )


def test_narrative_records_supersession_without_selecting_a_design() -> None:
    correction = (
        ROOT
        / "docs/evidence/f0-h0-v1-supersession-correction-2026-08-15.md"
    ).read_text(encoding="utf-8")
    evidence = (
        ROOT
        / "docs/evidence/f0-h0-corrected-split-feasibility-2026-08-15.md"
    ).read_text(encoding="utf-8")

    assert "superseded_by_corrected_split_design" in correction
    assert "completed_measurement_only_no_split_selection" in evidence
    assert RESULT_IDENTITY in evidence
    assert "mistakenly labelled the local date" in evidence
    assert "2026-08-14T17:54:02Z" in evidence
    assert "No design is selected" in evidence
    assert "No design is described as feasible or infeasible" in evidence

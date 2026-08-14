from __future__ import annotations

from pathlib import Path

from learned_ai.evaluation.human_f0h0_b2_freeze import (
    load_membership,
    load_plan,
    load_result,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / (
    "docs/experiments/f0-h0-design-b2-freeze-and-characterization-v1.json"
)
MEMBERSHIP = ROOT / (
    "docs/experiments/f0-h0-design-b2-frozen-membership-v1.json"
)
RESULT = ROOT / (
    "docs/evidence/"
    "f0-h0-design-b2-freeze-characterization-manifest-2026-08-15.json"
)
EVIDENCE = ROOT / (
    "docs/evidence/f0-h0-design-b2-freeze-characterization-2026-08-15.md"
)
OBJECTIVE = ROOT / (
    "docs/experiments/safe-human-trap-objective-and-measurement-v1.md"
)


def _artifacts() -> tuple[dict, dict, dict]:
    plan, _plan_sha = load_plan(PLAN)
    membership, _membership_sha = load_membership(MEMBERSHIP)
    result, _result_sha = load_result(RESULT)
    return plan, membership, result


def test_frozen_plan_membership_and_result_lineage_is_exact() -> None:
    plan, membership, result = _artifacts()

    assert plan["plan_identity"] == (
        "a4dc271d00a36394d4e5b61751f7536cf3e869cb90136fbe7bedd6016c6acb30"
    )
    assert membership["membership_identity"] == (
        "06c49903baf76ee7787af8333058e164cb54ea7a27035a1371747d6000d07b0b"
    )
    assert result["result_identity"] == (
        "183a39ab29ddfbec76a7188606b0a1297ffbdb845346a05753807f2c609b65e6"
    )
    assert result["lineage"]["plan_identity"] == plan["plan_identity"]
    assert result["lineage"]["membership_identity"] == membership[
        "membership_identity"
    ]


def test_frozen_counts_and_test_player_isolation_are_exact() -> None:
    _plan, membership, _result = _artifacts()

    assert {
        name: row["games"] for name, row in membership["partitions"].items()
    } == {
        "train": 36_949,
        "selection": 887,
        "confirmation": 386,
        "final-test": 847,
    }
    isolation = membership["test_segment_player_isolation"]
    assert isolation["player_keys"] == {
        "selection": 295,
        "confirmation": 160,
        "final-test": 322,
    }
    assert isolation["verified_disjoint"] is True
    assert set(isolation["pairwise_intersections"].values()) == {0}
    assert set(membership["pairwise_session_intersections"].values()) == {0}


def test_nonfinal_characterization_accounts_for_every_decision() -> None:
    _plan, _membership, result = _artifacts()
    rows = result["partition_characterization"]
    expected = {
        "train": (36_949, 2_216, 1_742_416, 15_135),
        "selection": (887, 295, 41_130, 399),
        "confirmation": (386, 160, 17_889, 160),
    }
    for name, (games, players, decisions, outcomes) in expected.items():
        row = rows[name]
        assert (
            row["games"],
            row["player_keys"],
            row["logical_plies"],
            row["strict_outcome_eligible_games"],
        ) == (games, players, decisions, outcomes)
        assert sum(row["decisions_by_phase"].values()) == decisions
        assert sum(row["decisions_by_actor_color"].values()) == decisions
        assert sum(row["strict_outcome_distribution"].values()) == outcomes


def test_final_test_exposes_only_membership_count_and_hash() -> None:
    _plan, membership, result = _artifacts()

    assert result["final_test"] == {
        "games": 847,
        "session_ids_identity": membership["partitions"]["final-test"][
            "session_ids_identity"
        ],
        "sealed": True,
        "content_statistics": None,
    }
    assert set(result["partition_characterization"]) == {
        "train",
        "selection",
        "confirmation",
    }
    for field in (
        "final_test_raw_game_files_opened",
        "final_test_decisions_loaded",
        "final_test_derived_features_loaded",
    ):
        assert result["access_audit"][field] == 0


def test_cost_branch_is_frozen_sample_without_published_oracle_results() -> None:
    _plan, membership, result = _artifacts()
    benchmark = result["malom_cost_benchmark"]

    assert benchmark["sample"]["decision_references"] == 256
    assert benchmark["passes"][0]["queries"] == 3_497
    assert benchmark["passes"][1]["queries"] == 3_497
    assert benchmark["passes"][0]["sector_cache_entries_added"] == 79
    assert benchmark["passes"][1]["sector_cache_entries_added"] == 0
    assert benchmark["projection"]["decision"] == "sample"
    assert benchmark["projection"]["selected_game_sessions"] == 10_000
    assert benchmark["projection"]["selected_game_sessions_identity"] == (
        membership["malom_cost_preregistration"][
            "fallback_game_session_ids_identity"
        ]
    )
    assert benchmark["published_oracle_outcomes"] == 0
    assert benchmark["published_safe_sets"] == 0


def test_scope_and_protected_access_remain_closed() -> None:
    _plan, _membership, result = _artifacts()
    assert result["scope"] == {
        "official_split_frozen": True,
        "nonfinal_characterization_only": True,
        "independent_support_computed": False,
        "modifiable_state_reachability_computed": False,
        "concentration_computed": False,
        "product_effect_upper_bound_computed": False,
        "models_loaded": 0,
        "games_started": 0,
        "search_batches_started": 0,
        "training_updates": 0,
    }
    for field in (
        "human_db_reads",
        "database_writes",
        "source_pool_2eb04f54_reads",
        "source_pool_2eb04f54_records_consumed",
    ):
        assert result["access_audit"][field] == 0


def test_narrative_records_binding_state_novelty_rule_and_identities() -> None:
    evidence = EVIDENCE.read_text(encoding="utf-8")
    objective = OBJECTIVE.read_text(encoding="utf-8")

    for text in (evidence, objective):
        assert "34.32%" in text
        assert "53.60%" in text
        assert "37.55%" in text
        assert "57.31%" in text
        assert (
            "06c49903baf76ee7787af8333058e164cb54ea7a27035a1371747d6000d07b0b"
            in text
        )
    assert "must not require that a canonical or `ring16` state was never" in (
        objective
    )

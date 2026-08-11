"""Checks for the published schedule-isolation experiment contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.target_refresh_equal_transition_diagnostic import (
    SCHEDULE_ISOLATION_CONTRACT_SCHEMA,
    load_equal_transition_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/experiments/"
    "sanmill-target-refresh-schedule-isolation-diagnostic-v2.json"
)
EXPECTED_PLAN_IDENTITY = (
    "0580389b3d696df9859ac9e7aea6c4b478bf6e791b7e27bf780d2a6e02db5b0b"
)
EXPECTED_DELEGATED_SCOPE_IDENTITY = (
    "a92e87bebe87e1a287be37c95c0974cafde662703ee05436a2c30b7d9584211a"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_schedule_isolation_contract_is_frozen() -> None:
    contract = load_equal_transition_contract(CONTRACT)

    assert contract["schema_version"] == SCHEDULE_ISOLATION_CONTRACT_SCHEMA
    assert contract["plan_identity"] == EXPECTED_PLAN_IDENTITY
    assert contract["pairing"]["seeds"] == [67, 68, 69]
    assert contract["common_training_contract"]["sanmill_node_ladder"] == [1000]
    assert (
        contract["common_training_contract"]["temperature_schedule_axis"]
        == "post-fork-transitions"
    )
    assert contract["authorization"] == {
        "arm_segments_authorized": 0,
        "launch_authorized": False,
        "prefix_segments_authorized": 0,
        "promotion_allowed": False,
        "publication_allowed": False,
    }
    main_review = contract["lineage"]["main_review"]
    assert main_review["reviewed_tip"] == (
        "028ef8e7b38a04cc076e2d714854fcc13177d0df"
    )
    assert main_review["cherry_picks_selected"] == []
    evidence = main_review["evidence"]
    assert _sha256(ROOT / evidence["path"]) == evidence["sha256"]


def test_published_schedule_isolation_implementation_hashes_match() -> None:
    contract = load_equal_transition_contract(CONTRACT)
    implementation = contract["analysis"]["result_implementation"]

    for component in ("module", "publisher"):
        record = implementation[component]
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_product_delegation_scope_is_unchanged_by_lineage_refresh() -> None:
    contract = load_equal_transition_contract(CONTRACT)
    delegated_scope = {
        key: value
        for key, value in contract.items()
        if key not in {"plan_identity", "lineage"}
    }

    assert canonical_sha256(delegated_scope) == EXPECTED_DELEGATED_SCOPE_IDENTITY


def test_published_schedule_isolation_outcome_grid_is_non_training() -> None:
    contract = load_equal_transition_contract(CONTRACT)
    outcome = contract["measurement_contract"]["outcome_measurement"]

    assert outcome["total_games"] == 288
    assert outcome["transition_boundaries"] == [4096, 8192]
    assert outcome["held_out"] is False
    assert outcome["optimizer_updates"] == 0
    assert outcome["training_games"] == 0
    assert outcome["writes_training_data"] is False

from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.evaluation.sanmill_data_query import SanmillDataQuerySession
from learned_ai.evaluation.sanmill_uci import (
    PINNED_SANMILL_COMMIT,
    PINNED_SANMILL_TREE,
    PREFIX12_REPLAY_INSTALLATION_CONTRACT,
    inspect_sanmill_installation,
)
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATHS = ROOT / "data" / "training_paths.local.json"
RUNTIME_DECISION = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-prefix12-human-replay-runtime-2026-08-01.json"
)


def _has_local_runtime() -> bool:
    if not LOCAL_PATHS.is_file():
        return False
    try:
        payload = json.loads(LOCAL_PATHS.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(payload.get("sanmill_prefix12_checkout"))


def test_prefix12_runtime_contract_is_explicit_and_distinct() -> None:
    contract = PREFIX12_REPLAY_INSTALLATION_CONTRACT

    assert contract.contract_id == "sanmill-prefix12-human-replay-v1"
    assert contract.path_lookup_key == "sanmill_prefix12_checkout"
    assert contract.expected_binary_sha256 == (
        "6502f7a2180769666c1ba6c801288a5ba079920e2bd6c1121f0e8b0c27e11e53"
    )
    assert contract.expected_binary_size == 4_109_312
    assert contract.require_exact_head


def test_prefix12_runtime_decision_binds_the_code_contract() -> None:
    payload = json.loads(RUNTIME_DECISION.read_text(encoding="utf-8"))
    identity = payload.pop("runtime_identity")
    contract = PREFIX12_REPLAY_INSTALLATION_CONTRACT

    assert identity == canonical_sha256(payload)
    assert payload["status"] == "pinned_for_source_only_human_history_replay"
    assert payload["source"]["commit"] == PINNED_SANMILL_COMMIT
    assert payload["source"]["tree"] == PINNED_SANMILL_TREE
    assert payload["source"]["path_lookup_key"] == contract.path_lookup_key
    assert payload["binary"]["sha256"] == contract.expected_binary_sha256
    assert payload["binary"]["byte_length"] == contract.expected_binary_size
    assert not payload["authorization"]["evaluation"]
    assert not payload["authorization"]["training"]


@pytest.mark.skipif(
    not _has_local_runtime(),
    reason="requires the ignored sanmill_prefix12_checkout registry entry",
)
def test_local_prefix12_runtime_replays_history_in_fresh_processes() -> None:
    installation = inspect_sanmill_installation(
        LOCAL_PATHS,
        contract=PREFIX12_REPLAY_INSTALLATION_CONTRACT,
    )
    actions = ("d6", "d2", "f4", "b4", "f6", "f2", "b6", "xf2")
    records = []

    for process_index in range(2):
        with SanmillDataQuerySession(installation) as session:
            response = session.history_summary(
                actions,
                request_id=f"prefix12-runtime-{process_index}",
                count_mode="logical",
            )
        assert response.status == "available"
        assert response.state is not None
        records.append(response.state)

    assert installation.commit == PINNED_SANMILL_COMMIT
    assert installation.checkout_head == PINNED_SANMILL_COMMIT
    assert installation.tree == PINNED_SANMILL_TREE
    assert installation.path_lookup_key == "sanmill_prefix12_checkout"
    assert installation.require_exact_head
    assert installation.portable_record()["checkout_policy"] == (
        "exact pinned commit with a clean worktree"
    )
    assert records[0] == records[1]
    assert records[0].action_token_count == 8
    assert records[0].logical_ply_count == 7
    assert records[0].logical_plies_by_side == (4, 3)

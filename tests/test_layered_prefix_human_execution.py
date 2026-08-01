from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.evaluation.layered_human_execution import (
    LayeredHumanExecutionError,
    build_layered_human_execution,
    verify_layered_human_execution,
)
from learned_ai.evaluation.sanmill_uci import (
    PREFIX12_REPLAY_INSTALLATION_CONTRACT,
    inspect_sanmill_installation,
)
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs" / "experiments"
EVIDENCE = ROOT / "docs" / "evidence"
LOCAL_PATHS = ROOT / "data" / "training_paths.local.json"
HUMAN_CORE = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json"
)
SOURCE_CORE = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json"
)
HUMAN_AUDIT = EVIDENCE / "sanmill-layered-human-source-audit-2026-07-25.json"
RUNTIME = (
    EXPERIMENTS / "sanmill-prefix12-human-replay-runtime-2026-08-01.json"
)
EXECUTION = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-human-execution-2026-08-01.json"
)
EXECUTION_DOC = EXECUTION.with_suffix(".md")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_local_runtime() -> bool:
    if not LOCAL_PATHS.is_file():
        return False
    try:
        return bool(_load(LOCAL_PATHS).get("sanmill_prefix12_checkout"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def test_frozen_human_execution_records_verify_portably() -> None:
    assert verify_layered_human_execution(
        _load(EXECUTION),
        human_core_decision=_load(HUMAN_CORE),
        source_core_decision=_load(SOURCE_CORE),
        human_audit=_load(HUMAN_AUDIT),
        runtime_decision=_load(RUNTIME),
    ) == {
        "records": 21,
        "prefix_identities": 21,
        "history_identities": 21,
    }


def test_human_execution_document_links_all_inputs() -> None:
    document = EXECUTION_DOC.read_text(encoding="utf-8")
    for target in (HUMAN_CORE, SOURCE_CORE, RUNTIME, EXECUTION):
        assert f"({target.name})" in document


def test_frozen_human_execution_rejects_runtime_record_drift() -> None:
    payload = _load(EXECUTION)
    payload = copy.deepcopy(payload)
    payload["records"][0]["execution_record"]["sanmill"]["license"][
        "sha256"
    ] = "0" * 64
    execution_record = payload["records"][0]["execution_record"]
    prefix_body = dict(execution_record)
    prefix_body.pop("prefix_identity")
    execution_record["prefix_identity"] = canonical_sha256(prefix_body)
    execution_evidence = {
        "records": payload["records"],
        "data_query": payload["data_query"],
    }
    payload["human_execution_identity"] = canonical_sha256(
        execution_evidence
    )

    with pytest.raises(
        LayeredHumanExecutionError,
        match="execution prefix evidence drifted",
    ):
        verify_layered_human_execution(
            payload,
            human_core_decision=_load(HUMAN_CORE),
            source_core_decision=_load(SOURCE_CORE),
            human_audit=_load(HUMAN_AUDIT),
            runtime_decision=_load(RUNTIME),
        )


@pytest.mark.skipif(
    not _has_local_runtime(),
    reason="requires the ignored sanmill_prefix12_checkout registry entry",
)
def test_local_runtime_exactly_reproduces_frozen_human_execution() -> None:
    installation = inspect_sanmill_installation(
        LOCAL_PATHS,
        contract=PREFIX12_REPLAY_INSTALLATION_CONTRACT,
    )
    generated = build_layered_human_execution(
        human_core_decision=_load(HUMAN_CORE),
        source_core_decision=_load(SOURCE_CORE),
        human_audit=_load(HUMAN_AUDIT),
        runtime_decision=_load(RUNTIME),
        installation=installation,
    )

    assert generated == _load(EXECUTION)

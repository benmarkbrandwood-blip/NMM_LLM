from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.publish_target_refresh_direct_crossplay_evidence import (
    DEFAULT_OUTPUT,
    DirectCrossplayEvidenceError,
    validate_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def _tracked_evidence() -> dict[str, object]:
    return json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))


def test_tracked_attempt_003_evidence_is_closed_and_non_promotional() -> None:
    evidence = _tracked_evidence()

    assert validate_evidence(evidence) == (
        "9a5df62da2c605d580c3d519d8aa18d45e7e444639eb458dd93976f31fcb8d19"
    )
    assert evidence["plan"] == {
        "path": (
            "docs/experiments/"
            "sanmill-target-refresh-direct-crossplay-v1-attempt-003.json"
        ),
        "plan_identity": (
            "2f1665e59aaa7af96af345381338689a7edd51c55401a13a8c9fd4c8a58535ff"
        ),
        "raw_sha256": (
            "17c43b513602479d00eed5b36a5b5c02a779c31111894f72ab7acf6bced20c25"
        ),
    }
    observed = evidence["observed_facts"]
    assert observed["games"] == 288
    assert observed["pairs"] == 144
    assert observed["decision"]["classification"] == (
        "material_no_refresh_direct_effect"
    )
    assert evidence["interpretation"]["permanent_no_refresh_selected"] is False
    assert evidence["claim_boundary"] == {
        "automatic_long_run_selection": False,
        "development_mechanism_evidence_only": True,
        "held_out_strength": False,
        "promotion": False,
        "publication": False,
    }


def test_attempt_003_evidence_identity_rejects_scientific_tampering() -> None:
    evidence = _tracked_evidence()
    tampered = copy.deepcopy(evidence)
    tampered["observed_facts"]["paired"]["mean_score_effect"] = 0.0

    with pytest.raises(DirectCrossplayEvidenceError, match="evidence_identity differs"):
        validate_evidence(tampered)


def test_attempt_003_evidence_file_is_canonical_lf_json() -> None:
    raw = DEFAULT_OUTPUT.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    assert DEFAULT_OUTPUT.resolve().is_relative_to(ROOT)

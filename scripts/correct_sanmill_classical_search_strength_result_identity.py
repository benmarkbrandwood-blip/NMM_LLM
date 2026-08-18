#!/usr/bin/env python3
"""Reseal the immutable classical-search result after JSON key coercion."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import sha256_file


ORIGINAL = Path(
    "docs/evidence/sanmill-classical-search-strength-v2-manifest-2026-08-18.json"
)
CORRECTED = Path(
    "docs/evidence/"
    "sanmill-classical-search-strength-v2-manifest-identity-corrected-"
    "2026-08-18.json"
)
CORRECTION = Path(
    "docs/evidence/"
    "sanmill-classical-search-strength-v2-result-identity-correction-"
    "2026-08-18.json"
)
ORIGINAL_FILE_SHA256 = (
    "1f20252abda2d697e7212c5b3a815994b970d6eee66850505c3b0db1fde5e079"
)
ORIGINAL_RECORDED_IDENTITY = (
    "1bdeadbff571be3f9cdf5a54e0b82c1abd7981d58fc440497b1f8420bc323b14"
)
EXPECTED_CORRECTED_IDENTITY = (
    "fe77312f303670c1bb8489f423926ac87fe8cbe702e252d366b452484c0bfe9f"
)
CORRECTION_SCHEMA = (
    "nmm.sanmill-classical-search-strength-result-identity-correction.v1"
)


def _verify_chain(path: Path) -> dict[str, Any]:
    previous: str | None = None
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        wrapper = json.loads(line)
        digest = wrapper["record_sha256"]
        if "record" in wrapper:
            record = wrapper["record"]
        else:
            record = dict(wrapper)
            record.pop("record_sha256")
        if record.get("previous_record_sha256") != previous:
            raise RuntimeError(f"ledger predecessor differs: {path}")
        if canonical_sha256(record) != digest:
            raise RuntimeError(f"ledger record hash differs: {path}")
        previous = digest
        count += 1
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "records": count,
        "tail_record_sha256": previous,
        "file_sha256": sha256_file(path),
    }


def main() -> int:
    original_path = ROOT / ORIGINAL
    corrected_path = ROOT / CORRECTED
    correction_path = ROOT / CORRECTION
    if sha256_file(original_path) != ORIGINAL_FILE_SHA256:
        raise RuntimeError("original malformed result file differs")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    body = dict(original)
    recorded_identity = body.pop("result_identity", None)
    if recorded_identity != ORIGINAL_RECORDED_IDENTITY:
        raise RuntimeError("original recorded identity differs")
    roundtrip_identity = canonical_sha256(body)
    if roundtrip_identity != EXPECTED_CORRECTED_IDENTITY:
        raise RuntimeError("corrected identity expectation differs")

    affected_paths = [
        "analysis.by_arm.classical-difficulty-9-nodes-13887000.work.completed_depths",
        "analysis.by_arm.classical-difficulty-10-nodes-18367000.work.completed_depths",
    ]
    for dotted in affected_paths:
        value: Any = body
        for part in dotted.split("."):
            value = value[part]
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and key.isdigit() for key in value
        ):
            raise RuntimeError(f"expected JSON-coerced depth keys absent: {dotted}")

    corrected = write_sealed_json(
        corrected_path,
        body,
        identity_field="result_identity",
    )
    if corrected["result_identity"] != EXPECTED_CORRECTED_IDENTITY:
        raise RuntimeError("written corrected result identity differs")
    reloaded = json.loads(corrected_path.read_text(encoding="utf-8"))
    reloaded_body = dict(reloaded)
    reloaded_identity = reloaded_body.pop("result_identity")
    if canonical_sha256(reloaded_body) != reloaded_identity:
        raise RuntimeError("corrected result is not round-trip verifiable")
    if reloaded_body != body:
        raise RuntimeError("corrected result changed semantic payload")

    raw = original["machine_records"]
    ledgers = {
        "classical": _verify_chain(ROOT / raw["raw_classical_ledger"]),
        "reproduction": _verify_chain(ROOT / raw["raw_reproduction_ledger"]),
    }
    if ledgers["classical"]["records"] != 192:
        raise RuntimeError("classical ledger count differs")
    if ledgers["reproduction"]["records"] != 96:
        raise RuntimeError("reproduction ledger count differs")

    correction = write_sealed_json(
        correction_path,
        {
            "schema_version": CORRECTION_SCHEMA,
            "status": "identity_corrected_without_measurement_reexecution",
            "detected_after_measurement": True,
            "original": {
                "path": str(ORIGINAL).replace("\\", "/"),
                "file_sha256": ORIGINAL_FILE_SHA256,
                "recorded_non_roundtrip_identity": ORIGINAL_RECORDED_IDENTITY,
                "recomputed_roundtrip_identity": roundtrip_identity,
                "preserved_byte_for_byte": True,
            },
            "cause": {
                "summary": "integer completed-depth histogram keys were hashed before JSON coerced them to strings",
                "affected_paths": affected_paths,
                "measurement_values_changed": False,
            },
            "corrected": {
                "path": str(CORRECTED).replace("\\", "/"),
                "result_identity": corrected["result_identity"],
                "file_sha256": sha256_file(corrected_path),
                "roundtrip_verified": True,
                "semantic_payload_equal_after_removing_identity": True,
                "only_field_difference": "result_identity",
            },
            "raw_ledger_audit": ledgers,
            "additional_complete_games": 0,
            "additional_engine_searches": 0,
            "additional_malom_queries": 0,
            "database_writes": 0,
            "claim_boundary_unchanged": original["claim_boundary"],
        },
        identity_field="correction_identity",
    )
    print(corrected["result_identity"])
    print(correction["correction_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

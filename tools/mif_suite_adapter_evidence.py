#!/usr/bin/env python3
"""Generate the pinned NMM_LLM MIF Suite adapter evidence manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIRECTORY = ROOT / "docs" / "evidence"
CAPABILITY_PATH = (
    EVIDENCE_DIRECTORY / "mif-suite-1.0-nmm-capability-2026-08-07.json"
)
DETERMINISTIC_REPORT_PATH = (
    EVIDENCE_DIRECTORY
    / "mif-suite-1.0-nmm-deterministic-report-2026-08-07.json"
)
DIFFERENTIAL_REPORT_PATH = (
    EVIDENCE_DIRECTORY
    / "mif-suite-1.0-nmm-differential-report-2026-08-07.json"
)
EVIDENCE_PATH = (
    EVIDENCE_DIRECTORY
    / "mif-suite-1.0-nmm-adapter-evidence-2026-08-07.json"
)

MIF_COMMIT = "3ee7e57c7d4c7208be91f62914f344a587fb0f70"
IMPLEMENTATION_COMMIT = "a7e7dbd5461cc2d8d8c0a09317d6091598202214"
EVIDENCE_COMMIT = "ae7911e37fa2bf45ea6074850453bbad2479438e"
SUITE_JCS_SHA256 = (
    "sha256:81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f"
)
SUITE_RAW_SHA256 = (
    "sha256:088ca33234289b06d9276aa4c430758222aa85d61621dee7bef4bfc6dcc069a4"
)
ARTIFACT_INDEX_RAW_SHA256 = (
    "sha256:5acbb714bed77e24eaac72fa5f24d2e54d1e17aaf568a8b60718c840281a6541"
)
CAPABILITY_RAW_SHA256 = (
    "sha256:cd661b1156bf7269f976e050446d01797c9959482f1e1843e21ae3ea7f70dcce"
)
DETERMINISTIC_REPORT_RAW_SHA256 = (
    "sha256:3463f438531fd52847df44fa4186dcba13ed22c7c570a0cc216d9a7eaa797665"
)
DIFFERENTIAL_REPORT_RAW_SHA256 = (
    "sha256:4c86725bfcd1759433374938c8d8eb2a1dacfa6ea3723592eff759162fce8da6"
)
CONFIG_DIGEST = (
    "sha256:c6eb5edc21773c017e7a2d5d9050b38cb08450658a286e64a395f1edc6b7074e"
)
FINALIZATION_LAUNCH_RAW_SHA256 = (
    "sha256:3ab079c44158979eb78221a64abe5347e9e3697b33972673659a0ea80053536d"
)
THREE_ADAPTER_CONFIG_DIGEST = (
    "sha256:133cc572ba786ebd544e9fe5fc89c67248432952a1a2fce451a3e1ec6bfda0f2"
)
RULESET_SEMANTIC_DIGESTS = [
    "sha256:173caf8189defd1ab7d4a3e8b9e26688a07fd77976bf09d56bff5fe0c273e1a1",
    "sha256:224f7e368e322a4cc8c1225a025fb548d5b41eb096d34b7ae0543182d1aa9393",
]
TESTED_CLASSES = [
    "identity",
    "key",
    "position",
    "replay",
    "ruleset",
    "transform",
]


class EvidenceGenerationError(RuntimeError):
    """Raised when a pinned evidence input no longer matches its contract."""


def _raw_sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceGenerationError(f"expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceGenerationError(message)


def _verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_hashes = {
        CAPABILITY_PATH: CAPABILITY_RAW_SHA256,
        DETERMINISTIC_REPORT_PATH: DETERMINISTIC_REPORT_RAW_SHA256,
        DIFFERENTIAL_REPORT_PATH: DIFFERENTIAL_REPORT_RAW_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = _raw_sha256(path)
        _require(actual == expected, f"raw SHA-256 mismatch for {path}: {actual}")

    capability = _load_json(CAPABILITY_PATH)
    deterministic = _load_json(DETERMINISTIC_REPORT_PATH)
    differential = _load_json(DIFFERENTIAL_REPORT_PATH)

    levels = {item["id"]: item["level"] for item in capability["classes"]}
    tested = [identifier for identifier, level in levels.items() if level == "tested"]
    rulesets = sorted(item["semanticDigest"] for item in capability["rulesets"])
    _require(capability["format"] == "MIFCAP/1.0", "unexpected capability format")
    _require(capability["suites"] == [SUITE_JCS_SHA256], "Suite pin mismatch")
    _require(tested == TESTED_CLASSES, "tested capability classes changed")
    _require(levels.get("conversion") == "none", "conversion must remain none")
    _require("full" not in levels, "full conformance must not be claimed")
    _require(capability["conversions"] == [], "conversion claims must remain empty")
    _require(rulesets == RULESET_SEMANTIC_DIGESTS, "Suite rulesets changed")

    _require(
        deterministic["protocol"] == "MIF-INTEROP-REPORT/1",
        "unexpected deterministic report protocol",
    )
    _require(
        deterministic["summary"] == {"failed": 0, "passed": 58},
        "deterministic report is not 58/58",
    )
    _require(
        deterministic["configDigest"] == CONFIG_DIGEST,
        "deterministic config digest changed",
    )

    summary = differential["summary"]
    _require(
        differential["protocol"] == "MIF-INTEROP-DIFFERENTIAL-REPORT/1",
        "unexpected differential report protocol",
    )
    _require(differential["status"] == "passed", "differential report failed")
    _require(
        differential["suiteConformance"] is False,
        "raw differential report must retain its candidate-only boundary",
    )
    _require(
        summary["runsPassed"] == 10 and summary["runsFailed"] == 0,
        "differential report is not 10/10",
    )
    _require(
        summary["negativePassed"] == 5 and summary["negativeFailed"] == 0,
        "mutation report is not 5/5",
    )
    _require(
        differential["configDigest"] == CONFIG_DIGEST,
        "differential config digest changed",
    )
    return capability, deterministic, differential


def build_evidence_manifest() -> dict[str, Any]:
    """Build the stable Suite-bound manifest after validating raw inputs."""

    _verify_inputs()
    return {
        "protocol": "MIF-SUITE-ADAPTER-EVIDENCE/1",
        "status": "suite-bound-evidence",
        "classification": "exact-for-tested-domain",
        "suiteConformance": True,
        "fullConformance": False,
        "conversionClaimed": False,
        "unexplainedDifferences": 0,
        "adapter": "nmm-llm-python",
        "mifCommit": MIF_COMMIT,
        "implementationCommit": IMPLEMENTATION_COMMIT,
        "evidenceCommit": EVIDENCE_COMMIT,
        "suiteJcsSha256": SUITE_JCS_SHA256,
        "suiteRawSha256": SUITE_RAW_SHA256,
        "artifactIndexRawSha256": ARTIFACT_INDEX_RAW_SHA256,
        "capabilityRawSha256": CAPABILITY_RAW_SHA256,
        "deterministicReportRawSha256": DETERMINISTIC_REPORT_RAW_SHA256,
        "differentialReportRawSha256": DIFFERENTIAL_REPORT_RAW_SHA256,
        "rulesetSemanticDigests": RULESET_SEMANTIC_DIGESTS,
        "testedClasses": TESTED_CLASSES,
        "artifacts": {
            "capability": {
                "path": (
                    "docs/evidence/"
                    "mif-suite-1.0-nmm-capability-2026-08-07.json"
                ),
                "rawSha256": CAPABILITY_RAW_SHA256,
                "format": "MIFCAP/1.0",
            },
            "deterministicReport": {
                "path": (
                    "docs/evidence/"
                    "mif-suite-1.0-nmm-deterministic-report-2026-08-07.json"
                ),
                "rawSha256": DETERMINISTIC_REPORT_RAW_SHA256,
                "protocol": "MIF-INTEROP-REPORT/1",
                "configDigest": CONFIG_DIGEST,
                "summary": {"passed": 58, "failed": 0},
            },
            "differentialReport": {
                "path": (
                    "docs/evidence/"
                    "mif-suite-1.0-nmm-differential-report-2026-08-07.json"
                ),
                "rawSha256": DIFFERENTIAL_REPORT_RAW_SHA256,
                "protocol": "MIF-INTEROP-DIFFERENTIAL-REPORT/1",
                "configDigest": CONFIG_DIGEST,
                "summary": {
                    "runsPassed": 10,
                    "runsFailed": 0,
                    "mutationFamiliesPassed": 5,
                    "mutationFamiliesFailed": 0,
                },
            },
        },
        "verification": {
            "finalizationLaunchRawSha256": FINALIZATION_LAUNCH_RAW_SHA256,
            "rawArtifactsByteIdenticalAcrossRuns": True,
            "focusedTests": {"passed": 66, "failed": 0},
            "workspaceRegression": {
                "collected": 1179,
                "passedAfterSerialRerun": 1171,
                "blocked": 8,
                "changedSubsystemFailures": 0,
                "blockedReason": (
                    "historical Sanmill strict-v2 binary bytes are unavailable; "
                    "machine-local bridge tests rejected the replacement binary"
                ),
            },
            "ruff": "passed",
            "focusedMypy": "passed",
            "compileall": "passed",
            "referenceGameplayImportMatches": 0,
            "threeAdapterPreflight": {
                "sanmillCheckoutCommit": (
                    "9d36d04b4d2a8cd5c660e9582426bedeb888b591"
                ),
                "sanmillImplementationCommit": (
                    "7e86de7e8156a7d7f46a6a6179a8878051699505"
                ),
                "configDigest": THREE_ADAPTER_CONFIG_DIGEST,
                "deterministicPassed": 58,
                "deterministicFailed": 0,
                "differentialRunsPassed": 10,
                "differentialRunsFailed": 0,
                "mutationFamiliesPassed": 5,
                "mutationFamiliesFailed": 0,
            },
        },
    }


def render_evidence_manifest() -> bytes:
    """Render the manifest using stable repository JSON formatting."""

    return (
        json.dumps(build_evidence_manifest(), ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=EVIDENCE_PATH)
    args = parser.parse_args()
    rendered = render_evidence_manifest()
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_bytes() != rendered:
            raise EvidenceGenerationError(
                f"evidence manifest is not reproducible: {output}"
            )
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

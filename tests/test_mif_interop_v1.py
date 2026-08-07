"""Focused invariants for the independent MIF-INTEROP/1 adapter."""

from __future__ import annotations

import pytest

from learned_ai.interop.mif_v1.adapter import MifInteropAdapter, capabilities
from learned_ai.interop.mif_v1.common import (
    MIF_ADAPTER_PROTOCOL_SHA256,
    MIF_CHINESE_SPEC_SHA256,
    MIF_COMMIT,
    MIF_DETERMINISTIC_CORPUS_SHA256,
    MIF_ENGLISH_SPEC_SHA256,
    MIF_EXECUTABLE_CORPUS_SHA256,
    MIF_INDEX_SHA256,
    MIF_LICENSE_SHA256,
    MIF_RELEASE_MANIFEST_SHA256,
    MIF_SMOKE_CORPUS_SHA256,
    MIF_SUITE_COMMIT,
    MIF_SUITE_JCS_SHA256,
    MIF_SUITE_RAW_SHA256,
    MifError,
    parse_ijson,
    transform_coordinate,
)
from learned_ai.interop.mif_v1.engine import repetition_root


def test_adapter_is_pinned_to_the_tested_suite_domain() -> None:
    value = capabilities()
    assert value["implementation"]["version"].endswith(MIF_SUITE_COMMIT[:12])
    assert value["suites"] == [MIF_SUITE_JCS_SHA256]
    assert value["testedCorpora"] == [
        {
            "digest": MIF_SMOKE_CORPUS_SHA256,
            "classes": [
                "identity",
                "position",
                "replay",
                "ruleset",
                "transform",
            ],
        },
        {
            "digest": MIF_DETERMINISTIC_CORPUS_SHA256,
            "classes": [
                "identity",
                "key",
                "position",
                "replay",
                "ruleset",
                "transform",
            ],
        },
    ]
    assert value["annotations"] == {
        "contractCommit": MIF_COMMIT,
        "wireCommit": MIF_COMMIT,
        "suiteCandidateCommit": MIF_SUITE_COMMIT,
        "suiteJcsSha256": MIF_SUITE_JCS_SHA256,
        "suiteRawSha256": MIF_SUITE_RAW_SHA256,
        "englishSpec": MIF_ENGLISH_SPEC_SHA256,
        "chineseSpec": MIF_CHINESE_SPEC_SHA256,
        "artifactIndex": MIF_INDEX_SHA256,
        "executableCorpus": MIF_EXECUTABLE_CORPUS_SHA256,
        "adapterProtocol": MIF_ADAPTER_PROTOCOL_SHA256,
        "smokeCorpus": MIF_SMOKE_CORPUS_SHA256,
        "deterministicCorpus": MIF_DETERMINISTIC_CORPUS_SHA256,
        "differentialLaunch": (
            "sha256:560ef369fde248bd96d3468a4336442db1d970ede04f488821509e69925fd48e"
        ),
        "releaseManifest": MIF_RELEASE_MANIFEST_SHA256,
        "license": MIF_LICENSE_SHA256,
        "scope": "exact-for-tested-domain; no full or conversion claim",
    }


def test_duplicate_json_member_is_rejected_after_unescaping() -> None:
    with pytest.raises(MifError, match="duplicate-member-after-unescape"):
        parse_ijson(b'{"payload":{},"pay\\u006coad":{}}')


def test_empty_sparse_repetition_tree_matches_frozen_contract() -> None:
    assert repetition_root([], 3) == (
        "sha256:e9fbf966ccdff764594a5e199e6aea0cc36034b46c8057cc3df88a088c20101a"
    )


def test_d4_rotation_uses_mif_coordinate_direction() -> None:
    assert transform_coordinate("a7", "r90ccw") == "a1"
    assert transform_coordinate("a1", "r90cw") == "a7"


def test_capabilities_request_has_one_protocol_response() -> None:
    response = MifInteropAdapter().handle(
        {
            "protocol": "MIF-INTEROP/1",
            "kind": "request",
            "requestId": "capabilities",
            "operation": "capabilities",
            "payload": {},
        }
    )
    assert response["status"] == "ok"
    assert response["result"]["capabilities"]["format"] == "MIFCAP/1.0"

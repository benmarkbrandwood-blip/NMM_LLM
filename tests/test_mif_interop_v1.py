"""Focused invariants for the independent MIF-INTEROP/1 adapter."""

from __future__ import annotations

import pytest

from learned_ai.interop.mif_v1.adapter import MifInteropAdapter, capabilities
from learned_ai.interop.mif_v1.common import (
    MIF_COMMIT,
    MIF_EXECUTABLE_CORPUS_SHA256,
    MifError,
    parse_ijson,
    transform_coordinate,
)
from learned_ai.interop.mif_v1.engine import repetition_root


def test_adapter_is_pinned_without_claiming_a_published_suite() -> None:
    value = capabilities()
    assert value["implementation"]["version"].endswith(MIF_COMMIT[:12])
    assert value["suites"] == []
    assert value["testedCorpora"] == []
    assert value["annotations"]["contractCommit"] == MIF_COMMIT
    assert value["annotations"]["executableCorpus"] == MIF_EXECUTABLE_CORPUS_SHA256


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

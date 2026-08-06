"""Candidate-3 deterministic diagnostic regressions."""

from __future__ import annotations

from typing import Any

import pytest

from learned_ai.interop.mif_v1.adapter import MifInteropAdapter
from tests.mif_interop_fixtures import example_manifest


def _request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    return MifInteropAdapter().handle(
        {
            "protocol": "MIF-INTEROP/1",
            "kind": "request",
            "requestId": f"candidate3-{operation}",
            "operation": operation,
            "payload": payload,
        }
    )


@pytest.mark.parametrize(
    ("value", "category", "code"),
    [
        (
            "MPK/1.0 mill24-state-v1 example-morris@1 "
            "structural-d4-v1 ........................ b p 9,9",
            "integrity",
            "mpk-semantic-digest-missing",
        ),
        (
            "MPK/1.0 mill24-state-v1 example-morris@1 "
            "sha256:224F7E368E322A4CC8C1225A025FB548D5B41EB096D34B7AE"
            "0543182D1AA9393 structural-d4-v1 ........................ b p 9,9",
            "canonical",
            "non-canonical-digest",
        ),
    ],
)
def test_candidate3_mpk_diagnostics(
    value: str,
    category: str,
    code: str,
) -> None:
    response = _request(
        "canonicalize",
        {"format": "MPK/1.0", "manifest": example_manifest(), "value": value},
    )
    assert response["status"] == "error"
    error = response["diagnostics"]["errors"][0]
    assert (error["category"], error["code"]) == (category, code)


def test_claim_during_obligation_is_inconsistent() -> None:
    response = _request(
        "execute",
        {
            "manifest": example_manifest(),
            "origin": (
                "MFEN/1.0 mill24-state-v1 BB....../......../W....... "
                "b p p 8,7 - 0 3 -"
            ),
            "events": [
                {"actor": "b", "at": "g7", "seq": 1, "type": "place"},
                {
                    "actor": "b",
                    "reason": "repetition",
                    "seq": 2,
                    "type": "claim-draw",
                },
            ],
            "repetitionSeed": [],
            "preOriginClaims": [],
        },
    )
    assert response["status"] == "error"
    assert response["diagnostics"]["errors"] == [
        {
            "category": "inconsistent",
            "code": "claim-during-obligation",
            "eventSeq": 2,
            "message": "claim-draw is forbidden during an obligation",
        }
    ]

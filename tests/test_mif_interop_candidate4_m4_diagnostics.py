"""Candidate-4 M4 diagnostic-policy regressions."""

from __future__ import annotations

from typing import Any

import pytest

from learned_ai.interop.mif_v1.adapter import MifInteropAdapter
from tests.mif_interop_fixtures import (
    EMPTY_ORIGIN,
    clone,
    example_manifest,
    offer_r1_mstate,
)


def _request(
    operation: str,
    payload: dict[str, Any],
    *,
    request_id: str,
) -> dict[str, Any]:
    return MifInteropAdapter().handle(
        {
            "protocol": "MIF-INTEROP/1",
            "kind": "request",
            "requestId": request_id,
            "operation": operation,
            "payload": payload,
        }
    )


def _remove_without_obligation_response() -> dict[str, Any]:
    return _request(
        "execute",
        {
            "manifest": example_manifest(),
            "origin": EMPTY_ORIGIN,
            "events": [
                {
                    "actor": "b",
                    "seq": 1,
                    "target": {"zone": "board", "at": "a7"},
                    "type": "remove",
                }
            ],
            "repetitionSeed": [],
            "preOriginClaims": [],
        },
        request_id="mutation-illegal-remove-without-obligation",
    )


def _replay_mismatch_response(kind: str) -> dict[str, Any]:
    mstate = clone(offer_r1_mstate())
    if kind == "checkpoint":
        mstate["current"] = mstate["current"].replace(" 0 0 -", " 0 1 -")
    elif kind == "repetition":
        mstate["repetitionHistory"] = []
    elif kind == "claims":
        mstate["claims"] = []
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"unsupported replay mismatch: {kind}")
    return _request(
        "replay",
        {"mstate": mstate, "manifest": example_manifest()},
        request_id=f"m4-{kind}-mismatch",
    )


def _semantic_diagnostics(response: dict[str, Any]) -> dict[str, Any]:
    diagnostics = clone(response["diagnostics"])
    diagnostics.pop("annotations", None)
    for error in diagnostics["errors"]:
        error.pop("message", None)
    return diagnostics


def test_remove_without_obligation_is_inconsistent_at_the_event() -> None:
    response = _remove_without_obligation_response()

    assert response["status"] == "error"
    assert response["diagnostics"]["errors"][0] == {
        "category": "inconsistent",
        "code": "remove-without-obligation",
        "eventSeq": 1,
        "message": "remove event has no pending obligation",
    }


def test_truncated_repetition_history_omits_compared_values() -> None:
    response = _replay_mismatch_response("repetition")
    error = response["diagnostics"]["errors"][0]

    assert response["status"] == "error"
    assert error["category"] == "replay"
    assert error["code"] == "repetition-history-mismatch"
    assert "expected" not in error
    assert "actual" not in error


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            _remove_without_obligation_response,
            {
                "format": "MIFDIAG/1.0",
                "errors": [
                    {
                        "category": "inconsistent",
                        "code": "remove-without-obligation",
                        "eventSeq": 1,
                    }
                ],
            },
        ),
        (
            lambda: _replay_mismatch_response("repetition"),
            {
                "format": "MIFDIAG/1.0",
                "errors": [
                    {
                        "category": "replay",
                        "code": "repetition-history-mismatch",
                    }
                ],
            },
        ),
    ],
    ids=["remove-without-obligation", "truncated-repetition-history"],
)
def test_m4_negative_diagnostics_match_semantic_equality(
    response: Any,
    expected: dict[str, Any],
) -> None:
    assert _semantic_diagnostics(response()) == expected


@pytest.mark.parametrize(
    ("kind", "code"),
    [
        ("checkpoint", "checkpoint-mismatch"),
        ("claims", "claims-mismatch"),
    ],
)
def test_adjacent_replay_mismatches_share_compact_diagnostic_policy(
    kind: str,
    code: str,
) -> None:
    error = _semantic_diagnostics(_replay_mismatch_response(kind))["errors"][0]

    assert error == {"category": "replay", "code": code}


def test_valid_replay_remains_successful() -> None:
    response = _request(
        "replay",
        {"mstate": offer_r1_mstate(), "manifest": example_manifest()},
        request_id="m4-valid-replay",
    )

    assert response["status"] == "ok"
    assert response["result"]["current"] == EMPTY_ORIGIN

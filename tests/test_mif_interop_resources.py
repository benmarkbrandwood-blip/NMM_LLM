from __future__ import annotations

from typing import Any

import pytest

from learned_ai.interop.mif_v1.adapter import MifInteropAdapter, capabilities
from learned_ai.interop.mif_v1.engine import replay
from tests.mif_interop_fixtures import (
    EMPTY_ORIGIN,
    clone,
    example_manifest,
    offer_r1_mstate,
)


LIMITS = {
    "events": 100_000,
    "repetition-entries": 100_000,
}


def test_capability_limits_match_enforced_contract() -> None:
    published = {
        item["name"]: item["limit"] for item in capabilities()["resourceLimits"]
    }
    assert published == {
        "events": LIMITS["events"],
        "interop-request-bytes": 16_777_216,
        "repetition-entries": LIMITS["repetition-entries"],
    }


def _request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    return MifInteropAdapter().handle(
        {
            "protocol": "MIF-INTEROP/1",
            "kind": "request",
            "requestId": f"limit-{operation}",
            "operation": operation,
            "payload": payload,
        }
    )


def _assert_limit(
    response: dict[str, Any],
    *,
    name: str,
    actual: int,
) -> None:
    assert response["status"] == "error"
    assert response["diagnostics"]["errors"] == [
        {
            "category": "resource",
            "code": "resource-limit",
            "resourceLimit": {
                "name": name,
                "limit": LIMITS[name],
                "actual": actual,
            },
            "message": f"{name} resource limit exceeded",
        }
    ]


def _oversized(name: str) -> list[Any]:
    return [None] * (LIMITS[name] + 1)


def test_execute_enforces_event_limit() -> None:
    response = _request(
        "execute",
        {
            "manifest": example_manifest(),
            "origin": EMPTY_ORIGIN,
            "events": _oversized("events"),
            "repetitionSeed": [],
            "preOriginClaims": [],
        },
    )
    _assert_limit(response, name="events", actual=100_001)


def test_resource_preflight_precedes_closed_object_validation() -> None:
    response = _request(
        "execute",
        {
            "manifest": example_manifest(),
            "origin": EMPTY_ORIGIN,
            "events": _oversized("events"),
            "repetitionSeed": [],
            "preOriginClaims": [],
            "unknownFutureMeaning": True,
        },
    )
    _assert_limit(response, name="events", actual=100_001)


def test_execute_enforces_repetition_seed_limit() -> None:
    response = _request(
        "execute",
        {
            "manifest": example_manifest(),
            "origin": EMPTY_ORIGIN,
            "events": [],
            "repetitionSeed": _oversized("repetition-entries"),
            "preOriginClaims": [],
        },
    )
    _assert_limit(response, name="repetition-entries", actual=100_001)


@pytest.mark.parametrize("operation", ["replay", "project-logical-turns"])
@pytest.mark.parametrize(
    ("member", "limit_name"),
    [
        ("events", "events"),
        ("repetitionHistory", "repetition-entries"),
    ],
)
def test_mstate_operations_enforce_resource_limits(
    operation: str,
    member: str,
    limit_name: str,
) -> None:
    mstate = clone(offer_r1_mstate())
    mstate[member] = _oversized(limit_name)
    response = _request(
        operation,
        {"mstate": mstate, "manifest": example_manifest()},
    )
    _assert_limit(response, name=limit_name, actual=100_001)


@pytest.mark.parametrize(
    ("member", "limit_name"),
    [
        ("events", "events"),
        ("repetitionHistory", "repetition-entries"),
    ],
)
def test_mstate_transform_enforces_resource_limits(
    member: str,
    limit_name: str,
) -> None:
    mstate = clone(offer_r1_mstate())
    mstate[member] = _oversized(limit_name)
    response = _request(
        "transform",
        {
            "kind": "mstate",
            "document": mstate,
            "manifest": example_manifest(),
            "transform": "r90ccw",
            "verifyReplay": True,
            "requireEquivalence": False,
        },
    )
    _assert_limit(response, name=limit_name, actual=100_001)


def test_decision_transform_enforces_repetition_limit() -> None:
    _, replay_result = replay(offer_r1_mstate(), example_manifest())
    response = _request(
        "transform",
        {
            "kind": "decision-state",
            "document": replay_result["decisionState"],
            "manifest": example_manifest(),
            "repetitionHistory": _oversized("repetition-entries"),
            "transform": "r90ccw",
            "verifyReplay": False,
            "requireEquivalence": False,
        },
    )
    _assert_limit(response, name="repetition-entries", actual=100_001)

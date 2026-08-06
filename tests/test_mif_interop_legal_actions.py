"""Candidate-3 legal gameplay-action projection regressions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from learned_ai.interop.mif_v1.adapter import MifInteropAdapter
from learned_ai.interop.mif_v1.common import POINTS
from learned_ai.interop.mif_v1.model import resolve_manifest
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


def _legal_actions(current: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    selected_manifest = manifest if manifest is not None else example_manifest()
    response = _request(
        "project-legal-actions",
        {"manifest": selected_manifest, "current": current},
    )
    assert response["status"] == "ok"
    document = response["result"]["document"]
    assert document["profile"] == "legal-actions-v1"
    assert document["stateProfile"] == "mill24-state-v1"
    assert document["semanticDigest"] == resolve_manifest(
        selected_manifest
    ).semantic_digest
    assert document["current"] == current
    return document


def test_project_legal_actions_for_initial_placing() -> None:
    document = _legal_actions(
        "MFEN/1.0 mill24-state-v1 ......../......../........ b p p 9,9 - 0 0 -"
    )
    assert document["actions"] == [
        {"actor": "b", "type": "place", "at": point} for point in POINTS
    ]


def test_project_legal_actions_includes_placing_moves_after_places() -> None:
    manifest = example_manifest()
    manifest["placing"]["movementAllowed"] = True
    current = (
        "MFEN/1.0 mill24-state-v1 W......./B......./........ "
        "w p p 8,8 - 0 0 -"
    )
    document = _legal_actions(current, manifest)
    empty = [point for point in POINTS if point not in {"a7", "b6"}]
    assert document["actions"] == [
        *({"actor": "w", "type": "place", "at": point} for point in empty),
        {"actor": "w", "type": "move", "from": "a7", "to": "d7"},
        {"actor": "w", "type": "move", "from": "a7", "to": "a4"},
    ]


def test_execute_does_not_apply_flying_during_placing() -> None:
    manifest = example_manifest()
    manifest["placing"]["movementAllowed"] = True
    response = _request(
        "execute",
        {
            "manifest": manifest,
            "origin": (
                "MFEN/1.0 mill24-state-v1 W......./B......./........ "
                "w p p 8,8 - 0 0 -"
            ),
            "events": [
                {
                    "actor": "w",
                    "from": "a7",
                    "seq": 1,
                    "to": "g7",
                    "type": "move",
                }
            ],
            "repetitionSeed": [],
            "preOriginClaims": [],
        },
    )
    assert response["status"] == "error"
    assert response["diagnostics"]["errors"][0] == {
        "category": "unreachable",
        "code": "illegal-move",
        "eventSeq": 1,
        "message": "non-adjacent movement is not legal",
    }


def test_project_legal_actions_enumerates_flying_moves() -> None:
    current = (
        "MFEN/1.0 mill24-state-v1 WWW...../BBB...../........ "
        "w m m 0,0 - 0 18 -"
    )
    document = _legal_actions(current)
    occupied = {"a7", "d7", "g7", "b6", "d6", "f6"}
    empty = [point for point in POINTS if point not in occupied]
    assert document["actions"] == [
        {
            "actor": "w",
            "type": "move",
            "from": source,
            "to": destination,
        }
        for source in ("a7", "d7", "g7")
        for destination in empty
    ]


def test_project_legal_actions_enumerates_pending_remove() -> None:
    current = (
        "MFEN/1.0 mill24-state-v1 BBB...../......../W....... "
        "b p r 8,6 b:mill:b:w:1:010000:w 0 4 -"
    )
    document = _legal_actions(current)
    assert document["actions"] == [
        {
            "actor": "b",
            "type": "remove",
            "target": {"zone": "board", "at": "c5"},
        }
    ]


def test_project_legal_actions_terminal_is_empty() -> None:
    manifest = example_manifest()
    manifest["pieces"] = {"black": 13, "minimumLive": 3, "white": 13}
    current = (
        "MFEN/1.0 mill24-state-v1 WBWBWBWB/BWBWBWBW/WBWBWBWB "
        "- o o 1,1 - 0 0 b:no-legal-primary-action"
    )
    assert _legal_actions(current, manifest)["actions"] == []


def test_project_legal_actions_rejects_unstabilized_boundary() -> None:
    manifest = deepcopy(example_manifest())
    manifest["pieces"] = {"black": 13, "minimumLive": 3, "white": 13}
    response = _request(
        "project-legal-actions",
        {
            "manifest": manifest,
            "current": (
                "MFEN/1.0 mill24-state-v1 WBWBWBWB/BWBWBWBW/WBWBWBWB "
                "w p p 1,1 - 0 0 -"
            ),
        },
    )
    assert response["status"] == "error"
    error = response["diagnostics"]["errors"][0]
    assert (error["category"], error["code"]) == (
        "inconsistent",
        "unstabilized-boundary",
    )

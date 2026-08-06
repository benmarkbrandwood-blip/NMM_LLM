"""Pinned MIF 1.0 candidate vectors used by independent adapter tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MIF_COMMIT = "f37ddfeb5fb8479991fa38eeb03c797bef8ae408"
SEMANTIC_DIGEST = "sha256:224f7e368e322a4cc8c1225a025fb548d5b41eb096d34b7ae0543182d1aa9393"
DOCUMENT_DIGEST = "sha256:62479b6f40efb8ab478bab3d2b725647213604fcd3cc9cd4c1f69357535ae257"
ORIGIN_SEMANTIC_DIGEST = "sha256:173caf8189defd1ab7d4a3e8b9e26688a07fd77976bf09d56bff5fe0c273e1a1"
ORIGIN_DOCUMENT_DIGEST = "sha256:9e8a7aa8f71fe2d8cc4d0d3bc5571f2c09e21f98b12b336b691f1cdbe5bb2833"
EMPTY_ORIGIN = "MFEN/1.0 mill24-state-v1 ......../......../........ b p p 9,9 - 0 0 -"


def example_manifest() -> dict[str, Any]:
    return {
        "boardFull": {"action": "disabled"},
        "captures": {
            "custodian": {
                "enabled": False,
                "lines": {"cross": False, "diagonal": False, "squareEdges": False},
                "maximumOwnLivePieces": None,
                "phases": ["moving"],
            },
            "intervention": {
                "enabled": False,
                "lines": {"cross": False, "diagonal": False, "squareEdges": False},
                "maximumOwnLivePieces": None,
                "phases": ["moving"],
            },
            "leap": {
                "enabled": False,
                "lines": {"cross": False, "diagonal": False, "squareEdges": False},
                "maximumOwnLivePieces": None,
                "phases": ["moving"],
            },
            "resolution": "target-commits-v1",
        },
        "draw": {
            "claimRights": {"profile": "stable-claim-rights-v1"},
            "noProgress": {
                "countedPrimaryActions": ["move"],
                "endgameLimit": 0,
                "endgamePredicate": "none",
                "evaluationBoundary": "stable-after-primary-sequence-v1",
                "mode": "automatic",
                "normalLimit": 0,
                "resetEvents": ["board-remove"],
            },
            "offers": {"expiry": "explicit-only"},
            "repetition": {
                "count": 3,
                "mode": "claim",
                "observation": "stable-primary-decision-v1",
                "projection": "repetition-observation-v1",
                "resetEvents": ["board-remove"],
                "summary": "reset-count-smt-v1",
            },
        },
        "flying": {"enabled": True, "maximumLive": 3},
        "format": "MRS/1.0",
        "id": "example-morris",
        "mills": {
            "delayedClearBoundary": "on-enter-moving-v1",
            "lineReuse": "unlimited",
            "movingEffect": "remove-opponent-board",
            "placingEffect": "remove-opponent-board",
            "removalMultiplicity": "one-per-primary",
            "reverseReformation": "allowed",
            "targetProtection": "outside-mill-first",
        },
        "pieces": {"black": 9, "minimumLive": 3, "white": 9},
        "placing": {
            "earlyStop": {"boundary": "after-unobligated-place-v1", "emptyPoints": 0},
            "movementAllowed": False,
            "noLegalPrimaryAction": "loss",
        },
        "semanticState": [],
        "semanticsProfile": "mif-finite-rules-v3",
        "stalemate": {"action": "loss", "boardRemovalTargets": "adjacent-opponent"},
        "status": "fixture",
        "title": "Example Morris",
        "topology": "mill24-orthogonal-v1",
        "turn": {"initial": "b", "placingEndActivePlayer": "retain"},
        "version": 1,
    }


def origin_manifest() -> dict[str, Any]:
    manifest = example_manifest()
    manifest["id"] = "x-origin-stabilization"
    manifest["title"] = "Origin stabilization executable fixture"
    manifest["pieces"] = {"black": 12, "minimumLive": 3, "white": 12}
    manifest["boardFull"] = {"action": "white-then-black-remove"}
    return manifest


def empty_observation(*, semantic_digest: str = SEMANTIC_DIGEST) -> dict[str, Any]:
    return {
        "action": "p",
        "board": "......../......../........",
        "hands": [9, 9],
        "phase": "p",
        "profile": "repetition-observation-v1",
        "semantic": {},
        "semanticDigest": semantic_digest,
        "side": "b",
        "stateProfile": "mill24-state-v1",
    }


def offer_r1_mstate(*, portable: bool = False) -> dict[str, Any]:
    if portable:
        ruleset = {
            "mode": "portable",
            "id": "example-morris",
            "version": 1,
            "semanticDigest": SEMANTIC_DIGEST,
            "documentDigest": DOCUMENT_DIGEST,
            "manifest": example_manifest(),
        }
    else:
        ruleset = {
            "mode": "reference",
            "id": "example-morris",
            "version": 1,
            "semanticDigest": SEMANTIC_DIGEST,
            "documentDigest": DOCUMENT_DIGEST,
        }
    return {
        "format": "MSTATE/1.0",
        "positionFormat": "MFEN/1.0",
        "stateProfile": "mill24-state-v1",
        "ruleset": ruleset,
        "origin": EMPTY_ORIGIN,
        "events": [{"actor": "b", "seq": 1, "type": "offer-draw"}],
        "current": EMPTY_ORIGIN,
        "repetitionHistory": [{"source": "origin", "key": empty_observation()}],
        "preOriginClaims": [],
        "claims": [
            {
                "source": "event",
                "actor": "b",
                "eventSeq": 1,
                "kind": "draw-offer",
                "status": "open",
            }
        ],
    }


def mill_removal_events() -> list[dict[str, Any]]:
    return [
        {"actor": "b", "at": "a7", "seq": 1, "type": "place"},
        {"actor": "w", "at": "d7", "seq": 2, "type": "place"},
        {"actor": "b", "at": "a4", "seq": 3, "type": "place"},
        {"actor": "w", "at": "d6", "seq": 4, "type": "place"},
        {"actor": "b", "at": "a1", "seq": 5, "type": "place"},
        {
            "actor": "b",
            "seq": 6,
            "target": {"at": "d7", "zone": "board"},
            "type": "remove",
        },
    ]


def mill_removal_mstate() -> dict[str, Any]:
    return {
        "format": "MSTATE/1.0",
        "positionFormat": "MFEN/1.0",
        "stateProfile": "mill24-state-v1",
        "ruleset": {
            "mode": "reference",
            "id": "example-morris",
            "version": 1,
            "semanticDigest": SEMANTIC_DIGEST,
            "documentDigest": DOCUMENT_DIGEST,
        },
        "origin": EMPTY_ORIGIN,
        "events": mill_removal_events(),
        "current": (
            "MFEN/1.0 mill24-state-v1 B.....BB/.W....../........ "
            "w p p 7,6 - 0 5 -"
        ),
        "repetitionHistory": [
            {
                "source": "event",
                "eventSeq": 6,
                "key": {
                    "profile": "repetition-observation-v1",
                    "stateProfile": "mill24-state-v1",
                    "semanticDigest": SEMANTIC_DIGEST,
                    "board": "B.....BB/.W....../........",
                    "side": "w",
                    "phase": "p",
                    "action": "p",
                    "hands": [7, 6],
                    "semantic": {},
                },
            }
        ],
        "preOriginClaims": [],
        "claims": [],
    }


def origin_mstate() -> dict[str, Any]:
    return {
        "format": "MSTATE/1.0",
        "positionFormat": "MFEN/1.0",
        "stateProfile": "mill24-state-v1",
        "ruleset": {
            "mode": "reference",
            "id": "x-origin-stabilization",
            "version": 1,
            "semanticDigest": ORIGIN_SEMANTIC_DIGEST,
            "documentDigest": ORIGIN_DOCUMENT_DIGEST,
        },
        "origin": "MFEN/1.0 mill24-state-v1 WBWBWBWB/BWBWBWBW/WBWBWBWB w m m 0,0 - 0 24 -",
        "events": [
            {
                "actor": "w",
                "seq": 1,
                "target": {"at": "d7", "zone": "board"},
                "type": "remove",
            },
            {
                "actor": "b",
                "seq": 2,
                "target": {"at": "a7", "zone": "board"},
                "type": "remove",
            },
        ],
        "current": "MFEN/1.0 mill24-state-v1 ..WBWBWB/BWBWBWBW/WBWBWBWB w m m 0,0 - 0 24 -",
        "repetitionHistory": [
            {
                "source": "event",
                "eventSeq": 2,
                "key": {
                    "profile": "repetition-observation-v1",
                    "stateProfile": "mill24-state-v1",
                    "semanticDigest": ORIGIN_SEMANTIC_DIGEST,
                    "board": "..WBWBWB/BWBWBWBW/WBWBWBWB",
                    "side": "w",
                    "phase": "m",
                    "action": "m",
                    "hands": [0, 0],
                    "semantic": {},
                },
            }
        ],
        "preOriginClaims": [],
        "claims": [],
    }


def clone(value: Any) -> Any:
    return deepcopy(value)

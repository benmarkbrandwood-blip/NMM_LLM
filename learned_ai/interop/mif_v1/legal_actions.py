"""Independent legal gameplay-action projection for MIF interoperability."""

from __future__ import annotations

from typing import Any, Mapping

from .common import POINT_INDEX, POINTS, fail, piece_for, require_object
from .model import MifState, ResolvedManifest, resolve_manifest, validate_state


def _movement_actions(
    state: MifState,
    manifest: ResolvedManifest,
) -> list[dict[str, str]]:
    actor = state.side
    live_piece = piece_for(actor)
    sources = [
        point
        for point in POINTS
        if state.board[POINT_INDEX[point]] == live_piece
    ]
    destinations = state.empty_points()
    flying = manifest.manifest["flying"]
    may_fly = (
        state.phase == "m"
        and flying["enabled"]
        and len(sources) <= flying["maximumLive"]
    )
    edges = {frozenset(edge) for edge in manifest.edges}
    return [
        {"actor": actor, "type": "move", "from": source, "to": destination}
        for source in sources
        for destination in destinations
        if may_fly or frozenset((source, destination)) in edges
    ]


def _primary_actions(
    state: MifState,
    manifest: ResolvedManifest,
) -> list[dict[str, str]]:
    actor = state.side
    hand_index = 0 if actor == "w" else 1
    actions: list[dict[str, str]] = []
    if state.action == "p" and state.hands[hand_index] > 0:
        actions.extend(
            {"actor": actor, "type": "place", "at": point}
            for point in state.empty_points()
        )
    if state.action == "m" or (
        state.action == "p" and manifest.manifest["placing"]["movementAllowed"]
    ):
        actions.extend(_movement_actions(state, manifest))
    return actions


def _remove_actions(state: MifState) -> list[dict[str, Any]]:
    board_actions: dict[tuple[str, str], dict[str, Any]] = {}
    hand_actions: dict[tuple[str, str], dict[str, Any]] = {}
    for branch in state.obligations:
        head = branch[0]
        if head.zone == "b":
            if not isinstance(head.targets, int):
                fail(
                    "inconsistent",
                    "obligation-target-mismatch",
                    "board removal has no concrete targets",
                )
            for index, point in enumerate(POINTS):
                if not head.targets & (1 << index):
                    continue
                if state.board[index] != piece_for(head.target_owner):
                    fail(
                        "inconsistent",
                        "obligation-target-mismatch",
                        "board removal target owner does not match",
                    )
                board_actions[(head.actor, point)] = {
                    "actor": head.actor,
                    "type": "remove",
                    "target": {"zone": "board", "at": point},
                }
        else:
            owner_index = 0 if head.target_owner == "w" else 1
            if state.hands[owner_index] == 0:
                fail(
                    "inconsistent",
                    "obligation-target-mismatch",
                    "hand removal target is empty",
                )
            hand_actions[(head.actor, head.target_owner)] = {
                "actor": head.actor,
                "type": "remove",
                "target": {"zone": "hand", "player": head.target_owner},
            }
    board = [
        board_actions[key]
        for key in sorted(
            board_actions,
            key=lambda item: (POINT_INDEX[item[1]], item[0]),
        )
    ]
    player_order = {"w": 0, "b": 1}
    hand = [
        hand_actions[key]
        for key in sorted(
            hand_actions,
            key=lambda item: (player_order[item[1]], item[0]),
        )
    ]
    return board + hand


def _no_progress_limit(state: MifState, manifest: ResolvedManifest) -> int:
    config = manifest.manifest["draw"]["noProgress"]
    if config["endgamePredicate"] == "either-player-live-equals-3" and (
        state.live_count("w") == 3 or state.live_count("b") == 3
    ):
        return config["endgameLimit"]
    return config["normalLimit"]


def _require_stable_primary_boundary(
    state: MifState,
    manifest: ResolvedManifest,
) -> list[dict[str, str]]:
    empty_count = len(state.empty_points())
    early_stop = manifest.manifest["placing"]["earlyStop"]["emptyPoints"]
    if state.phase == "p" and (
        state.hands == [0, 0]
        or (early_stop > 0 and empty_count <= early_stop)
    ):
        fail(
            "inconsistent",
            "unstabilized-boundary",
            "placing boundary has not been stabilized",
        )
    hand_index = 0 if state.side == "w" else 1
    expected_action = "p" if state.hands[hand_index] > 0 else "m"
    if state.action != expected_action:
        fail(
            "inconsistent",
            "unstabilized-boundary",
            "active phase has not been stabilized",
        )
    minimum = manifest.manifest["pieces"]["minimumLive"]
    if any(state.material_count(player) < minimum for player in ("w", "b")):
        fail(
            "inconsistent",
            "unstabilized-boundary",
            "material terminal has not been stabilized",
        )
    if empty_count == 0:
        fail(
            "inconsistent",
            "unstabilized-boundary",
            "board-full boundary has not been stabilized",
        )
    no_progress = manifest.manifest["draw"]["noProgress"]
    limit = _no_progress_limit(state, manifest)
    if (
        limit
        and no_progress["mode"] == "automatic"
        and state.no_progress >= limit
    ):
        fail(
            "inconsistent",
            "unstabilized-boundary",
            "automatic no-progress terminal has not been stabilized",
        )
    actions = _primary_actions(state, manifest)
    if not actions:
        fail(
            "inconsistent",
            "unstabilized-boundary",
            "no-legal-action boundary has not been stabilized",
        )
    return actions


def project_legal_actions(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project every legal gameplay action without applying a transition."""

    value = require_object(
        payload,
        required={"manifest", "current"},
        context="project-legal-actions payload",
    )
    manifest = resolve_manifest(value["manifest"])
    state = MifState.parse(value["current"])
    validate_state(state, manifest)
    current = state.serialize()
    if current != value["current"]:
        fail("canonical", "extension-order", "current MFEN is not canonical")

    if state.outcome != "-":
        actions: list[dict[str, Any]] = []
    elif state.obligations:
        actions = _remove_actions(state)
        if not actions:
            fail(
                "inconsistent",
                "unstabilized-boundary",
                "pending obligation has no legal action",
            )
    else:
        actions = _require_stable_primary_boundary(state, manifest)

    return {
        "document": {
            "profile": "legal-actions-v1",
            "stateProfile": "mill24-state-v1",
            "semanticDigest": manifest.semantic_digest,
            "current": current,
            "actions": actions,
        }
    }

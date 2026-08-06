"""Independent finite-rules executor and replay identities for MIF 1.0."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from .common import (
    MAX_EVENTS,
    MAX_REPETITION_ENTRIES,
    POINT_INDEX,
    POINTS,
    deep_copy_json,
    enforce_resource_limit,
    fail,
    jcs_bytes,
    other,
    piece_for,
    require_object,
    sha256_digest,
)
from .model import (
    MifState,
    Obligation,
    ResolvedManifest,
    resolve_manifest,
    resolve_ruleset_envelope,
    validate_state,
)


EVENT_COMMON = {"seq", "actor", "type", "annotations", "extensions"}
EVENT_KEYS = {
    "place": EVENT_COMMON | {"at", "interventionLine"},
    "move": EVENT_COMMON | {"from", "to", "interventionLine"},
    "remove": EVENT_COMMON | {"target"},
    "offer-draw": EVENT_COMMON,
    "accept-draw": EVENT_COMMON | {"offerEventSeq"},
    "decline-draw": EVENT_COMMON | {"offerEventSeq"},
    "withdraw-draw": EVENT_COMMON | {"offerEventSeq"},
    "claim-draw": EVENT_COMMON | {"reason"},
    "resign": EVENT_COMMON,
    "adjudicate": EVENT_COMMON | {"result", "reason", "authority"},
}
EVENT_REQUIRED = {
    "place": {"seq", "actor", "type", "at"},
    "move": {"seq", "actor", "type", "from", "to"},
    "remove": {"seq", "actor", "type", "target"},
    "offer-draw": {"seq", "actor", "type"},
    "accept-draw": {"seq", "actor", "type", "offerEventSeq"},
    "decline-draw": {"seq", "actor", "type", "offerEventSeq"},
    "withdraw-draw": {"seq", "actor", "type", "offerEventSeq"},
    "claim-draw": {"seq", "actor", "type", "reason"},
    "resign": {"seq", "actor", "type"},
    "adjudicate": {"seq", "actor", "type", "result", "reason", "authority"},
}


def _semantic_state(state: MifState) -> dict[str, str]:
    return dict(sorted(state.extensions.items()))


def repetition_observation(
    state: MifState,
    manifest: ResolvedManifest,
) -> dict[str, Any]:
    if state.outcome != "-" or state.obligations or state.action not in {"p", "m"}:
        fail("ineligible", "unstabilized-boundary", "state is not observable")
    return {
        "profile": "repetition-observation-v1",
        "stateProfile": "mill24-state-v1",
        "semanticDigest": manifest.semantic_digest,
        "board": state.board_field,
        "side": state.side,
        "phase": state.phase,
        "action": state.action,
        "hands": list(state.hands),
        "semantic": _semantic_state(state),
    }


def _empty_hashes() -> list[bytes]:
    hashes = [hashlib.sha256(b"\x00").digest()]
    for _ in range(256):
        previous = hashes[-1]
        hashes.append(hashlib.sha256(b"\x02" + previous + previous).digest())
    return hashes


EMPTY_HASHES = _empty_hashes()


def repetition_root(
    history: list[dict[str, Any]],
    threshold: int,
) -> str:
    if threshold == 0:
        fail("ineligible", "unsupported-profile", "disabled repetition has no root")
    observations: dict[bytes, tuple[bytes, int]] = {}
    for entry in history:
        key = entry["key"]
        canonical = jcs_bytes(key)
        digest = hashlib.sha256(canonical).digest()
        previous = observations.get(digest)
        if previous is not None and previous[0] != canonical:
            fail(
                "integrity",
                "repetition-observation-digest-collision",
                "unequal observations share a digest",
            )
        count = 1 if previous is None else previous[1] + 1
        observations[digest] = (canonical, min(count, threshold))
    nodes: dict[int, bytes] = {}
    for digest, (_, count) in observations.items():
        key = int.from_bytes(digest, "big")
        nodes[key] = hashlib.sha256(
            b"\x01" + digest + count.to_bytes(8, "big")
        ).digest()
    for height in range(256):
        parents: dict[int, list[bytes | None]] = {}
        for key, node_hash in nodes.items():
            parent = key >> 1
            pair = parents.setdefault(parent, [None, None])
            pair[key & 1] = node_hash
        nodes = {
            parent: hashlib.sha256(
                b"\x02"
                + (pair[0] if pair[0] is not None else EMPTY_HASHES[height])
                + (pair[1] if pair[1] is not None else EMPTY_HASHES[height])
            ).digest()
            for parent, pair in parents.items()
        }
    root = nodes.get(0, EMPTY_HASHES[256])
    return "sha256:" + root.hex()


def _validate_observation(value: Any, manifest: ResolvedManifest) -> dict[str, Any]:
    observation = require_object(
        value,
        required={
            "profile",
            "stateProfile",
            "semanticDigest",
            "board",
            "side",
            "phase",
            "action",
            "hands",
            "semantic",
        },
        context="repetition observation",
    )
    if (
        observation["profile"] != "repetition-observation-v1"
        or observation["stateProfile"] != "mill24-state-v1"
    ):
        fail("unsupported", "unsupported-profile", "unsupported repetition projection")
    if observation["semanticDigest"] != manifest.semantic_digest:
        fail("integrity", "semantic-digest-mismatch")
    return deep_copy_json(observation)


def _seed_repetition(value: Any, manifest: ResolvedManifest) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        fail("syntax", "array-required", "repetitionSeed must be an array")
    enforce_resource_limit(
        "repetition-entries",
        len(value),
        MAX_REPETITION_ENTRIES,
    )
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping) and "key" in item:
            entry = require_object(
                item,
                required={"source", "key"},
                optional={"eventSeq"},
                context="repetition seed entry",
            )
            if entry["source"] != "pre-origin" or "eventSeq" in entry:
                fail("inconsistent", "repetition-history-mismatch", "seed is not pre-origin")
            key = _validate_observation(entry["key"], manifest)
        else:
            key = _validate_observation(item, manifest)
        result.append({"source": "pre-origin", "key": key})
    return result


def _seed_claims(value: Any) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not isinstance(value, list):
        fail("syntax", "array-required", "preOriginClaims must be an array")
    claims: list[dict[str, Any]] = []
    open_offer: dict[str, Any] | None = None
    for item in value:
        claim = require_object(
            item,
            required={"actor", "kind", "status"},
            context="pre-origin claim",
        )
        if (
            claim["actor"] not in {"w", "b"}
            or claim["kind"] != "draw-offer"
            or claim["status"] not in {"open", "accepted", "declined", "withdrawn", "expired"}
        ):
            fail("syntax", "invalid-claim-audit", "invalid pre-origin claim")
        audit = {
            "source": "pre-origin",
            "actor": claim["actor"],
            "kind": "draw-offer",
            "status": claim["status"],
        }
        claims.append(audit)
        if claim["status"] == "open":
            if open_offer is not None:
                fail("inconsistent", "claims-mismatch", "more than one offer is open")
            open_offer = {
                "source": "pre-origin",
                "actor": claim["actor"],
                "offerEventSeq": 0,
            }
    return claims, open_offer


def _event_required(event: Any, expected_seq: int) -> Mapping[str, Any]:
    if not isinstance(event, Mapping):
        fail("syntax", "object-required", "event must be an object", event_seq=expected_seq)
    event_type = event.get("type")
    if event_type not in EVENT_KEYS:
        fail("unsupported", "unsupported-event", "unsupported event type", event_seq=expected_seq)
    require_object(
        event,
        required=EVENT_REQUIRED[event_type],
        optional=EVENT_KEYS[event_type] - EVENT_REQUIRED[event_type],
        context=f"event {expected_seq}",
    )
    if event.get("seq") != expected_seq:
        fail("inconsistent", "event-sequence-mismatch", "event sequences must start at one", event_seq=expected_seq)
    if "extensions" in event:
        fail("unsupported", "unsupported-profile", "event extensions are not implemented", event_seq=expected_seq)
    return event


@dataclass
class Execution:
    manifest: ResolvedManifest
    origin: str
    events: list[dict[str, Any]]
    pre_origin_claims: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    state: MifState
    repetition_history: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    claim_rights: dict[str, Any] | None
    open_offer: dict[str, Any] | None

    @property
    def result(self) -> dict[str, Any]:
        return {"trace": self.trace, "final": self.trace[-1]}


class FiniteRulesExecutor:
    """Stateful execution of the supported frozen corpus rulesets."""

    def __init__(
        self,
        manifest: ResolvedManifest,
        origin: str,
        repetition_seed: Any,
        pre_origin_claims: Any,
    ) -> None:
        self.manifest = manifest
        self.state = MifState.parse(origin)
        validate_state(self.state, manifest)
        self.origin = self.state.serialize()
        self.repetition_history = _seed_repetition(repetition_seed, manifest)
        self.pre_origin_claims = deep_copy_json(pre_origin_claims)
        self.claims, self.open_offer = _seed_claims(pre_origin_claims)
        self.claim_rights: dict[str, Any] | None = None
        self.trace: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._origin_had_obligation = bool(self.state.obligations)
        self._origin_generated_obligation = False

    def run(self, events: Any) -> Execution:
        if not isinstance(events, list):
            fail("syntax", "array-required", "events must be an array")
        enforce_resource_limit("events", len(events), MAX_EVENTS)
        had_obligation = bool(self.state.obligations)
        if not had_obligation:
            self._stabilize(source="origin", event_seq=None)
        self._origin_generated_obligation = not had_obligation and bool(self.state.obligations)
        self.trace.append(self._snapshot("origin", None))
        for expected_seq, raw_event in enumerate(events, start=1):
            event = _event_required(raw_event, expected_seq)
            self._apply_event(event)
            copied = deep_copy_json(event)
            self.events.append(copied)
            self.trace.append(self._snapshot("event", expected_seq))
        return Execution(
            manifest=self.manifest,
            origin=self.origin,
            events=self.events,
            pre_origin_claims=deep_copy_json(self.pre_origin_claims),
            trace=self.trace,
            state=self.state.clone(),
            repetition_history=deep_copy_json(self.repetition_history),
            claims=deep_copy_json(self.claims),
            claim_rights=deep_copy_json(self.claim_rights),
            open_offer=deep_copy_json(self.open_offer),
        )

    def _snapshot(self, boundary: str, event_seq: int | None) -> dict[str, Any]:
        decision = self._decision_state()
        return {
            "boundary": boundary,
            "eventSeq": event_seq,
            "current": self.state.serialize(),
            "repetitionHistory": deep_copy_json(self.repetition_history),
            "claims": deep_copy_json(self.claims),
            "claimRights": deep_copy_json(self.claim_rights),
            "decisionState": decision,
            "decisionDigest": sha256_digest(decision),
        }

    def _decision_state(self) -> dict[str, Any]:
        repetition = self.manifest.manifest["draw"]["repetition"]
        if repetition["count"]:
            repetition_summary: dict[str, Any] | None = {
                "profile": "reset-count-smt-v1",
                "root": repetition_root(self.repetition_history, repetition["count"]),
            }
        else:
            repetition_summary = None
        no_progress = self.manifest.manifest["draw"]["noProgress"]
        limits = [
            value
            for value in (no_progress["normalLimit"], no_progress["endgameLimit"])
            if value
        ]
        normalized_progress = min(self.state.no_progress, max(limits)) if limits else None
        semantic_offer = None
        if self.open_offer is not None and self.state.outcome == "-":
            offerer = self.open_offer["actor"]
            non_offerer = other(offerer)
            operations = [
                {"actor": offerer, "action": "withdraw"},
                {"actor": non_offerer, "action": "accept"},
                {"actor": non_offerer, "action": "decline"},
            ]
            player_order = {"w": 0, "b": 1}
            action_order = {"accept": 0, "decline": 1, "withdraw": 2}
            operations.sort(key=lambda item: (player_order[item["actor"]], action_order[item["action"]]))
            semantic_offer = {"offerer": offerer, "available": operations}
        return {
            "profile": "decision-state-v1",
            "stateProfile": "mill24-state-v1",
            "semanticDigest": self.manifest.semantic_digest,
            "board": self.state.board_field,
            "side": self.state.side,
            "phase": self.state.phase,
            "action": self.state.action,
            "hands": list(self.state.hands),
            "obligations": self.state.obligations_field,
            "noProgress": normalized_progress,
            "outcome": self.state.outcome,
            "semantic": _semantic_state(self.state),
            "repetitionSummary": repetition_summary,
            "openOffer": semantic_offer,
            "claimRights": deep_copy_json(self.claim_rights),
        }

    def _observation_applies(self) -> bool:
        repetition = self.manifest.manifest["draw"]["repetition"]
        if not repetition["count"] or self.state.outcome != "-" or self.state.obligations:
            return False
        if repetition["observation"] == "stable-primary-decision-v1":
            return self.state.action in {"p", "m"}
        return self.state.phase == "m" and self.state.action == "m"

    def _append_observation(self, source: str, event_seq: int | None) -> int:
        if not self._observation_applies():
            return 0
        key = repetition_observation(self.state, self.manifest)
        entry: dict[str, Any] = {"source": source, "key": key}
        if source == "event":
            if event_seq is None:
                fail("inconsistent", "repetition-history-mismatch")
            entry["eventSeq"] = event_seq
        enforce_resource_limit(
            "repetition-entries",
            len(self.repetition_history) + 1,
            MAX_REPETITION_ENTRIES,
        )
        self.repetition_history.append(entry)
        canonical = jcs_bytes(key)
        return sum(1 for item in self.repetition_history if jcs_bytes(item["key"]) == canonical)

    def _expire_claim_right(self) -> None:
        self.claim_rights = None

    def _expire_offer(self, event_seq: int | None) -> None:
        if self.open_offer is None:
            return
        source = self.open_offer["source"]
        source_seq = self.open_offer["offerEventSeq"]
        for claim in self.claims:
            if claim["source"] != source:
                continue
            if source == "event" and claim.get("eventSeq") != source_seq:
                continue
            if claim["status"] == "open":
                claim["status"] = "expired"
                if event_seq is not None:
                    claim["resolvedEventSeq"] = event_seq
                break
        self.open_offer = None

    def _terminal(self, result: str, reason: str, event_seq: int | None = None) -> None:
        self._expire_claim_right()
        self._expire_offer(event_seq)
        self.state.side = "-"
        self.state.phase = "o"
        self.state.action = "o"
        self.state.obligations = []
        self.state.outcome = f"{result}:{reason}"

    def _set_primary_action(self) -> None:
        if self.state.outcome != "-" or self.state.obligations:
            return
        hand_index = 0 if self.state.side == "w" else 1
        self.state.phase = "p" if self.state.hands[hand_index] > 0 else "m"
        self.state.action = self.state.phase

    def _global_placing_boundary(self) -> bool:
        if self.state.phase == "m" and self.state.hands == [0, 0]:
            return False
        early = self.manifest.manifest["placing"]["earlyStop"]["emptyPoints"]
        boundary = self.state.hands == [0, 0] or (
            early > 0 and len(self.state.empty_points()) <= early
        )
        if not boundary:
            return False
        self.state.hands = [0, 0]
        self.state.board = ["." if piece in {"w", "b"} else piece for piece in self.state.board]
        selected = self.manifest.manifest["turn"]["placingEndActivePlayer"]
        if selected in {"w", "b"}:
            self.state.side = selected
        self.state.phase = "m"
        self.state.action = "m"
        return True

    def _target_mask(self, player: str, *, protect_mills: bool) -> int:
        live = [index for index, piece in enumerate(self.state.board) if piece == piece_for(player)]
        if protect_mills and live:
            protected: set[int] = set()
            for line in self.manifest.lines:
                if all(self.state.board[POINT_INDEX[point]] == piece_for(player) for point in line):
                    protected.update(POINT_INDEX[point] for point in line)
            outside = [index for index in live if index not in protected]
            if outside:
                live = outside
        return sum(1 << index for index in live)

    def _create_board_full_effect(self) -> None:
        action = self.manifest.manifest["boardFull"]["action"]
        if action == "disabled":
            return
        if action == "white-loses":
            self._terminal("b", "board-full")
            return
        if action == "draw":
            self._terminal("d", "board-full")
            return
        if action == "white-then-black-remove":
            first, second, after = "w", "b", "w"
        elif action == "black-then-white-remove":
            first, second, after = "b", "w", "b"
        elif action == "active-player-removes":
            first = self.state.side
            second = None
            after = other(first)
        else:
            fail("unsupported", "unsupported-profile", "board-full action is unsupported")
        first_item = Obligation(
            first,
            "board-full",
            "b",
            other(first),
            1,
            self._target_mask(other(first), protect_mills=False),
            "q" if second is not None else after,
        )
        branch = [first_item]
        if second is not None:
            branch.append(
                Obligation(second, "board-full", "b", other(second), 1, "~", after)
            )
        if not isinstance(first_item.targets, int) or first_item.targets == 0:
            fail("unreachable", "obligation-target-mismatch", "board-full has no target")
        self.state.obligations = [branch]
        self.state.side = first
        self.state.action = "r"

    def _minimum_material(self, event_seq: int | None = None) -> bool:
        minimum = self.manifest.manifest["pieces"]["minimumLive"]
        deficient = [player for player in ("w", "b") if self.state.material_count(player) < minimum]
        if not deficient:
            return False
        if len(deficient) == 2:
            self._terminal("d", "fewer-than-minimum", event_seq)
        else:
            self._terminal(other(deficient[0]), "fewer-than-minimum", event_seq)
        return True

    def _is_adjacent(self, source: str, destination: str) -> bool:
        edge = frozenset((source, destination))
        return any(frozenset(item) == edge for item in self.manifest.edges)

    def _legal_move_exists(self, player: str) -> bool:
        empty = self.state.empty_points()
        if not empty:
            return False
        live = [POINTS[index] for index, piece in enumerate(self.state.board) if piece == piece_for(player)]
        flying = self.manifest.manifest["flying"]
        if flying["enabled"] and len(live) <= flying["maximumLive"]:
            return bool(live)
        return any(self._is_adjacent(source, destination) for source in live for destination in empty)

    def _legal_primary_exists(self) -> bool:
        player = self.state.side
        hand_index = 0 if player == "w" else 1
        if self.state.phase == "p":
            if self.state.hands[hand_index] > 0 and self.state.empty_points():
                return True
            return bool(self.manifest.manifest["placing"]["movementAllowed"] and self._legal_move_exists(player))
        return self._legal_move_exists(player)

    def _no_progress_limit(self) -> int:
        config = self.manifest.manifest["draw"]["noProgress"]
        if config["endgamePredicate"] == "either-player-live-equals-3" and (
            self.state.live_count("w") == 3 or self.state.live_count("b") == 3
        ):
            return config["endgameLimit"]
        return config["normalLimit"]

    def _derive_claim_rights(self, repetition_count: int) -> None:
        reasons: list[str] = []
        no_progress = self.manifest.manifest["draw"]["noProgress"]
        limit = self._no_progress_limit()
        if limit and no_progress["mode"] == "claim" and self.state.no_progress >= limit:
            reasons.append("no-progress")
        repetition = self.manifest.manifest["draw"]["repetition"]
        if repetition["count"] and repetition["mode"] == "claim" and repetition_count >= repetition["count"]:
            reasons.append("repetition")
        self.claim_rights = {"actor": self.state.side, "reasons": reasons} if reasons else None

    def _stabilize(self, source: str, event_seq: int | None) -> None:
        if self.state.outcome != "-" or self.state.obligations:
            return
        self._global_placing_boundary()
        if not self.state.empty_points():
            self._create_board_full_effect()
            if self.state.outcome != "-" or self.state.obligations:
                return
        if self._minimum_material(event_seq):
            return
        self._set_primary_action()
        if self.state.phase == "p" and not self._legal_primary_exists():
            policy = self.manifest.manifest["placing"]["noLegalPrimaryAction"]
            if policy == "apply-board-full":
                self._create_board_full_effect()
            elif policy == "loss":
                self._terminal(other(self.state.side), "no-legal-primary-action", event_seq)
            else:
                self._terminal("d", "no-legal-primary-action", event_seq)
            return
        if self.state.phase == "m" and not self._legal_primary_exists():
            action = self.manifest.manifest["stalemate"]["action"]
            if action == "loss":
                self._terminal(other(self.state.side), "no-legal-move", event_seq)
            else:
                self._terminal("d", "no-legal-move", event_seq)
            return
        repetition_count = self._append_observation(source, event_seq)
        repetition = self.manifest.manifest["draw"]["repetition"]
        if repetition["count"] and repetition["mode"] == "automatic" and repetition_count >= repetition["count"]:
            self._terminal("d", "repetition", event_seq)
            return
        no_progress = self.manifest.manifest["draw"]["noProgress"]
        limit = self._no_progress_limit()
        if limit and no_progress["mode"] == "automatic" and self.state.no_progress >= limit:
            self._terminal("d", "no-progress", event_seq)
            return
        self._derive_claim_rights(repetition_count)
        self._set_primary_action()

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        if self.state.outcome != "-" and event["type"] != "adjudicate":
            fail(
                "unreachable",
                "event-after-terminal",
                "no player event is legal after terminal",
                event_seq=event["seq"],
            )
        event_type = event["type"]
        if event_type == "place":
            self._apply_place(event)
        elif event_type == "move":
            self._apply_move(event)
        elif event_type == "remove":
            self._apply_remove(event)
        elif event_type == "offer-draw":
            self._apply_offer(event)
        elif event_type in {"accept-draw", "decline-draw", "withdraw-draw"}:
            self._apply_offer_resolution(event)
        elif event_type == "claim-draw":
            self._apply_claim(event)
        elif event_type == "resign":
            self._apply_resign(event)
        else:
            self._apply_adjudicate(event)

    def _require_primary_actor(self, event: Mapping[str, Any]) -> str:
        actor = event["actor"]
        if actor not in {"w", "b"} or actor != self.state.side:
            fail(
                "unreachable",
                "side-obligation-actor-mismatch",
                "event actor is not the active player",
                event_seq=event["seq"],
            )
        if self.state.obligations or self.state.action not in {"p", "m"}:
            fail(
                "unreachable",
                "unstabilized-boundary",
                "primary action attempted outside a stable boundary",
                event_seq=event["seq"],
            )
        return actor

    def _expire_offer_for_primary(self, actor: str, event_seq: int) -> None:
        if (
            self.open_offer is not None
            and self.open_offer["actor"] != actor
            and self.manifest.manifest["draw"]["offers"]["expiry"]
            == "on-opponent-primary-action"
        ):
            self._expire_offer(event_seq)

    def _new_mill_line_ids(self, destination: str, actor: str) -> list[int]:
        live = piece_for(actor)
        return [
            line_id
            for line_id, line in enumerate(self.manifest.lines)
            if destination in line
            and all(self.state.board[POINT_INDEX[point]] == live for point in line)
        ]

    def _update_no_progress_after_primary(
        self,
        event_type: str,
        *,
        formed_mill: bool,
    ) -> None:
        config = self.manifest.manifest["draw"]["noProgress"]
        if config["normalLimit"] == 0 and config["endgameLimit"] == 0:
            self.state.no_progress = 0
            return
        resets = set(config["resetEvents"])
        if event_type == "place" and "place" in resets:
            self.state.no_progress = 0
            return
        if formed_mill and "mill-formation" in resets:
            self.state.no_progress = 0
            return
        if event_type in config["countedPrimaryActions"]:
            self.state.no_progress += 1

    def _reset_repetition(self, event_name: str) -> None:
        if event_name in self.manifest.manifest["draw"]["repetition"]["resetEvents"]:
            self.repetition_history = []

    def _create_mill_obligation(self, actor: str, formed_lines: list[int]) -> None:
        if not formed_lines:
            return
        opponent = other(actor)
        protect = self.manifest.manifest["mills"]["targetProtection"] == "outside-mill-first"
        target_mask = self._target_mask(opponent, protect_mills=protect)
        target_count = target_mask.bit_count()
        if target_count == 0:
            return
        multiplicity = self.manifest.manifest["mills"]["removalMultiplicity"]
        count = 1 if multiplicity == "one-per-primary" else len(formed_lines)
        count = min(count, target_count)
        self.state.obligations = [
            [Obligation(actor, "mill", "b", opponent, count, target_mask, other(actor))]
        ]
        self.state.side = actor
        self.state.action = "r"

    def _apply_place(self, event: Mapping[str, Any]) -> None:
        actor = self._require_primary_actor(event)
        if self.state.action != "p":
            fail("unreachable", "illegal-place", "state does not accept placement", event_seq=event["seq"])
        at = event["at"]
        if at not in POINT_INDEX:
            fail("syntax", "invalid-coordinate", "invalid placement point", event_seq=event["seq"])
        index = POINT_INDEX[at]
        hand_index = 0 if actor == "w" else 1
        if self.state.hands[hand_index] == 0 or self.state.board[index] != ".":
            fail("unreachable", "illegal-place", "placement is not legal", event_seq=event["seq"])
        self._expire_claim_right()
        self._expire_offer_for_primary(actor, event["seq"])
        self.state.board[index] = piece_for(actor)
        self.state.hands[hand_index] -= 1
        self.state.primary_ply += 1
        formed_lines = self._new_mill_line_ids(at, actor)
        formed_mill = bool(formed_lines)
        self._reset_repetition("place")
        if formed_mill:
            self._reset_repetition("mill-formation")
        self._update_no_progress_after_primary("place", formed_mill=formed_mill)
        self.state.side = other(actor)
        self._create_mill_obligation(actor, formed_lines)
        if not self.state.obligations:
            self._stabilize(source="event", event_seq=event["seq"])

    def _apply_move(self, event: Mapping[str, Any]) -> None:
        actor = self._require_primary_actor(event)
        if self.state.action != "m" and not (
            self.state.action == "p" and self.manifest.manifest["placing"]["movementAllowed"]
        ):
            fail("unreachable", "illegal-move", "state does not accept movement", event_seq=event["seq"])
        source, destination = event["from"], event["to"]
        if source not in POINT_INDEX or destination not in POINT_INDEX:
            fail("syntax", "invalid-coordinate", "invalid movement point", event_seq=event["seq"])
        source_index = POINT_INDEX[source]
        destination_index = POINT_INDEX[destination]
        if (
            self.state.board[source_index] != piece_for(actor)
            or self.state.board[destination_index] != "."
        ):
            fail("unreachable", "illegal-move", "source or destination is illegal", event_seq=event["seq"])
        live = self.state.live_count(actor)
        flying = self.manifest.manifest["flying"]
        may_fly = flying["enabled"] and live <= flying["maximumLive"]
        if not may_fly and not self._is_adjacent(source, destination):
            fail("unreachable", "illegal-move", "non-adjacent movement is not legal", event_seq=event["seq"])
        self._expire_claim_right()
        self._expire_offer_for_primary(actor, event["seq"])
        self.state.board[source_index] = "."
        self.state.board[destination_index] = piece_for(actor)
        self.state.primary_ply += 1
        formed_lines = self._new_mill_line_ids(destination, actor)
        formed_mill = bool(formed_lines)
        if formed_mill:
            self._reset_repetition("mill-formation")
        self._update_no_progress_after_primary("move", formed_mill=formed_mill)
        self.state.side = other(actor)
        self._create_mill_obligation(actor, formed_lines)
        if not self.state.obligations:
            self._stabilize(source="event", event_seq=event["seq"])

    def _promote_head(self, item: Obligation) -> None:
        if item.zone == "b" and item.targets == "~":
            protect = (
                item.cause == "mill"
                and self.manifest.manifest["mills"]["targetProtection"] == "outside-mill-first"
            )
            item.targets = self._target_mask(item.target_owner, protect_mills=protect)
        self.state.side = item.actor
        self.state.action = "r"

    def _matching_obligation_branch(
        self,
        actor: str,
        target: Mapping[str, Any],
    ) -> tuple[int, Obligation]:
        candidates: list[tuple[int, Obligation, str]] = []
        for branch_index, branch in enumerate(self.state.obligations):
            head = branch[0]
            if head.actor != actor:
                continue
            if target.get("zone") == "board" and head.zone == "b":
                at = target.get("at")
                if at in POINT_INDEX and isinstance(head.targets, int) and head.targets & (1 << POINT_INDEX[at]):
                    candidates.append((branch_index, head, branch[0].serialize()))
            elif target.get("zone") == "hand" and head.zone == "h" and target.get("player") == head.target_owner:
                candidates.append((branch_index, head, branch[0].serialize()))
        if not candidates:
            fail(
                "unreachable",
                "obligation-target-mismatch",
                "remove target does not match an obligation",
            )
        candidates.sort(key=lambda item: item[2].encode("ascii"))
        branch_index, head, _ = candidates[0]
        return branch_index, head

    def _apply_remove(self, event: Mapping[str, Any]) -> None:
        if not self.state.obligations:
            fail(
                "unreachable",
                "remove-without-obligation",
                "remove event has no pending obligation",
                event_seq=event["seq"],
            )
        actor = event["actor"]
        if actor not in {"w", "b"} or actor != self.state.side:
            fail(
                "unreachable",
                "side-obligation-actor-mismatch",
                event_seq=event["seq"],
            )
        target_value = event["target"]
        if not isinstance(target_value, Mapping) or target_value.get("zone") not in {
            "board",
            "hand",
        }:
            fail(
                "syntax",
                "x-event-shape",
                "invalid structured remove target",
                event_seq=event["seq"],
            )
        target = require_object(
            target_value,
            required=(
                {"zone", "at"}
                if target_value["zone"] == "board"
                else {"zone", "player"}
            ),
            context="remove target",
        )
        branch_index, _ = self._matching_obligation_branch(actor, target)
        branch = self.state.obligations[branch_index]
        head = branch[0]
        self.state.obligations = [branch]
        self._expire_claim_right()
        if target["zone"] == "board":
            at = target["at"]
            index = POINT_INDEX[at]
            if self.state.board[index] != piece_for(head.target_owner):
                fail(
                    "unreachable",
                    "obligation-target-mismatch",
                    "board target owner does not match",
                    event_seq=event["seq"],
                )
            self.state.board[index] = "."
            if "board-remove" in self.manifest.manifest["draw"]["noProgress"]["resetEvents"]:
                self.state.no_progress = 0
            self._reset_repetition("board-remove")
        else:
            player_index = 0 if head.target_owner == "w" else 1
            if self.state.hands[player_index] == 0:
                fail("unreachable", "obligation-target-mismatch", event_seq=event["seq"])
            self.state.hands[player_index] -= 1
            if "hand-remove" in self.manifest.manifest["draw"]["noProgress"]["resetEvents"]:
                self.state.no_progress = 0
            self._reset_repetition("hand-remove")
        if self._minimum_material(event["seq"]):
            return
        head.remaining -= 1
        if head.remaining > 0:
            protect = (
                head.cause == "mill"
                and self.manifest.manifest["mills"]["targetProtection"] == "outside-mill-first"
            )
            if head.zone == "b":
                head.targets = self._target_mask(head.target_owner, protect_mills=protect)
            self._promote_head(head)
            return
        completed = branch.pop(0)
        if branch:
            self._promote_head(branch[0])
            return
        self.state.obligations = []
        self.state.side = completed.after
        self._stabilize(source="event", event_seq=event["seq"])

    def _require_draw_boundary(self, event: Mapping[str, Any]) -> None:
        if self.state.outcome != "-" or self.state.obligations or self.state.action not in {"p", "m"}:
            fail("unreachable", "unstabilized-boundary", "draw negotiation requires a stable boundary", event_seq=event["seq"])

    def _apply_offer(self, event: Mapping[str, Any]) -> None:
        self._require_draw_boundary(event)
        if event["actor"] != self.state.side:
            fail("unreachable", "invalid-draw-actor", event_seq=event["seq"])
        if self.open_offer is not None:
            fail("unreachable", "offer-already-open", event_seq=event["seq"])
        audit = {
            "source": "event",
            "actor": event["actor"],
            "eventSeq": event["seq"],
            "kind": "draw-offer",
            "status": "open",
        }
        self.claims.append(audit)
        self.open_offer = {
            "source": "event",
            "actor": event["actor"],
            "offerEventSeq": event["seq"],
        }

    def _open_offer_audit(self) -> dict[str, Any]:
        if self.open_offer is None:
            fail("unreachable", "offer-not-open", "no draw offer is open")
        for claim in self.claims:
            if claim["source"] != self.open_offer["source"] or claim["status"] != "open":
                continue
            if claim["source"] == "event" and claim.get("eventSeq") != self.open_offer["offerEventSeq"]:
                continue
            return claim
        fail("inconsistent", "claims-mismatch", "open offer lacks audit record")

    def _apply_offer_resolution(self, event: Mapping[str, Any]) -> None:
        self._require_draw_boundary(event)
        audit = self._open_offer_audit()
        if event["offerEventSeq"] != self.open_offer["offerEventSeq"]:
            fail("unreachable", "offer-reference-mismatch", event_seq=event["seq"])
        event_type = event["type"]
        offerer = self.open_offer["actor"]
        if event_type == "withdraw-draw":
            expected_actor, status = offerer, "withdrawn"
        elif event_type == "decline-draw":
            expected_actor, status = other(offerer), "declined"
        else:
            expected_actor, status = other(offerer), "accepted"
        if event["actor"] != expected_actor:
            fail("unreachable", "invalid-draw-actor", event_seq=event["seq"])
        audit["status"] = status
        audit["resolvedEventSeq"] = event["seq"]
        self.open_offer = None
        if event_type == "accept-draw":
            self._terminal("d", "agreement", event["seq"])

    def _apply_claim(self, event: Mapping[str, Any]) -> None:
        if self.state.obligations:
            fail(
                "inconsistent",
                "claim-during-obligation",
                "claim-draw is forbidden during an obligation",
                event_seq=event["seq"],
            )
        self._require_draw_boundary(event)
        reason = event["reason"]
        if (
            self.claim_rights is None
            or self.claim_rights["actor"] != event["actor"]
            or reason not in self.claim_rights["reasons"]
        ):
            fail("unreachable", "claim-right-unavailable", event_seq=event["seq"])
        self.claims.append(
            {
                "source": "event",
                "actor": event["actor"],
                "eventSeq": event["seq"],
                "kind": "draw-claim",
                "status": "accepted",
            }
        )
        self._terminal("d", reason, event["seq"])

    def _apply_resign(self, event: Mapping[str, Any]) -> None:
        if event["actor"] not in {"w", "b"}:
            fail("syntax", "invalid-event-actor", event_seq=event["seq"])
        self._terminal(other(event["actor"]), "resignation", event["seq"])

    def _apply_adjudicate(self, event: Mapping[str, Any]) -> None:
        if event["actor"] != "system" or event["result"] not in {"w", "b", "d"}:
            fail("syntax", "invalid-adjudication", event_seq=event["seq"])
        if not isinstance(event["reason"], str) or not isinstance(event["authority"], str) or not event["authority"]:
            fail("syntax", "invalid-adjudication", event_seq=event["seq"])
        self._terminal(event["result"], "adjudication", event["seq"])


def execute(
    manifest_value: Any,
    origin: Any,
    events: Any,
    repetition_seed: Any,
    pre_origin_claims: Any,
) -> Execution:
    manifest = resolve_manifest(manifest_value)
    executor = FiniteRulesExecutor(
        manifest,
        origin,
        repetition_seed,
        pre_origin_claims,
    )
    return executor.run(events)


def _mstate_required(value: Any) -> Mapping[str, Any]:
    mstate = require_object(
        value,
        required={
            "format",
            "positionFormat",
            "stateProfile",
            "ruleset",
            "origin",
            "events",
            "current",
            "repetitionHistory",
            "preOriginClaims",
            "claims",
        },
        optional={"annotations", "extensions"},
        context="MSTATE/1.0",
    )
    if (
        mstate["format"] != "MSTATE/1.0"
        or mstate["positionFormat"] != "MFEN/1.0"
        or mstate["stateProfile"] != "mill24-state-v1"
    ):
        fail("unsupported", "unsupported-profile", "unsupported MSTATE profile")
    if "extensions" in mstate:
        fail("unsupported", "unsupported-profile", "MSTATE extensions are not implemented")
    if not isinstance(mstate["events"], list):
        fail("syntax", "array-required", "events must be an array")
    enforce_resource_limit("events", len(mstate["events"]), MAX_EVENTS)
    if not isinstance(mstate["repetitionHistory"], list):
        fail("syntax", "array-required", "repetitionHistory must be an array")
    enforce_resource_limit(
        "repetition-entries",
        len(mstate["repetitionHistory"]),
        MAX_REPETITION_ENTRIES,
    )
    return mstate


def _pre_origin_repetition(mstate: Mapping[str, Any], manifest: ResolvedManifest) -> list[dict[str, Any]]:
    history = mstate["repetitionHistory"]
    if not isinstance(history, list):
        fail("syntax", "array-required", "repetitionHistory must be an array")
    seed: list[dict[str, Any]] = []
    seen_non_seed = False
    for entry_value in history:
        entry = require_object(
            entry_value,
            required={"source", "key"},
            optional={"eventSeq"},
            context="repetition history entry",
        )
        if entry["source"] == "pre-origin" and not seen_non_seed:
            if "eventSeq" in entry:
                fail("inconsistent", "repetition-history-mismatch")
            seed.append({"source": "pre-origin", "key": _validate_observation(entry["key"], manifest)})
        else:
            seen_non_seed = True
    return seed


def resumption_state(execution: Execution) -> tuple[dict[str, Any], str]:
    pre_origin_repetition = [
        deep_copy_json(entry)
        for entry in execution.repetition_history
        if entry["source"] == "pre-origin"
    ]
    prefix = {
        "origin": execution.origin,
        "preOriginRepetition": pre_origin_repetition,
        "preOriginClaims": deep_copy_json(execution.pre_origin_claims),
        "events": deep_copy_json(execution.events),
    }
    last_seq = execution.events[-1]["seq"] if execution.events else 0
    value = {
        "profile": "resumption-state-v1",
        "positionFormat": "MFEN/1.0",
        "stateProfile": "mill24-state-v1",
        "semanticDigest": execution.manifest.semantic_digest,
        "current": execution.state.serialize(),
        "replayPrefixDigest": sha256_digest(prefix),
        "lastEventSeq": last_seq,
        "repetitionHistory": deep_copy_json(execution.repetition_history),
        "claims": deep_copy_json(execution.claims),
        "openOffer": deep_copy_json(execution.open_offer),
        "claimRights": deep_copy_json(execution.claim_rights),
    }
    return value, sha256_digest(value)


def replay(mstate_value: Any, caller_manifest: Any | None) -> tuple[Execution, dict[str, Any]]:
    mstate = _mstate_required(mstate_value)
    manifest = resolve_ruleset_envelope(mstate["ruleset"], caller_manifest)
    repetition_seed = _pre_origin_repetition(mstate, manifest)
    executor = FiniteRulesExecutor(
        manifest,
        mstate["origin"],
        repetition_seed,
        mstate["preOriginClaims"],
    )
    execution = executor.run(mstate["events"])
    actual_current = execution.state.serialize()
    if actual_current != mstate["current"]:
        fail(
            "replay",
            "checkpoint-mismatch",
            "replayed current does not match MSTATE checkpoint",
            expected=mstate["current"],
            actual=actual_current,
            include_expected=True,
            include_actual=True,
        )
    if execution.repetition_history != mstate["repetitionHistory"]:
        fail(
            "replay",
            "repetition-history-mismatch",
            "replayed repetition window does not match MSTATE",
            expected=mstate["repetitionHistory"],
            actual=execution.repetition_history,
            include_expected=True,
            include_actual=True,
        )
    if execution.claims != mstate["claims"]:
        fail(
            "replay",
            "claims-mismatch",
            "replayed claim audit does not match MSTATE",
            expected=mstate["claims"],
            actual=execution.claims,
            include_expected=True,
            include_actual=True,
        )
    decision = execution.trace[-1]["decisionState"]
    decision_digest = execution.trace[-1]["decisionDigest"]
    resumption, resumption_digest_value = resumption_state(execution)
    result = {
        "current": actual_current,
        "trace": execution.trace,
        "repetitionHistory": deep_copy_json(execution.repetition_history),
        "claims": deep_copy_json(execution.claims),
        "claimRights": deep_copy_json(execution.claim_rights),
        "decisionState": decision,
        "decisionDigest": decision_digest,
        "resumptionState": resumption,
        "resumptionDigest": resumption_digest_value,
    }
    return execution, result

"""MIF 1.0 manifests, MFEN state and structural-key handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .common import (
    MAX_EXACT_INTEGER,
    POINTS,
    TRANSFORM_IDS,
    fail,
    piece_for,
    require_digest,
    require_identifier,
    require_object,
    sha256_digest,
    topology_edges,
    topology_lines,
    transform_board,
    transform_coordinate,
    transform_line_bits,
    transform_mask,
)


CAUSE_ORDER = {
    "leap": 0,
    "intervention": 1,
    "custodian": 2,
    "mill": 3,
    "mill-count": 4,
    "stalemate": 5,
    "board-full": 6,
}
STANDARD_OUTCOMES = {
    "fewer-than-minimum",
    "no-legal-move",
    "no-legal-primary-action",
    "board-full",
    "no-progress",
    "repetition",
    "agreement",
    "resignation",
    "adjudication",
}
FEATURE_EXTENSION = {
    "last-mill": "lm",
    "placement-count": "pc",
    "used-lines": "ul",
}


def _uint(text: str, *, context: str, positive: bool = False) -> int:
    if not text or (len(text) > 1 and text.startswith("0")) or not text.isascii():
        fail("syntax", "invalid-integer", f"invalid {context}")
    if not text.isdecimal():
        fail("syntax", "invalid-integer", f"invalid {context}")
    value = int(text)
    if value > MAX_EXACT_INTEGER or (positive and value == 0):
        fail("syntax", "integer-out-of-range", f"invalid {context}")
    return value


def _board_field(board: list[str]) -> str:
    joined = "".join(board)
    return f"{joined[:8]}/{joined[8:16]}/{joined[16:]}"


@dataclass
class Obligation:
    actor: str
    cause: str
    zone: str
    target_owner: str
    remaining: int
    targets: int | str | None
    after: str

    @classmethod
    def parse(cls, text: str) -> "Obligation":
        parts = text.split(":")
        if len(parts) != 7:
            fail("syntax", "invalid-obligation", "obligation has seven fields")
        actor, cause, zone, owner, remaining_text, targets_text, after = parts
        if actor not in {"w", "b"} or owner not in {"w", "b"}:
            fail("syntax", "invalid-obligation", "invalid obligation player")
        if cause not in CAUSE_ORDER or zone not in {"b", "h"}:
            fail("syntax", "invalid-obligation", "invalid obligation cause or zone")
        remaining = _uint(remaining_text, context="obligation remaining", positive=True)
        if after not in {"w", "b", "q"}:
            fail("syntax", "invalid-obligation", "invalid obligation continuation")
        if zone == "h":
            if targets_text != "-":
                fail("syntax", "invalid-obligation", "hand obligation target must be '-'")
            targets: int | str | None = None
        elif targets_text == "~":
            targets = "~"
        elif len(targets_text) == 6 and all(
            character in "0123456789abcdef" for character in targets_text
        ):
            targets = int(targets_text, 16)
        else:
            fail("syntax", "invalid-obligation", "invalid board target mask")
        return cls(actor, cause, zone, owner, remaining, targets, after)

    def serialize(self) -> str:
        if self.zone == "h":
            target_text = "-"
        elif self.targets == "~":
            target_text = "~"
        elif isinstance(self.targets, int):
            target_text = f"{self.targets:06x}"
        else:
            fail("inconsistent", "obligation-target-mismatch", "missing targets")
        return ":".join(
            (
                self.actor,
                self.cause,
                self.zone,
                self.target_owner,
                str(self.remaining),
                target_text,
                self.after,
            )
        )

    def transformed(self, transform: str) -> "Obligation":
        targets = self.targets
        if self.zone == "b" and isinstance(targets, int):
            targets = transform_mask(targets, transform)
        return Obligation(
            self.actor,
            self.cause,
            self.zone,
            self.target_owner,
            self.remaining,
            targets,
            self.after,
        )


@dataclass
class MifState:
    board: list[str]
    side: str
    phase: str
    action: str
    hands: list[int]
    obligations: list[list[Obligation]]
    no_progress: int
    primary_ply: int
    outcome: str
    extensions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: Any) -> "MifState":
        if not isinstance(text, str) or not text.isascii():
            fail("syntax", "invalid-mfen", "MFEN must be US-ASCII text")
        fields = text.split(" ")
        if any(field == "" for field in fields) or len(fields) < 11:
            fail("syntax", "invalid-mfen", "MFEN field separation is invalid")
        if fields[0] != "MFEN/1.0":
            fail("syntax", "unsupported-format", "expected MFEN/1.0")
        if fields[1] != "mill24-state-v1":
            fail("unsupported", "unsupported-profile", "unsupported state profile")
        rings = fields[2].split("/")
        if len(rings) != 3 or any(len(ring) != 8 for ring in rings):
            fail("syntax", "invalid-board", "MFEN board must contain three rings")
        board = list("".join(rings))
        if any(piece not in "WBwb." for piece in board):
            fail("syntax", "invalid-board", "invalid MFEN board character")
        side, phase, action = fields[3:6]
        if side not in {"w", "b", "-"}:
            fail("syntax", "invalid-side", "invalid MFEN side")
        if phase not in {"p", "m", "o"} or action not in {"p", "m", "r", "o"}:
            fail("syntax", "invalid-phase-action", "invalid MFEN phase/action")
        hands_text = fields[6].split(",")
        if len(hands_text) != 2:
            fail("syntax", "invalid-hands", "hands must contain white,black")
        hands = [_uint(item, context="hand count") for item in hands_text]
        obligations: list[list[Obligation]] = []
        if fields[7] != "-":
            for branch_text in fields[7].split("|"):
                branch = [Obligation.parse(item) for item in branch_text.split(";")]
                if not branch:
                    fail("syntax", "invalid-obligation", "empty obligation branch")
                obligations.append(branch)
        no_progress = _uint(fields[8], context="no-progress")
        primary_ply = _uint(fields[9], context="primary-ply")
        outcome = fields[10]
        if outcome != "-":
            outcome_parts = outcome.split(":", 1)
            if (
                len(outcome_parts) != 2
                or outcome_parts[0] not in {"w", "b", "d"}
                or not (
                    outcome_parts[1] in STANDARD_OUTCOMES
                    or outcome_parts[1].startswith("x-")
                )
            ):
                fail("syntax", "invalid-outcome", "invalid MFEN outcome")
        extensions: dict[str, str] = {}
        for extension in fields[11:]:
            if "=" not in extension:
                fail("syntax", "invalid-extension", "extension lacks '='")
            key, value = extension.split("=", 1)
            require_identifier(key, context="extension key")
            if key in extensions:
                fail("syntax", "duplicate-extension", f"duplicate extension {key}")
            extensions[key] = value
        return cls(
            board,
            side,
            phase,
            action,
            hands,
            obligations,
            no_progress,
            primary_ply,
            outcome,
            extensions,
        )

    def clone(self) -> "MifState":
        return MifState(
            list(self.board),
            self.side,
            self.phase,
            self.action,
            list(self.hands),
            [
                [
                    Obligation(
                        item.actor,
                        item.cause,
                        item.zone,
                        item.target_owner,
                        item.remaining,
                        item.targets,
                        item.after,
                    )
                    for item in branch
                ]
                for branch in self.obligations
            ],
            self.no_progress,
            self.primary_ply,
            self.outcome,
            dict(self.extensions),
        )

    @property
    def board_field(self) -> str:
        return _board_field(self.board)

    @property
    def obligations_field(self) -> str:
        if not self.obligations:
            return "-"
        branch_texts = [";".join(item.serialize() for item in branch) for branch in self.obligations]
        branch_texts.sort(
            key=lambda value: (
                CAUSE_ORDER[value.split(":", 2)[1]],
                value.encode("ascii"),
            )
        )
        return "|".join(branch_texts)

    def serialize(self) -> str:
        parts = [
            "MFEN/1.0",
            "mill24-state-v1",
            self.board_field,
            self.side,
            self.phase,
            self.action,
            f"{self.hands[0]},{self.hands[1]}",
            self.obligations_field,
            str(self.no_progress),
            str(self.primary_ply),
            self.outcome,
        ]
        parts.extend(f"{key}={self.extensions[key]}" for key in sorted(self.extensions))
        return " ".join(parts)

    def live_count(self, player: str) -> int:
        return self.board.count(piece_for(player))

    def material_count(self, player: str) -> int:
        index = 0 if player == "w" else 1
        return self.live_count(player) + self.hands[index]

    def empty_points(self) -> list[str]:
        return [POINTS[index] for index, piece in enumerate(self.board) if piece == "."]

    def transformed(self, manifest: "ResolvedManifest", transform: str) -> "MifState":
        result = self.clone()
        result.board = list(transform_board(self.board_field, transform).replace("/", ""))
        result.obligations = [
            [item.transformed(transform) for item in branch]
            for branch in self.obligations
        ]
        transformed_extensions: dict[str, str] = {}
        for key, value in self.extensions.items():
            if key == "lm":
                halves = value.split(";")
                if len(halves) != 2:
                    fail("syntax", "invalid-extension", "invalid lm extension")
                converted: list[str] = []
                for half in halves:
                    points = half.split(",")
                    if len(points) != 2:
                        fail("syntax", "invalid-extension", "invalid lm extension")
                    converted.append(
                        ",".join(
                            point if point == "-" else transform_coordinate(point, transform)
                            for point in points
                        )
                    )
                transformed_extensions[key] = ";".join(converted)
            elif key == "pc":
                transformed_extensions[key] = value
            elif key == "ul":
                values = value.split(",")
                if len(values) != 2:
                    fail("syntax", "invalid-extension", "invalid ul extension")
                width = 4 if manifest.manifest["topology"] == "mill24-orthogonal-v1" else 5
                transformed_extensions[key] = ",".join(
                    f"{transform_line_bits(int(item, 16), manifest.manifest['topology'], transform):0{width}x}"
                    for item in values
                )
            else:
                fail("unsupported", "unsupported-profile", f"unknown state extension {key}")
        result.extensions = transformed_extensions
        return result


@dataclass(frozen=True)
class ResolvedManifest:
    manifest: Mapping[str, Any]
    semantic_projection: Mapping[str, Any]
    semantic_digest: str
    document_digest: str
    lines: tuple[tuple[str, str, str], ...]
    edges: tuple[tuple[str, str], ...]


MRS_REQUIRED = {
    "format",
    "id",
    "version",
    "title",
    "status",
    "semanticsProfile",
    "topology",
    "pieces",
    "turn",
    "flying",
    "placing",
    "mills",
    "captures",
    "boardFull",
    "stalemate",
    "draw",
    "semanticState",
}


def semantic_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    flying = manifest["flying"]
    projected_flying = (
        dict(flying) if flying["enabled"] else {"enabled": False}
    )
    placing = manifest["placing"]
    projected_placing = {
        "movementAllowed": placing["movementAllowed"],
        "earlyStop": (
            dict(placing["earlyStop"])
            if placing["earlyStop"]["emptyPoints"] != 0
            else {"emptyPoints": 0}
        ),
        "noLegalPrimaryAction": placing["noLegalPrimaryAction"],
    }
    projected_captures: dict[str, Any] = {"resolution": manifest["captures"]["resolution"]}
    for name in ("custodian", "intervention", "leap"):
        mechanism = manifest["captures"][name]
        projected_captures[name] = (
            dict(mechanism) if mechanism["enabled"] else {"enabled": False}
        )
    draw = manifest["draw"]
    no_progress = draw["noProgress"]
    repetition = draw["repetition"]
    projected_draw = {
        "noProgress": (
            dict(no_progress)
            if no_progress["normalLimit"] or no_progress["endgameLimit"]
            else {"enabled": False}
        ),
        "repetition": (
            dict(repetition) if repetition["count"] else {"count": 0}
        ),
        "offers": dict(draw["offers"]),
        "claimRights": dict(draw["claimRights"]),
    }
    projection: dict[str, Any] = {
        "profile": "mrs-semantic-v1",
        "semanticsProfile": manifest["semanticsProfile"],
        "topology": manifest["topology"],
        "pieces": dict(manifest["pieces"]),
        "turn": dict(manifest["turn"]),
        "flying": projected_flying,
        "placing": projected_placing,
        "mills": dict(manifest["mills"]),
        "captures": projected_captures,
        "boardFull": dict(manifest["boardFull"]),
        "stalemate": dict(manifest["stalemate"]),
        "draw": projected_draw,
        "semanticState": list(manifest["semanticState"]),
    }
    if manifest.get("extensions"):
        fail("unsupported", "unsupported-profile", "MRS extensions are not implemented")
    return projection


def _require_supported_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest["semanticsProfile"] != "mif-finite-rules-v3":
        fail("unsupported", "unsupported-profile", "unsupported rules semantics")
    if manifest["semanticState"]:
        fail(
            "unsupported",
            "unsupported-profile",
            "this adapter supports the corpus rulesets with no semantic state extension",
        )
    captures = manifest["captures"]
    if captures.get("resolution") != "target-commits-v1" or any(
        captures[name].get("enabled") for name in ("custodian", "intervention", "leap")
    ):
        fail("unsupported", "unsupported-profile", "capture mechanisms are not implemented")
    mills = manifest["mills"]
    supported_mills = {
        "placingEffect": "remove-opponent-board",
        "movingEffect": "remove-opponent-board",
        "targetProtection": mills.get("targetProtection"),
        "lineReuse": "unlimited",
        "reverseReformation": "allowed",
        "delayedClearBoundary": "on-enter-moving-v1",
        "removalMultiplicity": mills.get("removalMultiplicity"),
    }
    if (
        mills.get("placingEffect") != supported_mills["placingEffect"]
        or mills.get("movingEffect") != supported_mills["movingEffect"]
        or mills.get("targetProtection") not in {"outside-mill-first", "all-opponent"}
        or mills.get("lineReuse") != "unlimited"
        or mills.get("reverseReformation") != "allowed"
        or mills.get("delayedClearBoundary") != "on-enter-moving-v1"
        or mills.get("removalMultiplicity") not in {"one-per-primary", "one-per-new-line"}
    ):
        fail("unsupported", "unsupported-profile", "mill effect profile is not implemented")
    if manifest["stalemate"]["action"] not in {"loss", "draw"}:
        fail("unsupported", "unsupported-profile", "stalemate action is not implemented")


def resolve_manifest(value: Any, *, require_runtime_support: bool = True) -> ResolvedManifest:
    manifest = require_object(
        value,
        required=MRS_REQUIRED,
        optional={"description", "annotations", "extensions"},
        context="MRS/1.0",
    )
    if manifest["format"] != "MRS/1.0":
        fail("syntax", "unsupported-format", "expected MRS/1.0")
    require_identifier(manifest["id"], context="ruleset id")
    if not isinstance(manifest["version"], int) or not 1 <= manifest["version"] <= MAX_EXACT_INTEGER:
        fail("syntax", "integer-out-of-range", "invalid ruleset version")
    if manifest["topology"] not in {"mill24-orthogonal-v1", "mill24-diagonal-v1"}:
        fail("unsupported", "unsupported-profile", "unsupported topology")
    pieces = manifest["pieces"]
    require_object(
        pieces,
        required={"white", "black", "minimumLive"},
        context="pieces",
    )
    if any(not isinstance(pieces[key], int) or pieces[key] <= 0 for key in pieces):
        fail("syntax", "integer-out-of-range", "piece counts must be positive")
    if pieces["minimumLive"] > min(pieces["white"], pieces["black"]):
        fail("inconsistent", "manifest-conflict", "minimum live exceeds initial material")
    placing = manifest["placing"]
    policy = placing["noLegalPrimaryAction"]
    if policy not in {"apply-board-full", "loss", "draw"}:
        fail("inconsistent", "no-legal-primary-action-policy-invalid")
    if policy == "apply-board-full" and manifest["boardFull"]["action"] == "disabled":
        fail(
            "inconsistent",
            "no-legal-primary-action-policy-invalid",
            "apply-board-full requires an enabled board-full effect",
        )
    projection = semantic_projection(manifest)
    resolved = ResolvedManifest(
        manifest=manifest,
        semantic_projection=projection,
        semantic_digest=sha256_digest(projection),
        document_digest=sha256_digest(manifest),
        lines=topology_lines(manifest["topology"]),
        edges=topology_edges(manifest["topology"]),
    )
    if require_runtime_support:
        _require_supported_manifest(manifest)
    return resolved


def validate_state(state: MifState, manifest: ResolvedManifest) -> None:
    expected_extensions = {
        FEATURE_EXTENSION[feature] for feature in manifest.manifest["semanticState"]
    }
    if set(state.extensions) != expected_extensions:
        fail(
            "inconsistent",
            "required-semantic-state-missing",
            "MFEN semantic state does not match the MRS declaration",
        )
    pieces = manifest.manifest["pieces"]
    for player, index, key in (("w", 0, "white"), ("b", 1, "black")):
        occupied = state.board.count(piece_for(player)) + state.board.count(player)
        if occupied + state.hands[index] > pieces[key]:
            fail("inconsistent", "manifest-conflict", "state material exceeds manifest")
    if state.outcome == "-":
        if state.side not in {"w", "b"} or state.phase not in {"p", "m"}:
            fail("inconsistent", "invalid-ongoing-state", "ongoing state is not active")
        if state.obligations:
            actors = {branch[0].actor for branch in state.obligations}
            if actors != {state.side} or state.action != "r":
                fail(
                    "inconsistent",
                    "side-obligation-actor-mismatch",
                    "obligation actor and state side/action disagree",
                )
            for branch in state.obligations:
                for index, item in enumerate(branch):
                    if item.after == "q" and index == len(branch) - 1:
                        fail("inconsistent", "invalid-obligation", "q requires a successor")
                    if item.after != "q" and index != len(branch) - 1:
                        fail("inconsistent", "invalid-obligation", "intermediate item requires q")
                head = branch[0]
                if head.zone == "b" and (
                    not isinstance(head.targets, int) or head.targets == 0
                ):
                    fail(
                        "inconsistent",
                        "obligation-target-mismatch",
                        "board obligation head needs concrete targets",
                    )
        elif state.action != state.phase:
            fail("inconsistent", "invalid-phase-action", "stable action must match phase")
    elif (
        state.side != "-"
        or state.phase != "o"
        or state.action != "o"
        or state.obligations
    ):
        fail("inconsistent", "invalid-terminal-state", "terminal state is not normalized")


def resolve_ruleset_envelope(
    envelope_value: Any,
    caller_manifest: Any | None,
) -> ResolvedManifest:
    envelope = require_object(
        envelope_value,
        required={"mode", "id", "version", "semanticDigest"},
        optional={"documentDigest", "manifest"},
        context="ruleset envelope",
    )
    mode = envelope["mode"]
    if mode == "portable":
        if "manifest" not in envelope or "documentDigest" not in envelope:
            fail("integrity", "manifest-missing", "portable ruleset lacks manifest")
        if caller_manifest is not None and caller_manifest != envelope["manifest"]:
            fail("integrity", "manifest-conflict", "caller and portable manifests differ")
        manifest_value = envelope["manifest"]
    elif mode == "reference":
        if "manifest" in envelope:
            fail("syntax", "closed-object-mismatch", "reference envelope embeds manifest")
        if caller_manifest is None:
            fail(
                "integrity",
                "manifest-missing",
                "reference ruleset requires caller resolver",
            )
        manifest_value = caller_manifest
    else:
        fail("syntax", "invalid-ruleset-mode", "ruleset mode must be portable/reference")
    resolved = resolve_manifest(manifest_value)
    if envelope["id"] != resolved.manifest["id"] or envelope["version"] != resolved.manifest["version"]:
        fail("integrity", "manifest-conflict", "ruleset identity disagrees with manifest")
    require_digest(envelope["semanticDigest"], context="semantic digest")
    if envelope["semanticDigest"] != resolved.semantic_digest:
        fail("integrity", "semantic-digest-mismatch")
    if "documentDigest" in envelope:
        require_digest(envelope["documentDigest"], context="document digest")
        if envelope["documentDigest"] != resolved.document_digest:
            fail("integrity", "document-digest-mismatch")
    return resolved


def canonicalize_mfen(value: Any, manifest_value: Any | None) -> str:
    if manifest_value is None:
        fail("integrity", "manifest-missing", "MFEN semantic validation needs MRS")
    manifest = resolve_manifest(manifest_value)
    state = MifState.parse(value)
    validate_state(state, manifest)
    return state.serialize()


def canonicalize_mpk(value: Any, manifest_value: Any | None) -> str:
    if manifest_value is None:
        fail("integrity", "manifest-missing", "MPK binding needs MRS")
    manifest = resolve_manifest(manifest_value, require_runtime_support=False)
    if not isinstance(value, str) or not value.isascii():
        fail("syntax", "invalid-mpk", "MPK must be US-ASCII text")
    fields = value.split(" ")
    if any(field == "" for field in fields):
        fail("syntax", "invalid-mpk", "invalid MPK fields")
    if (
        len(fields) == 8
        and fields[:2] == ["MPK/1.0", "mill24-state-v1"]
        and fields[3] == "structural-d4-v1"
    ):
        fail("integrity", "mpk-semantic-digest-missing")
    if len(fields) < 9:
        fail("syntax", "invalid-mpk", "invalid MPK fields")
    if fields[0] != "MPK/1.0" or fields[1] != "mill24-state-v1":
        fail("unsupported", "unsupported-profile", "unsupported MPK profile")
    expected_ref = f"{manifest.manifest['id']}@{manifest.manifest['version']}"
    if fields[2] != expected_ref:
        fail("integrity", "manifest-conflict", "MPK ruleset reference mismatch")
    if not fields[3]:
        fail("integrity", "mpk-semantic-digest-missing")
    if (
        fields[3].startswith("sha256:")
        and len(fields[3]) == 71
        and all(character in "0123456789abcdefABCDEF" for character in fields[3][7:])
        and any(character in "ABCDEF" for character in fields[3][7:])
    ):
        fail("canonical", "non-canonical-digest")
    require_digest(fields[3], context="MPK semantic digest")
    if fields[3] != manifest.semantic_digest:
        fail("integrity", "semantic-digest-mismatch")
    key_profile = fields[4]
    if key_profile != "structural-d4-v1":
        fail("unsupported", "unsupported-profile", "only structural-d4-v1 is supported")
    board = fields[5]
    if len(board) != 24 or any(piece not in "WBwb." for piece in board):
        fail("syntax", "invalid-board", "MPK board must have 24 characters")
    if fields[6] not in {"w", "b"} or fields[7] not in {"p", "m"}:
        fail("syntax", "invalid-mpk", "MPK side or phase is invalid")
    hands_parts = fields[8].split(",")
    if len(hands_parts) != 2:
        fail("syntax", "invalid-hands", "MPK hands are invalid")
    hands = f"{_uint(hands_parts[0], context='hand')},{_uint(hands_parts[1], context='hand')}"
    if len(fields) != 9:
        fail("unsupported", "unsupported-profile", "MPK key extensions are not implemented")
    candidates = []
    for transform in TRANSFORM_IDS:
        transformed_board = transform_board(board, transform, slashes=False)
        candidates.append(
            " ".join(
                (
                    "MPK/1.0",
                    "mill24-state-v1",
                    expected_ref,
                    manifest.semantic_digest,
                    key_profile,
                    transformed_board,
                    fields[6],
                    fields[7],
                    hands,
                )
            )
        )
    return min(candidates, key=lambda item: item.encode("ascii"))

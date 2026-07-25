"""Strict deterministic subprocess bridge for the pinned Sanmill UCI CLI.

Sanmill owns historical rule state.  This module deliberately exposes only a
fixed-node, single-threaded, fail-closed contract; it is not a general UCI
client and it never chooses a replacement move after an engine failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from game.board import POSITIONS, BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.phase_corpus import project_tgf_fen
from learned_ai.training.run_contract import canonical_sha256


PINNED_SANMILL_COMMIT = "db65eb3e73189d934d615d0f47519d395193c646"
PINNED_SANMILL_SHORT_COMMIT = PINNED_SANMILL_COMMIT[:10]
PINNED_SANMILL_TREE = "b8fa6c0119c2dec4443efc59deab8b7d835e0c88"
SANMILL_BINARY_RELATIVE = (
    Path("target") / "release" / ("tgf.exe" if os.name == "nt" else "tgf")
)
EXPECTED_SANMILL_BINARY_SHA256 = (
    "cac2ec6fe45a9d798a89c6b8a5f52c767aa1c885a1156a96269b44ebf81976cc"
)
EXPECTED_SANMILL_BINARY_SIZE = 4_109_312
STRICT_BUILD_COMMAND = "cargo build --release -p tgf-cli"
STRICT_HASH_MIB = 16
STRICT_PROTOCOL_VERSION = 1
EXPECTED_RULES_IDENTITY_SHA256 = (
    "3e62cb93a1e0afe4534ce4824d233344816050b547bb8761dd7fe985d8ad399f"
)
SANMILL_LICENSE_RELATIVE = Path("Copying.txt")
EXPECTED_SANMILL_LICENSE_SHA256 = (
    "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
)
SANMILL_OPENING_BOOK_RELATIVE = (
    Path("src")
    / "ui"
    / "flutter_app"
    / "assets"
    / "opening_books"
    / "nmm"
    / "opening_book.json"
)
EXPECTED_OPENING_BOOK_SHA256 = (
    "cdc4768bc461c22177634985a4cc1d92452774e2992515b937fed8812eb076f5"
)
EXPECTED_OPENING_BOOK_ORACLE_ENTRIES = 109
EXPECTED_OPENING_BOOK_RECOMMENDATIONS = 437
REMOVED_INVALID_ORACLE_KEY = (
    "****OO*O/O@O*@OO@/@@**@*O* b p p 8 1 6 2 0 0 -1 -1 -1 -1 0 0 8 ids:nodes"
)
REMOVED_INVALID_ORACLE_KEY_SHA256 = (
    "904777ade504367c4e62446f105f1b125aaea7d6bec217984518025d8df3b0d1"
)
_SANMILL_PINNED_SOURCE_SCOPE = (
    "crates",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo",
    "docs/UCI_CLI_BRIDGE.md",
    SANMILL_OPENING_BOOK_RELATIVE.as_posix(),
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_COORDINATES = frozenset(POSITIONS)
_OPTION_NAME = re.compile(r"^option name (?P<name>.+?) type ")
_ERROR_PREFIX = "info string sanmill_error "
_LOGICAL_TURN_PREFIX = "info string sanmill_logical_turn "
_STATE_PREFIX = "info string sanmill_state "
_PROTOCOL_ERRORS = (
    "info string unsupported setoption:",
    "info string invalid fen ignored:",
    "info string unknown command:",
)
_WINNER_NAMES = {-1: "none", 0: "white", 1: "black", 2: "draw"}
_OUTCOME_REASON_NAMES = {
    0: "ongoing",
    1: "loseFewerThanThree",
    2: "drawFiftyMoveLegacy",
    3: "drawFullBoard",
    4: "loseFullBoard",
    5: "drawThreefoldRepetition",
    6: "loseNoLegalMoves",
    7: "drawStalemateCondition",
    8: "drawFiftyMove",
    9: "drawEndgameFiftyMove",
}
_DRAW_REASON_CODES = frozenset({2, 3, 5, 7, 8, 9})
_WIN_REASON_CODES = frozenset({1, 4, 6})


class SanmillBridgeError(RuntimeError):
    """Raised when identity, protocol, rule, or reproducibility checks fail."""


class SanmillProtocolError(SanmillBridgeError):
    """A versioned fail-closed error emitted by Sanmill."""

    def __init__(
        self,
        *,
        code: str,
        command: str,
        message: str,
        action_index: int | None = None,
        token: str | None = None,
    ) -> None:
        self.code = code
        self.command = command
        self.message = message
        self.action_index = action_index
        self.token = token
        detail = f"Sanmill {command} error {code}: {message}"
        if action_index is not None:
            detail += f" (action_index={action_index}, token={token!r})"
        super().__init__(detail)


@dataclass(frozen=True)
class SanmillInstallation:
    checkout: Path
    commit: str
    checkout_head: str
    tree: str
    binary: Path
    binary_sha256: str
    binary_size: int
    license_sha256: str

    def portable_record(self) -> dict[str, Any]:
        return {
            "path_lookup_key": "sanmill_checkout",
            "commit": self.commit,
            "checkout_head": self.checkout_head,
            "checkout_policy": (
                "pinned commit or descendant with no changes in the pinned "
                "CLI, rule, build, bridge-document, or opening-book scope"
            ),
            "tree": self.tree,
            "binary_relative_path": SANMILL_BINARY_RELATIVE.as_posix(),
            "binary_sha256": self.binary_sha256,
            "binary_size": self.binary_size,
            "build_command": STRICT_BUILD_COMMAND,
            "strict_failure_protocol_version": STRICT_PROTOCOL_VERSION,
            "license": {
                "spdx": "AGPL-3.0-or-later",
                "relative_path": SANMILL_LICENSE_RELATIVE.as_posix(),
                "sha256": self.license_sha256,
            },
        }


@dataclass(frozen=True)
class SanmillOpeningBookGate:
    """Audited book identity and the remaining UCI activation gate."""

    asset_sha256: str
    oracle_entries: int
    oracle_recommendations: int
    removed_invalid_key_sha256: str

    def portable_record(self) -> dict[str, Any]:
        return {
            "requested_for_future_formal_baseline": True,
            "active_in_bridge_smoke": False,
            "asset_relative_path": SANMILL_OPENING_BOOK_RELATIVE.as_posix(),
            "asset_sha256": self.asset_sha256,
            "oracle_entries": self.oracle_entries,
            "oracle_recommendations": self.oracle_recommendations,
            "legality_audit": {
                "authority": "pinned-sanmill-uci-legal-actions",
                "checked_recommendations": self.oracle_recommendations,
                "illegal_recommendations": 0,
                "duplicate_recommendations": 0,
            },
            "removed_invalid_oracle_recommendation": {
                "raw_key_sha256": self.removed_invalid_key_sha256,
                "present": False,
                "historical_reason": "c3 was already occupied in the source position",
            },
            "uci_support": "not-advertised-at-pinned-commit",
            "remaining_gate": (
                "expose a deterministic fail-closed UCI book interface and "
                "freeze its paired-opening diversity policy"
            ),
        }


@dataclass(frozen=True)
class UciSearchResult:
    bestmove: str
    depth: int
    nodes: int
    score_kind: str
    score: int
    elapsed_seconds: float
    raw_line: str

    @property
    def terminal_token(self) -> bool:
        return self.bestmove in {"draw", "none", "0000"}

    def semantic_record(self) -> dict[str, Any]:
        return {
            "bestmove": self.bestmove,
            "depth": self.depth,
            "nodes": self.nodes,
            "score_kind": self.score_kind,
            "score": self.score,
        }


@dataclass(frozen=True)
class UciOutcomeState:
    winner_code: int
    reason_code: int

    @property
    def terminal(self) -> bool:
        return self.winner_code != -1

    @property
    def winner(self) -> str:
        return _WINNER_NAMES[self.winner_code]

    @property
    def reason(self) -> str:
        return _OUTCOME_REASON_NAMES[self.reason_code]

    def portable_record(self) -> dict[str, Any]:
        return {
            "terminal": self.terminal,
            "winner": self.winner,
            "winner_code": self.winner_code,
            "reason": self.reason,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class UciStateOutcome:
    terminal: bool
    winner: str | None
    winner_code: int | None
    reason: str
    reason_code: str

    def portable_record(self) -> dict[str, Any]:
        return {
            "terminal": self.terminal,
            "winner": self.winner,
            "winner_code": self.winner_code,
            "reason": self.reason,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class UciPositionState:
    status: str
    ruleset_id: str
    rules_identity_sha256: str
    rules_options: Mapping[str, Any]
    history_origin: str
    fen: str
    side_to_move: str | None
    phase: str
    action: str
    pending_removal_count: int
    pending_removals: tuple[int, int]
    legal_actions: tuple[str, ...]
    action_token_count: int
    logical_ply_count: int
    logical_plies_by_side: tuple[int, int]
    no_capture_count: int
    repetition_current_count: int
    repetition_history_length: int
    snapshot_history_length: int
    history_sha256: str
    terminal: bool
    winner: str | None
    winner_code: int | None
    outcome_reason: str
    outcome_reason_code: str
    raw_line: str

    @property
    def removal_pending(self) -> bool:
        return self.pending_removal_count > 0

    @property
    def outcome(self) -> UciStateOutcome:
        return UciStateOutcome(
            terminal=self.terminal,
            winner=self.winner,
            winner_code=self.winner_code,
            reason=self.outcome_reason,
            reason_code=self.outcome_reason_code,
        )

    def portable_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ruleset_id": self.ruleset_id,
            "rules_identity_sha256": self.rules_identity_sha256,
            "history_origin": self.history_origin,
            "fen": self.fen,
            "side_to_move": self.side_to_move,
            "phase": self.phase,
            "action": self.action,
            "terminal": self.terminal,
            "removal_pending": self.removal_pending,
            "pending_removal_count": self.pending_removal_count,
            "pending_removals": list(self.pending_removals),
            "legal_actions": list(self.legal_actions),
            "action_token_count": self.action_token_count,
            "logical_ply_count": self.logical_ply_count,
            "logical_plies_by_side": list(self.logical_plies_by_side),
            "no_capture_count": self.no_capture_count,
            "repetition_current_count": self.repetition_current_count,
            "repetition_history_length": self.repetition_history_length,
            "snapshot_history_length": self.snapshot_history_length,
            "history_sha256": self.history_sha256,
            "outcome": self.outcome.portable_record(),
        }


@dataclass(frozen=True)
class UciLogicalTurnResult:
    status: str
    full_turn_actions: tuple[str, ...]
    logical_move_id: str | None
    model_action: Mapping[str, str | None] | None
    logical_ply_delta: int
    resulting_fen: str | None
    resulting_side_to_move: str | None
    terminal: bool
    winner: str | None
    winner_code: int | None
    outcome_reason: str
    effective_depth: int | None
    completed_depth: int | None
    score_kind: str | None
    score: int | None
    score_perspective: str | None
    node_budget: int
    primary_nodes: int
    removal_nodes: int
    total_nodes: int
    search_calls: int
    elapsed_seconds: float
    raw_line: str

    def semantic_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "full_turn_actions": list(self.full_turn_actions),
            "logical_move_id": self.logical_move_id,
            "model_action": (
                dict(self.model_action) if self.model_action is not None else None
            ),
            "logical_ply_delta": self.logical_ply_delta,
            "resulting_fen": self.resulting_fen,
            "resulting_side_to_move": self.resulting_side_to_move,
            "terminal": self.terminal,
            "winner": self.winner,
            "winner_code": self.winner_code,
            "outcome_reason": self.outcome_reason,
            "effective_depth": self.effective_depth,
            "completed_depth": self.completed_depth,
            "score_kind": self.score_kind,
            "score": self.score,
            "score_perspective": self.score_perspective,
            "node_budget": self.node_budget,
            "primary_nodes": self.primary_nodes,
            "removal_nodes": self.removal_nodes,
            "total_nodes": self.total_nodes,
            "search_calls": self.search_calls,
        }


def _machine_json_object(line: str, prefix: str, *, context: str) -> Mapping[str, Any]:
    if not line.startswith(prefix):
        raise SanmillBridgeError(f"{context} response has the wrong prefix")
    encoded = line.removeprefix(prefix)
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise SanmillBridgeError(f"{context} response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise SanmillBridgeError(f"{context} response must contain a JSON object")
    if payload.get("protocol_version") != STRICT_PROTOCOL_VERSION:
        raise SanmillBridgeError(
            f"{context} protocol version is not {STRICT_PROTOCOL_VERSION}"
        )
    return payload


def _required_string(
    payload: Mapping[str, Any],
    field: str,
    *,
    context: str,
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise SanmillBridgeError(f"{context}.{field} must be a non-empty string")
    return value


def _optional_string(
    payload: Mapping[str, Any],
    field: str,
    *,
    context: str,
) -> str | None:
    value = payload.get(field)
    if value is not None and (not isinstance(value, str) or not value):
        raise SanmillBridgeError(f"{context}.{field} must be null or a string")
    return value


def _required_int(
    payload: Mapping[str, Any],
    field: str,
    *,
    context: str,
    minimum: int = 0,
) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SanmillBridgeError(
            f"{context}.{field} must be an integer >= {minimum}"
        )
    return value


def _optional_int(
    payload: Mapping[str, Any],
    field: str,
    *,
    context: str,
) -> int | None:
    value = payload.get(field)
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        raise SanmillBridgeError(f"{context}.{field} must be null or an integer")
    return value


def _required_bool(
    payload: Mapping[str, Any],
    field: str,
    *,
    context: str,
) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise SanmillBridgeError(f"{context}.{field} must be a boolean")
    return value


def _nonnegative_int_pair(
    payload: Mapping[str, Any],
    field: str,
    *,
    context: str,
) -> tuple[int, int]:
    value = payload.get(field)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in value
        )
    ):
        raise SanmillBridgeError(
            f"{context}.{field} must contain two non-negative integers"
        )
    return value[0], value[1]


def parse_protocol_error(line: str) -> SanmillProtocolError:
    payload = _machine_json_object(line, _ERROR_PREFIX, context="sanmill_error")
    if payload.get("status") != "error":
        raise SanmillBridgeError("sanmill_error.status must be error")
    code = _required_string(payload, "code", context="sanmill_error")
    command = _required_string(payload, "command", context="sanmill_error")
    message = _required_string(payload, "message", context="sanmill_error")
    action_index = _optional_int(
        payload,
        "action_index",
        context="sanmill_error",
    )
    token = _optional_string(payload, "token", context="sanmill_error")
    if (action_index is None) != (token is None):
        raise SanmillBridgeError(
            "sanmill_error action_index and token must appear together"
        )
    if action_index is not None and action_index < 0:
        raise SanmillBridgeError("sanmill_error.action_index must be non-negative")
    return SanmillProtocolError(
        code=code,
        command=command,
        message=message,
        action_index=action_index,
        token=token,
    )


def _validate_sha256(value: str, *, context: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise SanmillBridgeError(f"{context} must be a lowercase SHA-256")
    return value


def parse_state_json_line(line: str) -> UciPositionState:
    payload = _machine_json_object(line, _STATE_PREFIX, context="sanmill_state")
    status = _required_string(payload, "status", context="sanmill_state")
    if status in {"position_unavailable", "error"}:
        code = _required_string(payload, "code", context="sanmill_state")
        message = _required_string(payload, "message", context="sanmill_state")
        if status == "position_unavailable":
            position_code = _required_string(
                payload,
                "position_error_code",
                context="sanmill_state",
            )
            message = f"{message}; rejected position code={position_code}"
        raise SanmillProtocolError(
            code=code,
            command="statejson",
            message=message,
        )
    if status not in {"ok", "terminal"}:
        raise SanmillBridgeError(f"unknown sanmill_state status: {status}")

    ruleset_id = _required_string(payload, "ruleset_id", context="sanmill_state")
    if ruleset_id != "nmm":
        raise SanmillBridgeError(f"unexpected Sanmill ruleset: {ruleset_id}")
    identity = payload.get("rules_identity")
    if not isinstance(identity, dict) or identity.get("format_version") != 1:
        raise SanmillBridgeError("invalid Sanmill rules identity")
    identity_sha256 = _validate_sha256(
        _required_string(identity, "sha256", context="sanmill_state.rules_identity"),
        context="sanmill_state.rules_identity.sha256",
    )
    if identity_sha256 != EXPECTED_RULES_IDENTITY_SHA256:
        raise SanmillBridgeError(
            "Sanmill active rule options differ from the pinned NMM contract"
        )
    rules_options = payload.get("rules_options")
    if not isinstance(rules_options, dict):
        raise SanmillBridgeError("sanmill_state.rules_options must be an object")
    history_origin = _required_string(
        payload,
        "history_origin",
        context="sanmill_state",
    )
    if history_origin not in {"game_start", "fresh_setup"}:
        raise SanmillBridgeError("unknown Sanmill history origin")
    fen = _required_string(payload, "fen", context="sanmill_state")
    if "\n" in fen or "\r" in fen:
        raise SanmillBridgeError("sanmill_state.fen must be one line")
    side_to_move = _optional_string(
        payload,
        "side_to_move",
        context="sanmill_state",
    )
    if side_to_move not in {None, "white", "black"}:
        raise SanmillBridgeError("unknown Sanmill side to move")
    phase = _required_string(payload, "phase", context="sanmill_state")
    if phase not in {"ready", "placing", "moving", "game_over"}:
        raise SanmillBridgeError("unknown Sanmill phase")
    action = _required_string(payload, "action", context="sanmill_state")
    if action not in {"place", "select", "remove", "game_over"}:
        raise SanmillBridgeError("unknown Sanmill atomic action")
    pending_removal = _required_bool(
        payload,
        "pending_removal",
        context="sanmill_state",
    )
    pending_removal_count = _required_int(
        payload,
        "pending_removal_count",
        context="sanmill_state",
    )
    pending_removals = _nonnegative_int_pair(
        payload,
        "pending_removals",
        context="sanmill_state",
    )
    if pending_removal != (pending_removal_count > 0):
        raise SanmillBridgeError("Sanmill pending-removal fields disagree")
    if pending_removal and action != "remove":
        raise SanmillBridgeError("pending Sanmill removal has the wrong action")
    if not pending_removal and action == "remove":
        raise SanmillBridgeError("Sanmill remove action lacks a pending removal")

    raw_legal = payload.get("legal_actions")
    if not isinstance(raw_legal, list) or any(
        not isinstance(item, str) for item in raw_legal
    ):
        raise SanmillBridgeError("sanmill_state.legal_actions must be strings")
    legal_actions = tuple(validate_uci_action_token(item) for item in raw_legal)
    if len(legal_actions) != len(set(legal_actions)):
        raise SanmillBridgeError("Sanmill advertised duplicate legal actions")
    if pending_removal != any(action.startswith("x") for action in legal_actions):
        raise SanmillBridgeError(
            "Sanmill pending-removal state and legal-action kinds disagree"
        )
    if pending_removal and any(
        not legal.startswith("x") for legal in legal_actions
    ):
        raise SanmillBridgeError("pending Sanmill removal advertised a primary action")

    action_token_count = _required_int(
        payload,
        "action_token_count",
        context="sanmill_state",
    )
    logical_ply_count = _required_int(
        payload,
        "logical_ply_count",
        context="sanmill_state",
    )
    pending_primary = 1 if pending_removal else 0
    minimum_pending_primary = (
        pending_primary if history_origin == "game_start" else 0
    )
    if not (
        logical_ply_count + minimum_pending_primary
        <= action_token_count
        <= 2 * logical_ply_count + pending_primary
    ):
        raise SanmillBridgeError(
            "Sanmill atomic-action and logical-ply counts disagree: "
            f"actions={action_token_count}, logical={logical_ply_count}, "
            f"pending={pending_removal}"
        )
    logical_plies_by_side = _nonnegative_int_pair(
        payload,
        "logical_plies_by_side",
        context="sanmill_state",
    )
    if sum(logical_plies_by_side) != logical_ply_count:
        raise SanmillBridgeError("Sanmill per-side logical-ply counts disagree")
    no_capture_count = _required_int(
        payload,
        "no_capture_count",
        context="sanmill_state",
    )
    repetition_current_count = _required_int(
        payload,
        "repetition_current_count",
        context="sanmill_state",
    )
    repetition_history_length = _required_int(
        payload,
        "repetition_history_length",
        context="sanmill_state",
    )
    snapshot_history_length = _required_int(
        payload,
        "snapshot_history_length",
        context="sanmill_state",
    )
    history_sha256 = _validate_sha256(
        _required_string(payload, "history_sha256", context="sanmill_state"),
        context="sanmill_state.history_sha256",
    )
    terminal = _required_bool(payload, "terminal", context="sanmill_state")
    winner = _optional_string(payload, "winner", context="sanmill_state")
    winner_code = _optional_int(payload, "winner_code", context="sanmill_state")
    outcome_reason = _required_string(
        payload,
        "outcome_reason",
        context="sanmill_state",
    )
    outcome_reason_code = _required_string(
        payload,
        "outcome_reason_code",
        context="sanmill_state",
    )
    if terminal != (status == "terminal"):
        raise SanmillBridgeError("Sanmill state status and terminal flag disagree")
    if terminal:
        if legal_actions:
            raise SanmillBridgeError("terminal Sanmill state still advertises actions")
        if outcome_reason == "ongoing" or outcome_reason_code == "ongoing":
            raise SanmillBridgeError("terminal Sanmill state has an ongoing outcome")
    elif (
        winner is not None
        or winner_code is not None
        or outcome_reason != "ongoing"
        or outcome_reason_code != "ongoing"
    ):
        raise SanmillBridgeError("ongoing Sanmill state has a terminal outcome")
    if winner not in {None, "white", "black"}:
        raise SanmillBridgeError("unknown Sanmill winner")
    if (winner, winner_code) not in {
        (None, None),
        ("white", 0),
        ("black", 1),
    }:
        raise SanmillBridgeError("Sanmill winner name and code disagree")

    return UciPositionState(
        status=status,
        ruleset_id=ruleset_id,
        rules_identity_sha256=identity_sha256,
        rules_options=dict(rules_options),
        history_origin=history_origin,
        fen=fen,
        side_to_move=side_to_move,
        phase=phase,
        action=action,
        pending_removal_count=pending_removal_count,
        pending_removals=pending_removals,
        legal_actions=legal_actions,
        action_token_count=action_token_count,
        logical_ply_count=logical_ply_count,
        logical_plies_by_side=logical_plies_by_side,
        no_capture_count=no_capture_count,
        repetition_current_count=repetition_current_count,
        repetition_history_length=repetition_history_length,
        snapshot_history_length=snapshot_history_length,
        history_sha256=history_sha256,
        terminal=terminal,
        winner=winner,
        winner_code=winner_code,
        outcome_reason=outcome_reason,
        outcome_reason_code=outcome_reason_code,
        raw_line=line,
    )


def _model_action_from_tokens(
    actions: Sequence[str],
) -> dict[str, str | None]:
    if len(actions) not in {1, 2}:
        raise SanmillBridgeError("logical turn must contain one or two actions")
    primary = validate_uci_action_token(actions[0])
    if primary.startswith("x"):
        raise SanmillBridgeError("logical turn begins with a removal")
    source: str | None
    target: str
    if "-" in primary:
        source, target = primary.split("-")
    else:
        source, target = None, primary
    capture = None
    if len(actions) == 2:
        removal = validate_uci_action_token(actions[1])
        if not removal.startswith("x"):
            raise SanmillBridgeError("logical turn second action is not a removal")
        capture = removal[1:]
    return {"from": source, "to": target, "capture": capture}


def parse_logical_turn_line(
    line: str,
    elapsed_seconds: float = 0.0,
) -> UciLogicalTurnResult:
    if elapsed_seconds < 0:
        raise SanmillBridgeError("logical-turn elapsed time must be non-negative")
    payload = _machine_json_object(
        line,
        _LOGICAL_TURN_PREFIX,
        context="sanmill_logical_turn",
    )
    status = _required_string(
        payload,
        "status",
        context="sanmill_logical_turn",
    )
    if status not in {"ok", "terminal"}:
        raise SanmillBridgeError(f"unknown logical-turn status: {status}")
    raw_actions = payload.get("full_turn_actions")
    if not isinstance(raw_actions, list) or any(
        not isinstance(item, str) for item in raw_actions
    ):
        raise SanmillBridgeError(
            "sanmill_logical_turn.full_turn_actions must be strings"
        )
    actions = tuple(validate_uci_action_token(item) for item in raw_actions)
    logical_ply_delta = _required_int(
        payload,
        "logical_ply_delta",
        context="sanmill_logical_turn",
    )
    terminal = _required_bool(
        payload,
        "terminal",
        context="sanmill_logical_turn",
    )
    winner = _optional_string(
        payload,
        "winner",
        context="sanmill_logical_turn",
    )
    winner_code = _optional_int(
        payload,
        "winner_code",
        context="sanmill_logical_turn",
    )
    if winner not in {None, "white", "black"} or (winner, winner_code) not in {
        (None, None),
        ("white", 0),
        ("black", 1),
    }:
        raise SanmillBridgeError("logical-turn winner fields disagree")
    outcome_reason = _required_string(
        payload,
        "outcome_reason",
        context="sanmill_logical_turn",
    )
    node_budget = _required_int(
        payload,
        "node_budget",
        context="sanmill_logical_turn",
        minimum=1,
    )
    primary_nodes = _required_int(
        payload,
        "primary_nodes",
        context="sanmill_logical_turn",
    )
    removal_nodes = _required_int(
        payload,
        "removal_nodes",
        context="sanmill_logical_turn",
    )
    total_nodes = _required_int(
        payload,
        "total_nodes",
        context="sanmill_logical_turn",
    )
    search_calls = _required_int(
        payload,
        "search_calls",
        context="sanmill_logical_turn",
    )
    if primary_nodes + removal_nodes != total_nodes or total_nodes > node_budget:
        raise SanmillBridgeError("logical-turn node accounting exceeds its budget")

    if status == "terminal":
        if (
            actions
            or logical_ply_delta != 0
            or not terminal
            or outcome_reason == "ongoing"
            or total_nodes != 0
            or search_calls != 0
        ):
            raise SanmillBridgeError("malformed terminal logical-turn response")
        return UciLogicalTurnResult(
            status=status,
            full_turn_actions=actions,
            logical_move_id=None,
            model_action=None,
            logical_ply_delta=logical_ply_delta,
            resulting_fen=None,
            resulting_side_to_move=None,
            terminal=terminal,
            winner=winner,
            winner_code=winner_code,
            outcome_reason=outcome_reason,
            effective_depth=None,
            completed_depth=None,
            score_kind=None,
            score=None,
            score_perspective=None,
            node_budget=node_budget,
            primary_nodes=primary_nodes,
            removal_nodes=removal_nodes,
            total_nodes=total_nodes,
            search_calls=search_calls,
            elapsed_seconds=elapsed_seconds,
            raw_line=line,
        )

    if not actions or logical_ply_delta != 1:
        raise SanmillBridgeError("successful logical turn is not one complete ply")
    expected_model_action = _model_action_from_tokens(actions)
    model_action = payload.get("model_action")
    if not isinstance(model_action, dict) or set(model_action) != {
        "from",
        "to",
        "capture",
    }:
        raise SanmillBridgeError("logical-turn model_action has the wrong shape")
    if dict(model_action) != expected_model_action:
        raise SanmillBridgeError(
            "logical-turn action tokens and model_action disagree"
        )
    logical_move_id = _required_string(
        payload,
        "logical_move_id",
        context="sanmill_logical_turn",
    )
    if logical_move_id != "".join(actions):
        raise SanmillBridgeError("logical-turn identifier disagrees with its actions")
    resulting_fen = _required_string(
        payload,
        "resulting_fen",
        context="sanmill_logical_turn",
    )
    resulting_side_to_move = _optional_string(
        payload,
        "resulting_side_to_move",
        context="sanmill_logical_turn",
    )
    if resulting_side_to_move not in {None, "white", "black"}:
        raise SanmillBridgeError("unknown logical-turn resulting side")
    effective_depth = _required_int(
        payload,
        "effective_depth",
        context="sanmill_logical_turn",
        minimum=1,
    )
    completed_depth = _required_int(
        payload,
        "completed_depth",
        context="sanmill_logical_turn",
        minimum=1,
    )
    if completed_depth > effective_depth:
        raise SanmillBridgeError("logical-turn completed depth exceeds its ceiling")
    score_kind = _required_string(
        payload,
        "score_kind",
        context="sanmill_logical_turn",
    )
    if score_kind not in {"cp", "mate"}:
        raise SanmillBridgeError("unknown logical-turn score kind")
    score = _required_int(
        payload,
        "score",
        context="sanmill_logical_turn",
        minimum=-(2**31),
    )
    score_perspective = _required_string(
        payload,
        "score_perspective",
        context="sanmill_logical_turn",
    )
    if score_perspective != "white":
        raise SanmillBridgeError("logical-turn score is not White-perspective")
    if total_nodes <= 0 or search_calls <= 0:
        raise SanmillBridgeError("successful logical turn did not search")
    if terminal:
        if outcome_reason == "ongoing":
            raise SanmillBridgeError("terminal logical turn has an ongoing outcome")
    elif (
        winner is not None
        or winner_code is not None
        or outcome_reason != "ongoing"
        or resulting_side_to_move is None
    ):
        raise SanmillBridgeError("ongoing logical turn has terminal fields")

    return UciLogicalTurnResult(
        status=status,
        full_turn_actions=actions,
        logical_move_id=logical_move_id,
        model_action=dict(model_action),
        logical_ply_delta=logical_ply_delta,
        resulting_fen=resulting_fen,
        resulting_side_to_move=resulting_side_to_move,
        terminal=terminal,
        winner=winner,
        winner_code=winner_code,
        outcome_reason=outcome_reason,
        effective_depth=effective_depth,
        completed_depth=completed_depth,
        score_kind=score_kind,
        score=score,
        score_perspective=score_perspective,
        node_budget=node_budget,
        primary_nodes=primary_nodes,
        removal_nodes=removal_nodes,
        total_nodes=total_nodes,
        search_calls=search_calls,
        elapsed_seconds=elapsed_seconds,
        raw_line=line,
    )


def parse_debug_outcome(lines: Sequence[str]) -> UciOutcomeState:
    values: dict[str, int] = {}
    for line in lines:
        for field in ("winner", "outcome_reason"):
            prefix = f"{field}:"
            if line.startswith(prefix):
                if field in values:
                    raise SanmillBridgeError(f"duplicate Sanmill debug field: {field}")
                try:
                    values[field] = int(line.removeprefix(prefix).strip())
                except ValueError as exc:
                    raise SanmillBridgeError(
                        f"non-integer Sanmill debug field: {line}"
                    ) from exc
    if set(values) != {"winner", "outcome_reason"}:
        raise SanmillBridgeError(
            "Sanmill debug output lacks authoritative outcome fields"
        )
    winner_code = values["winner"]
    reason_code = values["outcome_reason"]
    if winner_code not in _WINNER_NAMES or reason_code not in _OUTCOME_REASON_NAMES:
        raise SanmillBridgeError(
            "Sanmill debug output contains an unknown winner or outcome reason"
        )
    if winner_code == -1 and reason_code != 0:
        raise SanmillBridgeError("ongoing Sanmill outcome has a terminal reason")
    if winner_code == 2 and reason_code not in _DRAW_REASON_CODES:
        raise SanmillBridgeError("Sanmill draw has a non-draw outcome reason")
    if winner_code in {0, 1} and reason_code not in _WIN_REASON_CODES:
        raise SanmillBridgeError("Sanmill winner has a non-win outcome reason")
    return UciOutcomeState(winner_code=winner_code, reason_code=reason_code)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SanmillBridgeError(f"cannot read path registry: {path}") from exc
    if not isinstance(value, dict):
        raise SanmillBridgeError("path registry must contain a JSON object")
    return value


def _resolve_registry_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SanmillBridgeError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = _REPOSITORY_ROOT / path
    return path.resolve()


def _git_output(checkout: Path, *arguments: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={checkout.as_posix()}",
        "-C",
        str(checkout),
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        raise SanmillBridgeError("cannot execute Git for Sanmill identity") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SanmillBridgeError(f"Sanmill Git inspection failed: {detail}")
    return result.stdout.strip()


def inspect_sanmill_installation(
    paths_config: str | Path,
    *,
    binary_override: str | Path | None = None,
) -> SanmillInstallation:
    """Verify the pinned source and release binary without changing Sanmill."""
    config = _strict_json_object(Path(paths_config))
    checkout = _resolve_registry_path(
        config.get("sanmill_checkout"), field="sanmill_checkout"
    )
    if not checkout.is_dir():
        raise SanmillBridgeError("sanmill_checkout is not a directory")

    head = _git_output(checkout, "rev-parse", "HEAD")
    pinned_tree = _git_output(
        checkout,
        "rev-parse",
        f"{PINNED_SANMILL_COMMIT}^{{tree}}",
    )
    if pinned_tree != PINNED_SANMILL_TREE:
        raise SanmillBridgeError(
            "the pinned Sanmill commit no longer resolves to its recorded tree"
        )
    if head != PINNED_SANMILL_COMMIT:
        try:
            _git_output(
                checkout,
                "merge-base",
                "--is-ancestor",
                PINNED_SANMILL_COMMIT,
                head,
            )
        except SanmillBridgeError as exc:
            raise SanmillBridgeError(
                f"Sanmill HEAD does not descend from {PINNED_SANMILL_COMMIT}: {head}"
            ) from exc
        relevant_drift = _git_output(
            checkout,
            "diff",
            "--name-only",
            f"{PINNED_SANMILL_COMMIT}..{head}",
            "--",
            *_SANMILL_PINNED_SOURCE_SCOPE,
        )
        if relevant_drift:
            raise SanmillBridgeError(
                "Sanmill checkout changed pinned bridge source paths:\n"
                f"{relevant_drift}"
            )
    dirty = _git_output(checkout, "status", "--short", "--untracked-files=all")
    if dirty:
        raise SanmillBridgeError(f"Sanmill checkout is not clean:\n{dirty}")
    binary = (
        Path(binary_override).resolve()
        if binary_override is not None
        else checkout / SANMILL_BINARY_RELATIVE
    )
    if not binary.is_file():
        raise SanmillBridgeError(f"Sanmill UCI binary is absent: {binary}")
    binary_bytes = binary.read_bytes()
    binary_sha256 = hashlib.sha256(binary_bytes).hexdigest()
    if os.name != "nt":
        raise SanmillBridgeError("the pinned Sanmill binary identity is Windows-only")
    if (
        len(binary_bytes) != EXPECTED_SANMILL_BINARY_SIZE
        or binary_sha256 != EXPECTED_SANMILL_BINARY_SHA256
    ):
        raise SanmillBridgeError(
            "Sanmill UCI binary identity differs from the pinned strict build"
        )

    license_path = checkout / SANMILL_LICENSE_RELATIVE
    if not license_path.is_file():
        raise SanmillBridgeError("Sanmill license text is absent")
    license_sha256 = _sha256_file(license_path)
    if license_sha256 != EXPECTED_SANMILL_LICENSE_SHA256:
        raise SanmillBridgeError("Sanmill license identity differs from the pinned text")
    return SanmillInstallation(
        checkout=checkout,
        commit=PINNED_SANMILL_COMMIT,
        checkout_head=head,
        tree=pinned_tree,
        binary=binary,
        binary_sha256=binary_sha256,
        binary_size=len(binary_bytes),
        license_sha256=license_sha256,
    )


def inspect_sanmill_opening_book(
    installation: SanmillInstallation,
) -> SanmillOpeningBookGate:
    """Audit every book recommendation while keeping UCI book play disabled."""
    asset = installation.checkout / SANMILL_OPENING_BOOK_RELATIVE
    if not asset.is_file():
        raise SanmillBridgeError(f"Sanmill opening-book asset is absent: {asset}")
    try:
        payload = json.loads(asset.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SanmillBridgeError("cannot parse the Sanmill opening book") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schemaVersion") != 1
        or payload.get("variant") != "nmm"
        or payload.get("symmetry") != "ring16"
        or not isinstance(payload.get("oracle"), dict)
    ):
        raise SanmillBridgeError("Sanmill opening book lacks an Oracle object")

    asset_sha256 = _sha256_file(asset)
    if asset_sha256 != EXPECTED_OPENING_BOOK_SHA256:
        raise SanmillBridgeError(
            "pinned Sanmill opening-book identity differs from the audited asset"
        )
    oracle = payload["oracle"]
    key_sha256 = hashlib.sha256(REMOVED_INVALID_ORACLE_KEY.encode("utf-8")).hexdigest()
    if key_sha256 != REMOVED_INVALID_ORACLE_KEY_SHA256:
        raise SanmillBridgeError("removed opening-book key identity drifted")
    if REMOVED_INVALID_ORACLE_KEY in oracle:
        raise SanmillBridgeError(
            "the removed invalid Sanmill opening-book recommendation reappeared"
        )
    if len(oracle) != EXPECTED_OPENING_BOOK_ORACLE_ENTRIES:
        raise SanmillBridgeError("Sanmill opening-book Oracle entry count drifted")

    recommendation_count = 0
    with SanmillUciSession(installation) as session:
        for fen in sorted(oracle):
            recommendations = oracle[fen]
            if (
                not isinstance(fen, str)
                or not isinstance(recommendations, list)
                or not recommendations
                or any(not isinstance(move, str) for move in recommendations)
                or len(recommendations) != len(set(recommendations))
            ):
                raise SanmillBridgeError(f"invalid opening-book record shape: {fen!r}")
            session.new_game()
            session.position_fen(fen)
            legal_actions = set(session.position_state().legal_actions)
            for move in recommendations:
                recommendation_count += 1
                token = validate_uci_action_token(move)
                if token not in legal_actions:
                    raise SanmillBridgeError(
                        f"illegal Sanmill opening-book recommendation {move!r}: {fen}"
                    )
    if recommendation_count != EXPECTED_OPENING_BOOK_RECOMMENDATIONS:
        raise SanmillBridgeError("Sanmill opening-book recommendation count drifted")
    return SanmillOpeningBookGate(
        asset_sha256=asset_sha256,
        oracle_entries=len(oracle),
        oracle_recommendations=recommendation_count,
        removed_invalid_key_sha256=key_sha256,
    )


def strict_option_values(seed: int = 42) -> tuple[tuple[str, str], ...]:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SanmillBridgeError("search seed must be a non-negative integer")
    return (
        ("Threads", "1"),
        ("Hash", str(STRICT_HASH_MIB)),
        ("Ponder", "false"),
        ("MultiPV", "1"),
        ("SkillLevel", "30"),
        ("MoveTimeMs", "0"),
        ("AiIsLazy", "false"),
        ("IDSEnabled", "true"),
        ("DepthExtension", "true"),
        ("Shuffling", "false"),
        ("UseLazySmp", "false"),
        ("Algorithm", "2"),
        ("StrictFailurePolicy", "true"),
        ("DrawOnHumanExperience", "true"),
        ("UsePerfectDatabase", "false"),
        ("PatchAvoidTraps", "false"),
        ("PatchMakeTraps", "false"),
        ("SearchShuffleSeed", str(seed)),
        ("ConsiderMobility", "true"),
        ("FocusOnBlockingPaths", "false"),
        ("DeveloperMode", "false"),
        ("MaxQuiescenceDepth", "0"),
        ("PiecesCount", "9"),
        ("flyPieceCount", "3"),
        ("PiecesAtLeastCount", "3"),
        ("HasDiagonalLines", "false"),
        ("MillFormationActionInPlacingPhase", "0"),
        ("MayMoveInPlacingPhase", "false"),
        ("IsDefenderMoveFirst", "false"),
        ("MayRemoveMultiple", "false"),
        ("MayRemoveFromMillsAlways", "false"),
        ("RestrictRepeatedMillsFormation", "false"),
        ("OneTimeUseMill", "false"),
        ("CustodianCaptureEnabled", "false"),
        ("InterventionCaptureEnabled", "false"),
        ("LeapCaptureEnabled", "false"),
        ("BoardFullAction", "0"),
        ("StopPlacingWhenTwoEmptySquares", "false"),
        ("StalemateAction", "0"),
        ("MayFly", "true"),
        ("NMoveRule", "100"),
        ("EndgameNMoveRule", "100"),
        ("ThreefoldRepetitionRule", "true"),
    )


def strict_contract_record(seed: int = 42) -> dict[str, Any]:
    options = strict_option_values(seed)
    return {
        "schema_version": "nmm.sanmill-strict-uci-contract.v2",
        "sanmill_commit": PINNED_SANMILL_COMMIT,
        "command": [SANMILL_BINARY_RELATIVE.as_posix(), "mill", "uci"],
        "search_command": "go logical nodes <positive-N> [depth <positive-D>]",
        "state_command": "statejson",
        "protocol_versions": {
            "error": STRICT_PROTOCOL_VERSION,
            "logical_turn": STRICT_PROTOCOL_VERSION,
            "state": STRICT_PROTOCOL_VERSION,
        },
        "options": {name: value for name, value in options},
        "child_environment": "inherit non-TGF variables; remove all TGF_* variables",
        "random_failure_fallback": (
            "forbidden-by-strict-policy-and-logical-turn-search-path"
        ),
        "search_failure": "versioned-sanmill_error-no-substitution",
        "logical_turn_budget": (
            "one aggregate node ceiling covers primary and mandatory removal"
        ),
        "position_ownership": (
            "go logical does not mutate state; caller replays full_turn_actions "
            "and verifies statejson"
        ),
        "knowledge_sources": {
            "opening_book": {
                "requested_for_future_formal_baseline": True,
                "active_in_bridge_smoke": False,
                "reason": "UCI-interface-and-paired-diversity-policy-gate",
            },
            "human_database": {
                "active": False,
                "reason": "data-query-only-and-not-used-by-logical-search",
            },
            "perfect_database": {"active": False},
            "patch_and_trap": {"active": False},
        },
        "draw_on_human_experience_semantics": {
            "enabled": True,
            "purpose": "phase-aware automatic search-depth policy",
            "effective_in_smoke": True,
            "reason": "no-positive-explicit-depth-is-sent",
        },
        "contract_identity": canonical_sha256(
            {
                "commit": PINNED_SANMILL_COMMIT,
                "depth": "sanmill-phase-policy",
                "options": options,
                "fallback": "strict-policy-logical-turn",
                "protocol": STRICT_PROTOCOL_VERSION,
            }
        ),
    }


def validate_uci_action_token(token: str) -> str:
    if not isinstance(token, str) or not token or any(ch.isspace() for ch in token):
        raise SanmillBridgeError("UCI action must be one non-empty token")
    if token.startswith("x"):
        coordinate = token[1:]
        if coordinate not in _COORDINATES:
            raise SanmillBridgeError(f"invalid UCI removal token: {token}")
        return token
    if "-" in token:
        fields = token.split("-")
        if len(fields) != 2 or any(field not in _COORDINATES for field in fields):
            raise SanmillBridgeError(f"invalid UCI movement token: {token}")
        return token
    if token not in _COORDINATES:
        raise SanmillBridgeError(f"invalid UCI placement token: {token}")
    return token


def nmm_move_base(move: Mapping[str, Any]) -> str:
    source = move.get("from")
    target = move.get("to")
    if target not in _COORDINATES:
        raise SanmillBridgeError("NMM move has an invalid destination")
    if source is None:
        return str(target)
    if source not in _COORDINATES:
        raise SanmillBridgeError("NMM move has an invalid source")
    return f"{source}-{target}"


def assert_stable_legal_parity(
    board: BoardState,
    sanmill_actions: Sequence[str],
) -> list[dict[str, Any]]:
    """Return NMM atomic moves after checking stable primary-action parity."""
    if any(action.startswith("x") for action in sanmill_actions):
        raise SanmillBridgeError("stable Sanmill state advertised a removal")
    nmm_moves = get_all_legal_moves(board)
    nmm_bases = {nmm_move_base(move) for move in nmm_moves}
    sanmill_bases = {validate_uci_action_token(action) for action in sanmill_actions}
    if nmm_bases != sanmill_bases:
        raise SanmillBridgeError(
            "stable legal-action divergence: "
            f"Sanmill-only={sorted(sanmill_bases - nmm_bases)}, "
            f"NMM-only={sorted(nmm_bases - sanmill_bases)}"
        )
    return nmm_moves


def assert_pending_removal_parity(
    nmm_moves: Sequence[Mapping[str, Any]],
    primary_action: str,
    sanmill_actions: Sequence[str],
) -> tuple[str, ...]:
    expected = {
        f"x{move['capture']}"
        for move in nmm_moves
        if nmm_move_base(move) == primary_action and move.get("capture")
    }
    observed = {validate_uci_action_token(action) for action in sanmill_actions}
    if any(not action.startswith("x") for action in observed):
        raise SanmillBridgeError("pending-removal state advertised a primary move")
    if expected != observed:
        raise SanmillBridgeError(
            "pending-removal divergence: "
            f"Sanmill-only={sorted(observed - expected)}, "
            f"NMM-only={sorted(expected - observed)}"
        )
    if not expected:
        raise SanmillBridgeError("pending-removal state has no legal capture")
    return tuple(sorted(expected))


def atomic_move_for_actions(
    nmm_moves: Sequence[Mapping[str, Any]],
    primary_action: str,
    removal_action: str | None,
) -> dict[str, Any]:
    capture = removal_action[1:] if removal_action is not None else None
    matches = [
        dict(move)
        for move in nmm_moves
        if nmm_move_base(move) == primary_action and move.get("capture") == capture
    ]
    if len(matches) != 1:
        raise SanmillBridgeError(
            "staged Sanmill actions do not select exactly one NMM atomic move"
        )
    return matches[0]


def project_stable_sanmill_fen(tgf_fen: str) -> BoardState:
    projected = project_tgf_fen(tgf_fen)
    if projected is None:
        raise SanmillBridgeError("Sanmill FEN is pending a removal")
    return BoardState.from_fen_string(projected.fen)


def parse_search_line(line: str, elapsed_seconds: float) -> UciSearchResult:
    tokens = line.split()
    try:
        depth_index = tokens.index("depth")
        score_index = tokens.index("score")
        nodes_index = tokens.index("nodes")
        move_index = tokens.index("bestmove")
        depth = int(tokens[depth_index + 1])
        score_kind = tokens[score_index + 1]
        score = int(tokens[score_index + 2])
        nodes = int(tokens[nodes_index + 1])
        bestmove = tokens[move_index + 1]
    except (ValueError, IndexError) as exc:
        raise SanmillBridgeError(f"malformed Sanmill search result: {line}") from exc
    if score_kind not in {"cp", "mate"}:
        raise SanmillBridgeError(f"unknown Sanmill score kind: {score_kind}")
    if bestmove not in {"draw", "none", "0000"}:
        validate_uci_action_token(bestmove)
    if depth < 0 or nodes < 0 or elapsed_seconds < 0:
        raise SanmillBridgeError("Sanmill search result contains a negative metric")
    return UciSearchResult(
        bestmove=bestmove,
        depth=depth,
        nodes=nodes,
        score_kind=score_kind,
        score=score,
        elapsed_seconds=elapsed_seconds,
        raw_line=line,
    )


class SanmillUciSession:
    """One strict Sanmill process with deterministic options and timeouts."""

    def __init__(
        self,
        installation: SanmillInstallation,
        *,
        seed: int = 42,
        protocol_timeout: float = 10.0,
        search_timeout: float = 120.0,
    ) -> None:
        if protocol_timeout <= 0 or search_timeout <= 0:
            raise SanmillBridgeError("UCI timeouts must be positive")
        self.installation = installation
        self.seed = seed
        self.protocol_timeout = protocol_timeout
        self.search_timeout = search_timeout
        self.transcript: list[dict[str, str]] = []
        self.advertised_options: dict[str, str] = {}
        self.engine_identity: dict[str, str] = {}
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        child_env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("TGF_")
        }
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        )
        try:
            self._process = subprocess.Popen(
                [str(installation.binary), "mill", "uci"],
                cwd=installation.checkout,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_env,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise SanmillBridgeError("cannot start the Sanmill UCI process") from exc
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout,
            args=(self._process.stdout,),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "SanmillUciSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def stderr_text(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_lines)

    def _pump_stdout(self, stream: Any) -> None:
        try:
            for line in stream:
                self._stdout.put(line.rstrip("\r\n"))
        finally:
            self._stdout.put(None)

    def _pump_stderr(self, stream: Any) -> None:
        for line in stream:
            with self._stderr_lock:
                self._stderr_lines.append(line.rstrip("\r\n"))

    def _send(self, line: str) -> None:
        if "\n" in line or "\r" in line:
            raise SanmillBridgeError("UCI command contains a newline")
        if self._process.poll() is not None:
            raise SanmillBridgeError(
                f"Sanmill exited before command; stderr={self.stderr_text!r}"
            )
        assert self._process.stdin is not None
        self.transcript.append({"direction": "to_engine", "line": line})
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise SanmillBridgeError("Sanmill stdin failed") from exc

    def _read_until(
        self,
        predicate: Callable[[str], bool],
        *,
        timeout: float,
        context: str,
        defer_protocol_error: bool = False,
    ) -> tuple[str, list[str]]:
        deadline = time.monotonic() + timeout
        seen: list[str] = []
        deferred_error: SanmillBridgeError | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SanmillBridgeError(
                    f"Sanmill timeout while waiting for {context}; "
                    f"stderr={self.stderr_text!r}"
                )
            try:
                line = self._stdout.get(timeout=remaining)
            except queue.Empty as exc:
                raise SanmillBridgeError(
                    f"Sanmill timeout while waiting for {context}; "
                    f"stderr={self.stderr_text!r}"
                ) from exc
            if line is None:
                raise SanmillBridgeError(
                    f"Sanmill stdout closed while waiting for {context}; "
                    f"stderr={self.stderr_text!r}"
                )
            self.transcript.append({"direction": "from_engine", "line": line})
            seen.append(line)
            protocol_error: SanmillBridgeError | None = None
            if line.startswith(_ERROR_PREFIX):
                protocol_error = parse_protocol_error(line)
            elif line.startswith(_PROTOCOL_ERRORS):
                protocol_error = SanmillBridgeError(
                    f"Sanmill protocol error: {line}"
                )
            if protocol_error is not None:
                if not defer_protocol_error:
                    raise protocol_error
                if deferred_error is None:
                    deferred_error = protocol_error
                continue
            if predicate(line):
                if deferred_error is not None:
                    raise deferred_error
                return line, seen

    def _sync(self) -> list[str]:
        self._send("isready")
        _, lines = self._read_until(
            lambda line: line == "readyok",
            timeout=self.protocol_timeout,
            context="readyok",
            defer_protocol_error=True,
        )
        return lines

    def _initialize(self) -> None:
        self._send("uci")
        _, lines = self._read_until(
            lambda line: line == "uciok",
            timeout=self.protocol_timeout,
            context="uciok",
        )
        for line in lines:
            if line.startswith("id name "):
                self.engine_identity["name"] = line.removeprefix("id name ")
            elif line.startswith("id author "):
                self.engine_identity["author"] = line.removeprefix("id author ")
            match = _OPTION_NAME.match(line)
            if match:
                self.advertised_options[match.group("name")] = line
        if self.engine_identity != {
            "name": "TGF Mill Rust",
            "author": "The Sanmill developers",
        }:
            raise SanmillBridgeError(
                f"unexpected Sanmill UCI identity: {self.engine_identity}"
            )

        required = {name for name, _ in strict_option_values(self.seed)}
        required.update({"Clear Hash", "PerfectDatabasePath", "PatchPath", "TrapPath"})
        missing = sorted(required - self.advertised_options.keys())
        if missing:
            raise SanmillBridgeError(f"Sanmill omits required UCI options: {missing}")
        book_options = sorted(
            name
            for name in self.advertised_options
            if "openingbook" in name.lower().replace(" ", "")
        )
        if book_options:
            raise SanmillBridgeError(
                "pinned UCI unexpectedly advertises opening-book options; "
                f"freeze an explicit disabled value before use: {book_options}"
            )
        for empty_default in ("PerfectDatabasePath", "PatchPath", "TrapPath"):
            if "default <empty>" not in self.advertised_options[empty_default]:
                raise SanmillBridgeError(
                    f"{empty_default} does not advertise an empty default"
                )

        for name, value in strict_option_values(self.seed):
            self._send(f"setoption name {name} value {value}")
        self._sync()

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            self._send("quit")
            process.wait(timeout=2.0)
        except (SanmillBridgeError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("setoption name Clear Hash")
        self._sync()

    def position_startpos(self, actions: Sequence[str] = ()) -> None:
        moves = [validate_uci_action_token(action) for action in actions]
        command = "position startpos"
        if moves:
            command += " moves " + " ".join(moves)
        self._send(command)
        self._sync()

    def position_fen(self, fen: str, actions: Sequence[str] = ()) -> None:
        if not isinstance(fen, str) or not fen.strip() or "\n" in fen or "\r" in fen:
            raise SanmillBridgeError("Sanmill FEN must be one non-empty line")
        moves = [validate_uci_action_token(action) for action in actions]
        command = "position fen " + fen.strip()
        if moves:
            command += " moves " + " ".join(moves)
        self._send(command)
        self._sync()

    def export_fen(self) -> str:
        self._send("fen")
        line, _ = self._read_until(
            lambda value: value.startswith("fen "),
            timeout=self.protocol_timeout,
            context="exported FEN",
        )
        return line.removeprefix("fen ")

    def legal_moves(self) -> tuple[str, ...]:
        self._send("moves")
        line, _ = self._read_until(
            lambda value: value == "moves" or value.startswith("moves "),
            timeout=self.protocol_timeout,
            context="legal moves",
        )
        moves = tuple(line.split()[1:])
        for move in moves:
            validate_uci_action_token(move)
        if len(set(moves)) != len(moves):
            raise SanmillBridgeError("Sanmill advertised duplicate legal moves")
        return moves

    def history_summary(self) -> str:
        self._send("hist")
        line, _ = self._read_until(
            lambda value: value.startswith("hist "),
            timeout=self.protocol_timeout,
            context="repetition history",
        )
        return line

    def debug_outcome(self) -> UciOutcomeState:
        self._send("d")
        self._send("isready")
        _, lines = self._read_until(
            lambda value: value == "readyok",
            timeout=self.protocol_timeout,
            context="debug outcome and readyok",
        )
        return parse_debug_outcome(lines)

    def position_state(self) -> UciPositionState:
        """Compatibility alias for the authoritative machine-readable state."""
        return self.state_json()

    def state_json(self) -> UciPositionState:
        self._send("statejson")
        line, _ = self._read_until(
            lambda value: value.startswith(_STATE_PREFIX),
            timeout=self.protocol_timeout,
            context="sanmill_state",
        )
        return parse_state_json_line(line)

    def search_logical_turn(
        self,
        node_budget: int,
        *,
        depth: int | None = None,
    ) -> UciLogicalTurnResult:
        if (
            not isinstance(node_budget, int)
            or isinstance(node_budget, bool)
            or node_budget <= 0
        ):
            raise SanmillBridgeError("node budget must be a positive integer")
        if depth is not None and (
            not isinstance(depth, int)
            or isinstance(depth, bool)
            or depth <= 0
        ):
            raise SanmillBridgeError("logical search depth must be positive")

        state = self.state_json()
        if state.removal_pending:
            raise SanmillBridgeError(
                "refusing logical search from a pending-removal position"
            )
        command = f"go logical nodes {node_budget}"
        if depth is not None:
            command += f" depth {depth}"
        started = time.perf_counter()
        self._send(command)
        line, _ = self._read_until(
            lambda value: value.startswith(_LOGICAL_TURN_PREFIX),
            timeout=self.search_timeout,
            context="sanmill_logical_turn or sanmill_error",
        )
        result = parse_logical_turn_line(
            line,
            elapsed_seconds=time.perf_counter() - started,
        )
        if result.node_budget != node_budget:
            raise SanmillBridgeError(
                "Sanmill logical-turn response changed the requested node budget"
            )
        if state.terminal:
            if result.status != "terminal":
                raise SanmillBridgeError(
                    "terminal Sanmill root returned a non-terminal logical turn"
                )
            return result
        if result.status != "ok":
            raise SanmillBridgeError(
                "ongoing Sanmill root returned a terminal logical response"
            )

        board = project_stable_sanmill_fen(state.fen)
        nmm_moves = assert_stable_legal_parity(board, state.legal_actions)
        primary = result.full_turn_actions[0]
        removal = (
            result.full_turn_actions[1]
            if len(result.full_turn_actions) == 2
            else None
        )
        atomic = atomic_move_for_actions(nmm_moves, primary, removal)
        if dict(result.model_action or {}) != atomic:
            raise SanmillBridgeError(
                "Sanmill logical-turn model action differs from NMM mapping"
            )
        return result

    def _run_fixed_node_search(
        self,
        node_budget: int,
        legal: tuple[str, ...],
    ) -> UciSearchResult:
        command = f"go nodes {node_budget}"
        started = time.perf_counter()
        self._send(command)
        line, _ = self._read_until(
            lambda value: "bestmove" in value.split(),
            timeout=self.search_timeout,
            context="bestmove",
        )
        elapsed = time.perf_counter() - started
        result = parse_search_line(line, elapsed)
        if result.depth <= 0 and not result.terminal_token:
            raise SanmillBridgeError("Sanmill reported no positive search depth")
        if result.nodes > node_budget:
            raise SanmillBridgeError(
                f"Sanmill exceeded fixed node ceiling: {result.nodes}>{node_budget}"
            )
        if result.bestmove in {"none", "0000"}:
            if legal:
                raise SanmillBridgeError(
                    "Sanmill returned no move for a state with legal actions"
                )
            return result
        if result.bestmove == "draw":
            if legal:
                raise SanmillBridgeError(
                    "Sanmill returned draw while legal actions remained advertised"
                )
            return result
        if result.nodes <= 0:
            raise SanmillBridgeError("Sanmill returned a move without searching a node")
        if result.bestmove not in legal:
            raise SanmillBridgeError(
                f"Sanmill returned an illegal bestmove: {result.bestmove}"
            )
        return result

    def search_fixed_nodes(self, node_budget: int) -> UciSearchResult:
        if (
            not isinstance(node_budget, int)
            or isinstance(node_budget, bool)
            or node_budget <= 0
        ):
            raise SanmillBridgeError("node budget must be a positive integer")
        state = self.position_state()
        if state.terminal:
            raise SanmillBridgeError(
                "refusing to search a terminal position; inspect Sanmill state first"
            )
        return self._run_fixed_node_search(node_budget, state.legal_actions)

    def probe_terminal_draw(self, node_budget: int) -> UciSearchResult:
        """Exercise Sanmill's own draw short-circuit on a known terminal draw."""
        if (
            not isinstance(node_budget, int)
            or isinstance(node_budget, bool)
            or node_budget <= 0
        ):
            raise SanmillBridgeError("node budget must be a positive integer")
        state = self.position_state()
        if not state.terminal:
            raise SanmillBridgeError("draw probe requires a terminal Sanmill state")
        if state.outcome.winner != "draw":
            raise SanmillBridgeError("draw probe received a decisive Sanmill outcome")
        result = self._run_fixed_node_search(node_budget, state.legal_actions)
        if result.bestmove != "draw" or result.nodes != 0:
            raise SanmillBridgeError(
                "Sanmill terminal draw did not use its zero-node draw short-circuit"
            )
        return result


def runtime_record() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }

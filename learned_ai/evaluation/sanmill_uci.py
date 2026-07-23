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


PINNED_SANMILL_COMMIT = "6f080c5a6d15919bf0a45fa5528c45d4487a2b8f"
PINNED_SANMILL_SHORT_COMMIT = PINNED_SANMILL_COMMIT[:10]
PINNED_SANMILL_TREE = "8b52f4d084758414ebc9aa4db239448f69e10bcf"
SANMILL_BINARY_RELATIVE = (
    Path("target") / "release" / ("tgf.exe" if os.name == "nt" else "tgf")
)
EXPECTED_SANMILL_BINARY_SHA256 = (
    "b1c816ee40f6cb9a91916ad094e82175ee6c975c7d15c396e672af58a15dc1a6"
)
EXPECTED_SANMILL_BINARY_SIZE = 3_720_192
FAIL_CLOSED_ASSERTION = (
    b"main search returned MOVE_NONE; bug must be diagnosed before "
    b"release-mode fallback masks it"
)
STRICT_BUILD_COMMAND = (
    "cargo --config profile.release.debug-assertions=true build --release -p tgf-cli"
)
STRICT_HASH_MIB = 16
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

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_COORDINATES = frozenset(POSITIONS)
_OPTION_NAME = re.compile(r"^option name (?P<name>.+?) type ")
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


@dataclass(frozen=True)
class SanmillInstallation:
    checkout: Path
    commit: str
    tree: str
    binary: Path
    binary_sha256: str
    binary_size: int
    license_sha256: str

    def portable_record(self) -> dict[str, Any]:
        return {
            "path_lookup_key": "sanmill_checkout",
            "commit": self.commit,
            "tree": self.tree,
            "binary_relative_path": SANMILL_BINARY_RELATIVE.as_posix(),
            "binary_sha256": self.binary_sha256,
            "binary_size": self.binary_size,
            "fail_closed_assertion_present": True,
            "build_command": STRICT_BUILD_COMMAND,
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
class UciPositionState:
    fen: str
    side_to_move: str
    phase: str
    action: str
    legal_actions: tuple[str, ...]
    history: str
    outcome: UciOutcomeState

    @property
    def terminal(self) -> bool:
        return self.phase == "o"

    @property
    def removal_pending(self) -> bool:
        return self.action == "r"

    def portable_record(self) -> dict[str, Any]:
        return {
            "fen": self.fen,
            "side_to_move": self.side_to_move,
            "phase": self.phase,
            "action": self.action,
            "terminal": self.terminal,
            "removal_pending": self.removal_pending,
            "legal_actions": list(self.legal_actions),
            "history": self.history,
            "outcome": self.outcome.portable_record(),
        }


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
    """Verify source and fail-closed binary identity without changing Sanmill."""
    config = _strict_json_object(Path(paths_config))
    checkout = _resolve_registry_path(
        config.get("sanmill_checkout"), field="sanmill_checkout"
    )
    if not checkout.is_dir():
        raise SanmillBridgeError("sanmill_checkout is not a directory")

    head = _git_output(checkout, "rev-parse", "HEAD")
    if head != PINNED_SANMILL_COMMIT:
        raise SanmillBridgeError(
            f"Sanmill HEAD drift: expected {PINNED_SANMILL_COMMIT}, observed {head}"
        )
    dirty = _git_output(checkout, "status", "--short", "--untracked-files=all")
    if dirty:
        raise SanmillBridgeError(f"Sanmill checkout is not clean:\n{dirty}")
    tree = _git_output(checkout, "rev-parse", "HEAD^{tree}")
    if tree != PINNED_SANMILL_TREE:
        raise SanmillBridgeError(
            f"Sanmill tree drift: expected {PINNED_SANMILL_TREE}, observed {tree}"
        )

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
    if FAIL_CLOSED_ASSERTION not in binary_bytes:
        raise SanmillBridgeError(
            "Sanmill binary lacks the fail-closed assertion marker; rebuild with "
            "release debug assertions"
        )

    license_path = checkout / SANMILL_LICENSE_RELATIVE
    if not license_path.is_file():
        raise SanmillBridgeError("Sanmill license text is absent")
    license_sha256 = _sha256_file(license_path)
    if license_sha256 != EXPECTED_SANMILL_LICENSE_SHA256:
        raise SanmillBridgeError("Sanmill license identity differs from the pinned text")
    return SanmillInstallation(
        checkout=checkout,
        commit=head,
        tree=tree,
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
        ("IDSEnabled", "false"),
        ("DepthExtension", "true"),
        ("Shuffling", "false"),
        ("UseLazySmp", "false"),
        ("Algorithm", "2"),
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
        "schema_version": "nmm.sanmill-strict-uci-contract.v1",
        "sanmill_commit": PINNED_SANMILL_COMMIT,
        "command": [SANMILL_BINARY_RELATIVE.as_posix(), "mill", "uci"],
        "search_command": "go nodes <positive-N>",
        "options": {name: value for name, value in options},
        "child_environment": "inherit non-TGF variables; remove all TGF_* variables",
        "random_failure_fallback": "forbidden-by-release-debug-assertion",
        "bestmove_failure": "hard-error-no-substitution",
        "knowledge_sources": {
            "opening_book": {
                "requested_for_future_formal_baseline": True,
                "active_in_bridge_smoke": False,
                "reason": "UCI-interface-and-paired-diversity-policy-gate",
            },
            "human_database": {
                "active": False,
                "reason": "not-exposed-by-the-pinned-UCI-interface",
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
                "fallback": "release-debug-assertion",
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
    ) -> tuple[str, list[str]]:
        deadline = time.monotonic() + timeout
        seen: list[str] = []
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
            if line.startswith(_PROTOCOL_ERRORS):
                raise SanmillBridgeError(f"Sanmill protocol error: {line}")
            if predicate(line):
                return line, seen

    def _sync(self) -> list[str]:
        self._send("isready")
        _, lines = self._read_until(
            lambda line: line == "readyok",
            timeout=self.protocol_timeout,
            context="readyok",
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
        fen = self.export_fen()
        fields = fen.split()
        if len(fields) < 4 or fields[1] not in {"w", "b"}:
            raise SanmillBridgeError(f"malformed exported Sanmill FEN: {fen}")
        if fields[2] not in {"p", "m", "o"} or fields[3] not in {"p", "r", "s", "?"}:
            raise SanmillBridgeError(f"unknown Sanmill phase/action in FEN: {fen}")
        legal = self.legal_moves()
        outcome = self.debug_outcome()
        terminal = fields[2] == "o"
        if terminal != (not legal):
            raise SanmillBridgeError(
                "Sanmill terminal phase and legal-action availability disagree"
            )
        if terminal != outcome.terminal:
            raise SanmillBridgeError(
                "Sanmill exported phase and authoritative outcome disagree"
            )
        return UciPositionState(
            fen=fen,
            side_to_move=fields[1],
            phase=fields[2],
            action=fields[3],
            legal_actions=legal,
            history=self.history_summary(),
            outcome=outcome,
        )

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

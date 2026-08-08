"""Fail-closed Sanmill referee and opponent integration for training.

The historical bridge contracts in :mod:`learned_ai.evaluation.sanmill_uci`
remain immutable. This module adds the exact newer runtime required by the
fresh Sanmill-refereed lineage and keeps one complete action history for each
training game.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from game.board import BoardState
from game.rules import get_all_legal_moves, terminal_result
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_SANMILL_LICENSE_SHA256,
    EXPECTED_RULES_IDENTITY_SHA256,
    SANMILL_BINARY_RELATIVE,
    SANMILL_LICENSE_RELATIVE,
    SanmillBridgeError,
    SanmillInstallation,
    SanmillUciSession,
    UciLogicalTurnResult,
    UciPositionState,
    assert_stable_legal_parity,
    nmm_move_base,
    project_stable_sanmill_fen,
    strict_option_values,
)
from learned_ai.training.run_contract import canonical_sha256


TRAINING_SANMILL_COMMIT = "a6623f88959f7453594df274fbe1f128af7ff55e"
TRAINING_SANMILL_TREE = "17b9b0fd51ee8dac54c0454a6935978a47d19e0c"
TRAINING_SANMILL_BINARY_SHA256 = (
    "5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619"
)
TRAINING_SANMILL_BINARY_SIZE = 5_641_216
TRAINING_REFEREE_FORMAT = "SANMILL-STRICT-REFEREE-RULES/1"
TRAINING_REFEREE_PROFILE = "mif-stable-moving-v1"
TRAINING_REPETITION_OBSERVATION = "stable-moving-v1"
TRAINING_REFEREE_SEMANTIC_DIGEST = (
    "sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(checkout: Path, *arguments: str) -> str:
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
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SanmillBridgeError("cannot inspect the Sanmill training checkout") from exc
    return result.stdout.strip()


def inspect_sanmill_training_installation(
    checkout: str | Path,
) -> SanmillInstallation:
    """Verify the exact isolated source and release binary used by training."""
    root = Path(checkout).resolve(strict=False)
    if not root.is_dir():
        raise SanmillBridgeError("Sanmill training checkout is not a directory")
    if _git(root, "rev-parse", "HEAD") != TRAINING_SANMILL_COMMIT:
        raise SanmillBridgeError("Sanmill training checkout is at the wrong commit")
    if _git(root, "rev-parse", "HEAD^{tree}") != TRAINING_SANMILL_TREE:
        raise SanmillBridgeError("Sanmill training checkout has the wrong tree")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise SanmillBridgeError("Sanmill training checkout must be clean")

    binary = root / SANMILL_BINARY_RELATIVE
    if not binary.is_file():
        raise SanmillBridgeError("Sanmill training release binary is missing")
    binary_size = binary.stat().st_size
    binary_sha256 = _sha256_file(binary)
    if binary_size != TRAINING_SANMILL_BINARY_SIZE:
        raise SanmillBridgeError("Sanmill training binary size differs from the pin")
    if binary_sha256 != TRAINING_SANMILL_BINARY_SHA256:
        raise SanmillBridgeError("Sanmill training binary SHA-256 differs from the pin")

    license_path = root / SANMILL_LICENSE_RELATIVE
    if not license_path.is_file():
        raise SanmillBridgeError("Sanmill licence file is missing")
    license_sha256 = _sha256_file(license_path)
    if license_sha256 != EXPECTED_SANMILL_LICENSE_SHA256:
        raise SanmillBridgeError("Sanmill licence identity differs from the pin")

    return SanmillInstallation(
        checkout=root,
        commit=TRAINING_SANMILL_COMMIT,
        checkout_head=TRAINING_SANMILL_COMMIT,
        tree=TRAINING_SANMILL_TREE,
        binary=binary,
        binary_sha256=binary_sha256,
        binary_size=binary_size,
        license_sha256=license_sha256,
        path_lookup_key="sanmill_training_checkout",
        require_exact_head=True,
    )


def training_installation_record(
    installation: SanmillInstallation,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    record = installation.portable_record()
    record.update(
        {
            "contract_id": "nmm.sanmill-training-runtime.v1",
            "strict_referee": {
                "format": TRAINING_REFEREE_FORMAT,
                "profile": TRAINING_REFEREE_PROFILE,
                "repetitionObservation": TRAINING_REPETITION_OBSERVATION,
                "originCounted": True,
                "semanticDigest": TRAINING_REFEREE_SEMANTIC_DIGEST,
            },
            "search": {
                "command": "go logical nodes N [depth D]",
                "fixed_work": True,
                "random_failure_fallback": False,
            },
            "strict_options": dict(strict_option_values(seed)),
        }
    )
    record["identity"] = canonical_sha256(record)
    return record


def probe_sanmill_training_runtime(
    checkout: str | Path,
    *,
    node_budget: int,
    depth: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Verify identity plus cross-process deterministic logical search.

    The probe starts two new engine processes from the same fresh position.
    Timing and raw protocol text are deliberately excluded from the compared
    record; every gameplay-relevant field and the resulting referee state are
    retained.
    """
    if node_budget <= 0:
        raise SanmillBridgeError("Sanmill probe node budget must be positive")
    installation = inspect_sanmill_training_installation(checkout)
    observations: list[dict[str, Any]] = []
    for _ in range(2):
        with SanmillTrainingGame(installation, seed=seed) as game:
            applied = game.search_and_apply(
                BoardState.new_game(),
                node_budget=node_budget,
                depth=depth,
            )
            if applied.search is None:
                raise SanmillBridgeError("Sanmill probe omitted search evidence")
            search = applied.search
            observations.append(
                {
                    "actions": list(applied.actions),
                    "move": dict(applied.move),
                    "resulting_fen": search.resulting_fen,
                    "terminal": search.terminal,
                    "winner": search.winner,
                    "outcome_reason": search.outcome_reason,
                    "effective_depth": search.effective_depth,
                    "completed_depth": search.completed_depth,
                    "score_kind": search.score_kind,
                    "score": search.score,
                    "score_perspective": search.score_perspective,
                    "node_budget": search.node_budget,
                    "primary_nodes": search.primary_nodes,
                    "removal_nodes": search.removal_nodes,
                    "total_nodes": search.total_nodes,
                    "search_calls": search.search_calls,
                    "state": applied.state.portable_record(),
                }
            )
    if observations[0] != observations[1]:
        raise SanmillBridgeError(
            "Sanmill training search differs across fresh processes"
        )
    return {
        **training_installation_record(installation, seed=seed),
        "probe": {
            "fresh_processes": 2,
            "node_budget": node_budget,
            "depth": depth,
            "seed": seed,
            "deterministic": True,
            "observation_sha256": canonical_sha256(observations[0]),
            "first_turn": observations[0],
        },
    }


def nmm_move_actions(move: Mapping[str, Any]) -> tuple[str, ...]:
    actions = [nmm_move_base(move)]
    capture = move.get("capture")
    if capture is not None:
        actions.append(f"x{capture}")
    return tuple(actions)


def _normalise_move(move: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "from": move.get("from"),
        "to": move.get("to"),
        "capture": move.get("capture"),
    }


def _winner_name(color: str | None) -> str | None:
    return {None: None, "W": "white", "B": "black"}[color]


@dataclass(frozen=True)
class SanmillAppliedTurn:
    move: Mapping[str, str | None]
    actions: tuple[str, ...]
    state: UciPositionState
    search: UciLogicalTurnResult | None


class SanmillTrainingGame:
    """One complete, history-bearing Sanmill referee process."""

    def __init__(
        self,
        installation: SanmillInstallation,
        *,
        seed: int,
        session_factory: Callable[..., SanmillUciSession] = SanmillUciSession,
        protocol_timeout: float = 10.0,
        search_timeout: float = 120.0,
    ) -> None:
        self.installation = installation
        self.seed = seed
        self._session_factory = session_factory
        self._protocol_timeout = protocol_timeout
        self._search_timeout = search_timeout
        self._session: SanmillUciSession | None = None
        self._history: list[str] = []
        self._state: UciPositionState | None = None

    def __enter__(self) -> "SanmillTrainingGame":
        if self._session is not None:
            raise SanmillBridgeError("Sanmill training game is already open")
        self._session = self._session_factory(
            self.installation,
            seed=self.seed,
            protocol_timeout=self._protocol_timeout,
            search_timeout=self._search_timeout,
        )
        try:
            self._session.configure_strict_referee_profile(
                TRAINING_REFEREE_PROFILE
            )
            self._session.new_game()
            self._session.position_startpos()
            self._state = self._session.state_json()
            self._require_training_state(self._state)
            if self._state.action_token_count != 0:
                raise SanmillBridgeError("fresh Sanmill game has action history")
            if self._state.logical_ply_count != 0:
                raise SanmillBridgeError("fresh Sanmill game has logical history")
            self.assert_current_board(BoardState.new_game())
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    @property
    def state(self) -> UciPositionState:
        if self._state is None:
            raise SanmillBridgeError("Sanmill training game is not open")
        return self._state

    @property
    def session(self) -> SanmillUciSession:
        if self._session is None:
            raise SanmillBridgeError("Sanmill training game is not open")
        return self._session

    def _require_training_state(self, state: UciPositionState) -> None:
        if state.rules_identity_sha256 != EXPECTED_RULES_IDENTITY_SHA256:
            raise SanmillBridgeError("Sanmill training rules identity changed")
        identity = state.strict_referee_identity
        if identity is None:
            raise SanmillBridgeError("Sanmill omitted strict-referee identity")
        observed = identity.portable_record()
        expected = {
            "format": TRAINING_REFEREE_FORMAT,
            "profile": TRAINING_REFEREE_PROFILE,
            "repetitionObservation": TRAINING_REPETITION_OBSERVATION,
            "originCounted": True,
            "semanticDigest": TRAINING_REFEREE_SEMANTIC_DIGEST,
        }
        if observed != expected:
            raise SanmillBridgeError("Sanmill strict-referee identity changed")
        if state.removal_pending:
            raise SanmillBridgeError(
                "training may observe only complete logical-turn boundaries"
            )

    def assert_current_board(self, board: BoardState) -> None:
        state = self.state
        self._require_training_state(state)
        projected = project_stable_sanmill_fen(
            state.fen,
            terminal=state.terminal,
        )
        if projected.to_fen_string() != board.to_fen_string():
            raise SanmillBridgeError("Sanmill and NMM board mirrors diverged")

        local_terminal, local_winner, _ = terminal_result(board)
        if local_terminal:
            if not state.terminal or state.winner != _winner_name(local_winner):
                raise SanmillBridgeError(
                    "Sanmill and NMM board-terminal results diverged"
                )
        elif state.terminal and state.winner is not None:
            raise SanmillBridgeError(
                "Sanmill reported a board-terminal win absent from NMM mirror"
            )

        if state.terminal:
            if state.legal_actions:
                raise SanmillBridgeError("terminal Sanmill state advertises actions")
        else:
            assert_stable_legal_parity(board, state.legal_actions)

    def apply_nmm_move(
        self,
        board: BoardState,
        move: Mapping[str, Any],
        *,
        search_result: UciLogicalTurnResult | None = None,
    ) -> SanmillAppliedTurn:
        self.assert_current_board(board)
        if self.state.terminal:
            raise SanmillBridgeError("cannot apply a move after Sanmill termination")
        normalised = _normalise_move(move)
        legal = [_normalise_move(candidate) for candidate in get_all_legal_moves(board)]
        if normalised not in legal:
            raise SanmillBridgeError("NMM attempted an illegal training move")
        actions = nmm_move_actions(normalised)
        if search_result is not None:
            if search_result.full_turn_actions != actions:
                raise SanmillBridgeError(
                    "Sanmill search actions differ from the selected NMM move"
                )
            if dict(search_result.model_action or {}) != normalised:
                raise SanmillBridgeError(
                    "Sanmill search model action differs from the selected move"
                )

        previous = self.state
        self._history.extend(actions)
        self.session.position_startpos(self._history)
        state = self.session.state_json()
        self._require_training_state(state)
        if state.action_token_count != len(self._history):
            raise SanmillBridgeError("Sanmill action-token count drifted")
        if state.logical_ply_count != previous.logical_ply_count + 1:
            raise SanmillBridgeError("Sanmill logical-ply count drifted")
        expected_by_side = list(previous.logical_plies_by_side)
        expected_by_side[0 if board.turn == "W" else 1] += 1
        if state.logical_plies_by_side != tuple(expected_by_side):
            raise SanmillBridgeError("Sanmill per-side logical counts drifted")
        if state.history_sha256 == previous.history_sha256:
            raise SanmillBridgeError("Sanmill history identity did not advance")

        if search_result is not None:
            if search_result.resulting_fen != state.fen:
                raise SanmillBridgeError("Sanmill search/replay FEN mismatch")
            if search_result.terminal != state.terminal:
                raise SanmillBridgeError("Sanmill search/replay terminal mismatch")
            if search_result.winner != state.winner:
                raise SanmillBridgeError("Sanmill search/replay winner mismatch")
            if search_result.outcome_reason != state.outcome_reason:
                raise SanmillBridgeError("Sanmill search/replay reason mismatch")

        self._state = state
        self.assert_current_board(board.apply_move(normalised))
        return SanmillAppliedTurn(
            move=normalised,
            actions=actions,
            state=state,
            search=search_result,
        )

    def search_and_apply(
        self,
        board: BoardState,
        *,
        node_budget: int,
        depth: int | None = None,
    ) -> SanmillAppliedTurn:
        self.assert_current_board(board)
        if self.state.terminal:
            raise SanmillBridgeError("cannot search after Sanmill termination")
        result = self.session.search_logical_turn(node_budget, depth=depth)
        if result.status != "ok" or result.model_action is None:
            raise SanmillBridgeError("ongoing Sanmill root produced no move")
        return self.apply_nmm_move(
            board,
            result.model_action,
            search_result=result,
        )


class SanmillTrainingOpponent:
    """Trainer-compatible opponent whose search is committed by the referee."""

    def __init__(
        self,
        game: SanmillTrainingGame,
        *,
        node_budget: int,
        depth: int | None = None,
    ) -> None:
        if node_budget <= 0:
            raise ValueError("Sanmill node budget must be positive")
        self.game = game
        self.node_budget = node_budget
        self.depth = depth
        self.last_search_nodes = 0
        self.last_search_depth = 0
        self._last_turn: SanmillAppliedTurn | None = None

    def choose_move(self, board: BoardState) -> dict[str, str | None]:
        if self._last_turn is not None:
            raise SanmillBridgeError(
                "previous Sanmill opponent turn was not consumed"
            )
        turn = self.game.search_and_apply(
            board,
            node_budget=self.node_budget,
            depth=self.depth,
        )
        if turn.search is None:
            raise SanmillBridgeError("Sanmill opponent omitted search evidence")
        self.last_search_nodes = turn.search.total_nodes
        self.last_search_depth = turn.search.completed_depth or 0
        self._last_turn = turn
        return dict(turn.move)

    def consume_committed_turn(self, move: Mapping[str, Any]) -> SanmillAppliedTurn:
        turn = self._last_turn
        self._last_turn = None
        if turn is None or dict(turn.move) != _normalise_move(move):
            raise SanmillBridgeError(
                "Sanmill opponent move was not committed exactly once"
            )
        return turn

"""Exhaustive, host-path-free eight-ply Sanmill opening-book paths."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from learned_ai.evaluation.sanmill_data_query import (
    DataQueryCandidate,
    DataQueryState,
    SanmillDataQuerySession,
    portable_source_identity,
)
from learned_ai.evaluation.sanmill_prefix import (
    PREFIX_LOGICAL_PLIES,
    SanmillPrefixError,
)
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_OPENING_BOOK_ORACLE_ENTRIES,
    EXPECTED_OPENING_BOOK_RECOMMENDATIONS,
    EXPECTED_OPENING_BOOK_SHA256,
    EXPECTED_SANMILL_BINARY_SHA256,
    EXPECTED_SANMILL_BINARY_SIZE,
    EXPECTED_SANMILL_LICENSE_SHA256,
    PINNED_SANMILL_COMMIT,
    PINNED_SANMILL_TREE,
    SANMILL_BINARY_RELATIVE,
    SANMILL_LICENSE_RELATIVE,
    STRICT_BUILD_COMMAND,
    STRICT_PROTOCOL_VERSION,
    SanmillInstallation,
    validate_uci_action_token,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


BOOK_PATH_CORPUS_SCHEMA = "nmm.sanmill-book-path-corpus.v1"
BOOK_PATH_SCHEMA = "nmm.sanmill-complete-book-path.v1"
BOOK_LEAF_SCHEMA = "nmm.sanmill-incomplete-book-leaf.v1"
BOOK_ENUMERATION_ALGORITHM = "exhaustive-action-history-bfs-v1"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SanmillBookPathError(SanmillPrefixError):
    """Raised when enumeration, replay, identity, or corpus validation fails."""


def _object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SanmillBookPathError(f"{context} must be an object")
    return value


def _fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise SanmillBookPathError(f"{context} fields: {'; '.join(details)}")


def _string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SanmillBookPathError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context=context)


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise SanmillBookPathError(
            f"{context} must be an integer of at least {minimum}"
        )
    return value


def _optional_integer(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    return _integer(value, context=context)


def _boolean(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise SanmillBookPathError(f"{context} must be a boolean")
    return value


def _sha256(value: Any, *, context: str) -> str:
    digest = _string(value, context=context)
    if not _SHA256.fullmatch(digest):
        raise SanmillBookPathError(f"{context} must be a lowercase SHA-256")
    return digest


def _sha40(value: Any, *, context: str) -> str:
    digest = _string(value, context=context)
    if not _SHA40.fullmatch(digest):
        raise SanmillBookPathError(f"{context} must be a full lowercase Git SHA")
    return digest


def _string_list(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise SanmillBookPathError(f"{context} must be a string array")
    return tuple(value)


def _action_tokens(value: Any, *, context: str) -> tuple[str, ...]:
    tokens = _string_list(value, context=context)
    try:
        return tuple(validate_uci_action_token(token) for token in tokens)
    except SanmillPrefixError:
        raise
    except Exception as exc:
        raise SanmillBookPathError(f"{context} contains an invalid token") from exc


def _two_counts(value: Any, *, context: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise SanmillBookPathError(f"{context} must contain two counts")
    return (
        _integer(value[0], context=f"{context}[0]"),
        _integer(value[1], context=f"{context}[1]"),
    )


def _expected_side_counts(logical_count: int) -> tuple[int, int]:
    return ((logical_count + 1) // 2, logical_count // 2)


def _validate_outcome(
    *,
    terminal: bool,
    kind: str,
    winner: int | None,
    reason: str,
    context: str,
) -> None:
    if winner not in {None, 0, 1}:
        raise SanmillBookPathError(f"{context} winner is unsupported")
    if terminal != (kind != "ongoing"):
        raise SanmillBookPathError(f"{context} terminal flag disagrees with kind")
    if kind == "ongoing" and (winner is not None or reason != "ongoing"):
        raise SanmillBookPathError(f"{context} ongoing outcome is inconsistent")
    if kind == "win" and winner not in {0, 1}:
        raise SanmillBookPathError(f"{context} win lacks a winner")
    if kind == "draw" and winner is not None:
        raise SanmillBookPathError(f"{context} draw names a winner")


def _validate_sanmill_record(value: Mapping[str, Any]) -> None:
    payload = _object(value, context="book corpus Sanmill")
    _fields(
        payload,
        required={
            "path_lookup_key",
            "commit",
            "checkout_head",
            "checkout_policy",
            "tree",
            "binary_relative_path",
            "binary_sha256",
            "binary_size",
            "build_command",
            "strict_failure_protocol_version",
            "license",
        },
        context="book corpus Sanmill",
    )
    if payload["path_lookup_key"] != "sanmill_checkout":
        raise SanmillBookPathError("Sanmill path lookup key changed")
    if _sha40(payload["commit"], context="Sanmill commit") != PINNED_SANMILL_COMMIT:
        raise SanmillBookPathError("Sanmill pinned commit changed")
    _sha40(payload["checkout_head"], context="Sanmill checkout head")
    if _sha40(payload["tree"], context="Sanmill tree") != PINNED_SANMILL_TREE:
        raise SanmillBookPathError("Sanmill pinned tree changed")
    if payload["binary_relative_path"] != SANMILL_BINARY_RELATIVE.as_posix():
        raise SanmillBookPathError("Sanmill binary relative path changed")
    if (
        _sha256(payload["binary_sha256"], context="Sanmill binary SHA-256")
        != EXPECTED_SANMILL_BINARY_SHA256
        or _integer(
            payload["binary_size"],
            context="Sanmill binary size",
            minimum=1,
        )
        != EXPECTED_SANMILL_BINARY_SIZE
    ):
        raise SanmillBookPathError("Sanmill binary identity changed")
    if payload["build_command"] != STRICT_BUILD_COMMAND:
        raise SanmillBookPathError("Sanmill build command changed")
    if payload["strict_failure_protocol_version"] != STRICT_PROTOCOL_VERSION:
        raise SanmillBookPathError("Sanmill strict protocol version changed")
    _string(payload["checkout_policy"], context="Sanmill checkout policy")
    license_record = _object(payload["license"], context="Sanmill license")
    _fields(
        license_record,
        required={"spdx", "relative_path", "sha256"},
        context="Sanmill license",
    )
    if license_record["spdx"] != "AGPL-3.0-or-later":
        raise SanmillBookPathError("Sanmill license identifier changed")
    if license_record["relative_path"] != SANMILL_LICENSE_RELATIVE.as_posix():
        raise SanmillBookPathError("Sanmill license relative path changed")
    if (
        _sha256(license_record["sha256"], context="Sanmill license SHA-256")
        != EXPECTED_SANMILL_LICENSE_SHA256
    ):
        raise SanmillBookPathError("Sanmill license identity changed")


def _validate_book_source_identity(value: Mapping[str, Any]) -> None:
    source = _object(value, context="book source identity")
    _fields(
        source,
        required={"kind", "identity", "identity_sha256"},
        context="book source identity",
    )
    if source["kind"] != "book":
        raise SanmillBookPathError("book corpus has the wrong source kind")
    identity = _object(source["identity"], context="book content identity")
    _fields(
        identity,
        required={
            "kind",
            "schema_version",
            "variant",
            "symmetry",
            "sha256",
            "byte_length",
            "oracle_positions",
            "oracle_records",
            "source",
        },
        context="book content identity",
    )
    if (
        identity["kind"] != "opening_book"
        or identity["schema_version"] != 1
        or identity["variant"] != "nmm"
        or identity["source"] != "bundled"
        or identity["oracle_positions"] != EXPECTED_OPENING_BOOK_ORACLE_ENTRIES
        or identity["oracle_records"]
        != EXPECTED_OPENING_BOOK_RECOMMENDATIONS
    ):
        raise SanmillBookPathError("book content contract changed")
    _string(identity["symmetry"], context="book symmetry")
    if (
        _sha256(identity["sha256"], context="book content SHA-256")
        != EXPECTED_OPENING_BOOK_SHA256
    ):
        raise SanmillBookPathError("book content identity changed")
    for field in ("byte_length", "oracle_positions", "oracle_records"):
        _integer(identity[field], context=f"book content {field}", minimum=1)
    identity_sha256 = _sha256(
        source["identity_sha256"],
        context="book source identity SHA-256",
    )
    if canonical_sha256(identity) != identity_sha256:
        raise SanmillBookPathError("book source identity digest mismatch")


@dataclass(frozen=True)
class BookPathStep:
    logical_ply: int
    side: str
    candidate_logical_move_id: str
    candidate_stable_index: int
    candidate_source_group_id: str
    candidate_source_rank: int
    action_tokens: tuple[str, ...]
    compound_turn: bool
    input_fen: str
    input_history_sha256: str
    output_fen: str
    output_history_sha256: str
    output_action_token_count: int
    output_logical_ply_count: int
    output_logical_plies_by_side: tuple[int, int]
    output_terminal: bool
    output_outcome_kind: str
    output_winner: int | None
    output_reason: str

    def __post_init__(self) -> None:
        if self.logical_ply < 0 or self.logical_ply >= PREFIX_LOGICAL_PLIES:
            raise SanmillBookPathError("book-path step index is outside the prefix")
        if self.side != ("white" if self.logical_ply % 2 == 0 else "black"):
            raise SanmillBookPathError("book-path step side does not alternate")
        if not self.action_tokens or len(self.action_tokens) not in {1, 2}:
            raise SanmillBookPathError("book-path step has invalid action count")
        for token in self.action_tokens:
            try:
                validate_uci_action_token(token)
            except Exception as exc:
                raise SanmillBookPathError(
                    "book-path step has an invalid action token"
                ) from exc
        if (
            self.action_tokens[0].startswith("x")
            or (
                len(self.action_tokens) == 2
                and not self.action_tokens[1].startswith("x")
            )
        ):
            raise SanmillBookPathError(
                "book-path step is not a primary-plus-removal turn"
            )
        if self.compound_turn != (len(self.action_tokens) == 2):
            raise SanmillBookPathError("book-path compound flag is inconsistent")
        if not self.candidate_logical_move_id.startswith("book:"):
            raise SanmillBookPathError("book-path candidate has the wrong source")
        if self.candidate_stable_index < 0 or self.candidate_source_rank < 1:
            raise SanmillBookPathError("book-path candidate rank is invalid")
        if not self.candidate_source_group_id:
            raise SanmillBookPathError("book-path candidate group is empty")
        if self.output_logical_ply_count != self.logical_ply + 1:
            raise SanmillBookPathError("book-path step logical count drifted")
        if self.output_logical_plies_by_side != _expected_side_counts(
            self.output_logical_ply_count
        ):
            raise SanmillBookPathError("book-path step side counts drifted")
        _sha256(
            self.candidate_logical_move_id.split(":", 1)[-1],
            context="candidate logical move digest",
        )
        _sha256(self.input_history_sha256, context="step input history")
        _sha256(self.output_history_sha256, context="step output history")
        _validate_outcome(
            terminal=self.output_terminal,
            kind=self.output_outcome_kind,
            winner=self.output_winner,
            reason=self.output_reason,
            context="book-path step output",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_ply": self.logical_ply,
            "side": self.side,
            "candidate": {
                "logical_move_id": self.candidate_logical_move_id,
                "stable_index": self.candidate_stable_index,
                "source_group_id": self.candidate_source_group_id,
                "source_rank": self.candidate_source_rank,
            },
            "action_tokens": list(self.action_tokens),
            "compound_turn": self.compound_turn,
            "input": {
                "fen": self.input_fen,
                "history_sha256": self.input_history_sha256,
            },
            "output": {
                "fen": self.output_fen,
                "history_sha256": self.output_history_sha256,
                "action_token_count": self.output_action_token_count,
                "logical_ply_count": self.output_logical_ply_count,
                "logical_plies_by_side": list(
                    self.output_logical_plies_by_side
                ),
                "terminal": self.output_terminal,
                "outcome_kind": self.output_outcome_kind,
                "winner": self.output_winner,
                "reason": self.output_reason,
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> "BookPathStep":
        payload = _object(value, context="book-path step")
        _fields(
            payload,
            required={
                "logical_ply",
                "side",
                "candidate",
                "action_tokens",
                "compound_turn",
                "input",
                "output",
            },
            context="book-path step",
        )
        candidate = _object(payload["candidate"], context="step candidate")
        _fields(
            candidate,
            required={
                "logical_move_id",
                "stable_index",
                "source_group_id",
                "source_rank",
            },
            context="step candidate",
        )
        input_state = _object(payload["input"], context="step input")
        _fields(
            input_state,
            required={"fen", "history_sha256"},
            context="step input",
        )
        output = _object(payload["output"], context="step output")
        _fields(
            output,
            required={
                "fen",
                "history_sha256",
                "action_token_count",
                "logical_ply_count",
                "logical_plies_by_side",
                "terminal",
                "outcome_kind",
                "winner",
                "reason",
            },
            context="step output",
        )
        winner = output["winner"]
        if winner is not None:
            winner = _integer(winner, context="step output winner")
            if winner not in {0, 1}:
                raise SanmillBookPathError("step output winner is unsupported")
        return cls(
            logical_ply=_integer(
                payload["logical_ply"],
                context="step logical_ply",
            ),
            side=_string(payload["side"], context="step side"),
            candidate_logical_move_id=_string(
                candidate["logical_move_id"],
                context="candidate logical_move_id",
            ),
            candidate_stable_index=_integer(
                candidate["stable_index"],
                context="candidate stable_index",
            ),
            candidate_source_group_id=_string(
                candidate["source_group_id"],
                context="candidate source_group_id",
            ),
            candidate_source_rank=_integer(
                candidate["source_rank"],
                context="candidate source_rank",
                minimum=1,
            ),
            action_tokens=_action_tokens(
                payload["action_tokens"],
                context="step action_tokens",
            ),
            compound_turn=_boolean(
                payload["compound_turn"],
                context="step compound_turn",
            ),
            input_fen=_string(input_state["fen"], context="step input fen"),
            input_history_sha256=_sha256(
                input_state["history_sha256"],
                context="step input history",
            ),
            output_fen=_string(output["fen"], context="step output fen"),
            output_history_sha256=_sha256(
                output["history_sha256"],
                context="step output history",
            ),
            output_action_token_count=_integer(
                output["action_token_count"],
                context="step output action count",
            ),
            output_logical_ply_count=_integer(
                output["logical_ply_count"],
                context="step output logical count",
            ),
            output_logical_plies_by_side=_two_counts(
                output["logical_plies_by_side"],
                context="step output side counts",
            ),
            output_terminal=_boolean(
                output["terminal"],
                context="step output terminal",
            ),
            output_outcome_kind=_string(
                output["outcome_kind"],
                context="step output outcome kind",
            ),
            output_winner=winner,
            output_reason=_string(
                output["reason"],
                context="step output reason",
            ),
        )


@dataclass(frozen=True)
class CompleteBookPath:
    action_tokens: tuple[str, ...]
    steps: tuple[BookPathStep, ...]
    final_fen: str
    final_history_sha256: str
    final_action_token_count: int
    final_terminal: bool
    final_outcome_kind: str
    final_winner: int | None
    final_reason: str
    path_identity: str = ""

    def __post_init__(self) -> None:
        if len(self.steps) != PREFIX_LOGICAL_PLIES:
            raise SanmillBookPathError("complete book path must have eight steps")
        if [step.logical_ply for step in self.steps] != list(
            range(PREFIX_LOGICAL_PLIES)
        ):
            raise SanmillBookPathError("book path step indices are not contiguous")
        flattened = tuple(
            action for step in self.steps for action in step.action_tokens
        )
        if flattened != self.action_tokens:
            raise SanmillBookPathError("book path action history is inconsistent")
        if self.final_action_token_count != len(self.action_tokens):
            raise SanmillBookPathError("book path final action count is inconsistent")
        cumulative_action_count = 0
        for step in self.steps:
            cumulative_action_count += len(step.action_tokens)
            if step.output_action_token_count != cumulative_action_count:
                raise SanmillBookPathError(
                    "book path step action count is inconsistent"
                )
        for index in range(1, len(self.steps)):
            before = self.steps[index]
            previous = self.steps[index - 1]
            if (
                before.input_fen != previous.output_fen
                or before.input_history_sha256
                != previous.output_history_sha256
            ):
                raise SanmillBookPathError("book path state chain is inconsistent")
            if previous.output_terminal:
                raise SanmillBookPathError(
                    "book path continues after a terminal state"
                )
        final = self.steps[-1]
        if (
            self.final_fen != final.output_fen
            or self.final_history_sha256 != final.output_history_sha256
            or self.final_action_token_count != final.output_action_token_count
            or self.final_terminal != final.output_terminal
            or self.final_outcome_kind != final.output_outcome_kind
            or self.final_winner != final.output_winner
            or self.final_reason != final.output_reason
        ):
            raise SanmillBookPathError("book path final state is inconsistent")
        _validate_outcome(
            terminal=self.final_terminal,
            kind=self.final_outcome_kind,
            winner=self.final_winner,
            reason=self.final_reason,
            context="book path final",
        )
        expected = canonical_sha256(self._identity_body())
        if self.path_identity and self.path_identity != expected:
            raise SanmillBookPathError("book path identity mismatch")
        object.__setattr__(self, "path_identity", expected)

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema_version": BOOK_PATH_SCHEMA,
            "action_tokens": list(self.action_tokens),
            "logical_ply_count": PREFIX_LOGICAL_PLIES,
            "logical_plies_by_side": [4, 4],
            "steps": [step.to_dict() for step in self.steps],
            "final": {
                "fen": self.final_fen,
                "history_sha256": self.final_history_sha256,
                "action_token_count": self.final_action_token_count,
                "terminal": self.final_terminal,
                "outcome_kind": self.final_outcome_kind,
                "winner": self.final_winner,
                "reason": self.final_reason,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_body(), "path_identity": self.path_identity}

    @classmethod
    def from_dict(cls, value: Any) -> "CompleteBookPath":
        payload = _object(value, context="complete book path")
        _fields(
            payload,
            required={
                "schema_version",
                "action_tokens",
                "logical_ply_count",
                "logical_plies_by_side",
                "steps",
                "final",
                "path_identity",
            },
            context="complete book path",
        )
        if payload["schema_version"] != BOOK_PATH_SCHEMA:
            raise SanmillBookPathError("unsupported book-path schema")
        if payload["logical_ply_count"] != PREFIX_LOGICAL_PLIES:
            raise SanmillBookPathError("book-path logical count changed")
        if payload["logical_plies_by_side"] != [4, 4]:
            raise SanmillBookPathError("book-path side counts changed")
        if not isinstance(payload["steps"], list):
            raise SanmillBookPathError("book-path steps must be an array")
        final = _object(payload["final"], context="book-path final state")
        _fields(
            final,
            required={
                "fen",
                "history_sha256",
                "action_token_count",
                "terminal",
                "outcome_kind",
                "winner",
                "reason",
            },
            context="book-path final state",
        )
        winner = final["winner"]
        if winner is not None:
            winner = _integer(winner, context="book-path final winner")
            if winner not in {0, 1}:
                raise SanmillBookPathError("book-path final winner is unsupported")
        return cls(
            action_tokens=_action_tokens(
                payload["action_tokens"],
                context="book-path action_tokens",
            ),
            steps=tuple(BookPathStep.from_dict(item) for item in payload["steps"]),
            final_fen=_string(final["fen"], context="book-path final fen"),
            final_history_sha256=_sha256(
                final["history_sha256"],
                context="book-path final history",
            ),
            final_action_token_count=_integer(
                final["action_token_count"],
                context="book-path final action count",
            ),
            final_terminal=_boolean(
                final["terminal"],
                context="book-path final terminal",
            ),
            final_outcome_kind=_string(
                final["outcome_kind"],
                context="book-path final outcome kind",
            ),
            final_winner=winner,
            final_reason=_string(
                final["reason"],
                context="book-path final reason",
            ),
            path_identity=_sha256(
                payload["path_identity"],
                context="book path identity",
            ),
        )


@dataclass(frozen=True)
class IncompleteBookLeaf:
    kind: str
    logical_ply_count: int
    action_tokens: tuple[str, ...]
    fen: str
    history_sha256: str
    outcome_kind: str
    winner: int | None
    reason: str
    leaf_identity: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"book_miss", "terminal"}:
            raise SanmillBookPathError("unknown incomplete leaf kind")
        if self.logical_ply_count >= PREFIX_LOGICAL_PLIES:
            raise SanmillBookPathError("incomplete leaf reached complete depth")
        if self.logical_ply_count != sum(
            not token.startswith("x") for token in self.action_tokens
        ):
            raise SanmillBookPathError("incomplete leaf logical count drifted")
        for token in self.action_tokens:
            try:
                validate_uci_action_token(token)
            except Exception as exc:
                raise SanmillBookPathError(
                    "incomplete leaf has an invalid action token"
                ) from exc
        _sha256(self.history_sha256, context="incomplete leaf history")
        if self.kind == "book_miss" and self.outcome_kind != "ongoing":
            raise SanmillBookPathError("book-miss leaf is unexpectedly terminal")
        if self.kind == "terminal" and self.outcome_kind == "ongoing":
            raise SanmillBookPathError("terminal leaf has an ongoing outcome")
        _validate_outcome(
            terminal=self.kind == "terminal",
            kind=self.outcome_kind,
            winner=self.winner,
            reason=self.reason,
            context="incomplete leaf",
        )
        expected = canonical_sha256(self._identity_body())
        if self.leaf_identity and self.leaf_identity != expected:
            raise SanmillBookPathError("incomplete leaf identity mismatch")
        object.__setattr__(self, "leaf_identity", expected)

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema_version": BOOK_LEAF_SCHEMA,
            "kind": self.kind,
            "logical_ply_count": self.logical_ply_count,
            "action_tokens": list(self.action_tokens),
            "fen": self.fen,
            "history_sha256": self.history_sha256,
            "outcome": {
                "kind": self.outcome_kind,
                "winner": self.winner,
                "reason": self.reason,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_body(), "leaf_identity": self.leaf_identity}

    @classmethod
    def from_dict(cls, value: Any) -> "IncompleteBookLeaf":
        payload = _object(value, context="incomplete book leaf")
        _fields(
            payload,
            required={
                "schema_version",
                "kind",
                "logical_ply_count",
                "action_tokens",
                "fen",
                "history_sha256",
                "outcome",
                "leaf_identity",
            },
            context="incomplete book leaf",
        )
        if payload["schema_version"] != BOOK_LEAF_SCHEMA:
            raise SanmillBookPathError("unsupported book-leaf schema")
        outcome = _object(payload["outcome"], context="book-leaf outcome")
        _fields(
            outcome,
            required={"kind", "winner", "reason"},
            context="book-leaf outcome",
        )
        winner = outcome["winner"]
        if winner is not None:
            winner = _integer(winner, context="book-leaf winner")
            if winner not in {0, 1}:
                raise SanmillBookPathError("book-leaf winner is unsupported")
        return cls(
            kind=_string(payload["kind"], context="book-leaf kind"),
            logical_ply_count=_integer(
                payload["logical_ply_count"],
                context="book-leaf logical count",
            ),
            action_tokens=_action_tokens(
                payload["action_tokens"],
                context="book-leaf action_tokens",
            ),
            fen=_string(payload["fen"], context="book-leaf fen"),
            history_sha256=_sha256(
                payload["history_sha256"],
                context="book-leaf history",
            ),
            outcome_kind=_string(
                outcome["kind"],
                context="book-leaf outcome kind",
            ),
            winner=winner,
            reason=_string(outcome["reason"], context="book-leaf reason"),
            leaf_identity=_sha256(
                payload["leaf_identity"],
                context="book-leaf identity",
            ),
        )


@dataclass(frozen=True)
class BookDepthAudit:
    input_logical_ply: int
    input_prefix_count: int
    available_input_count: int
    book_miss_input_count: int
    terminal_input_count: int
    candidate_edge_count: int
    unique_child_history_count: int
    compound_edge_count: int

    def __post_init__(self) -> None:
        values = (
            self.input_logical_ply,
            self.input_prefix_count,
            self.available_input_count,
            self.book_miss_input_count,
            self.terminal_input_count,
            self.candidate_edge_count,
            self.unique_child_history_count,
            self.compound_edge_count,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in values
        ):
            raise SanmillBookPathError("depth audit contains an invalid count")
        if (
            self.available_input_count
            + self.book_miss_input_count
            + self.terminal_input_count
            != self.input_prefix_count
        ):
            raise SanmillBookPathError("depth input status counts do not sum")
        if self.candidate_edge_count != self.unique_child_history_count:
            raise SanmillBookPathError("depth child histories are not unique")
        if self.compound_edge_count > self.candidate_edge_count:
            raise SanmillBookPathError("depth compound count exceeds edges")

    def to_dict(self) -> dict[str, int]:
        return {
            "input_logical_ply": self.input_logical_ply,
            "input_prefix_count": self.input_prefix_count,
            "available_input_count": self.available_input_count,
            "book_miss_input_count": self.book_miss_input_count,
            "terminal_input_count": self.terminal_input_count,
            "candidate_edge_count": self.candidate_edge_count,
            "unique_child_history_count": self.unique_child_history_count,
            "compound_edge_count": self.compound_edge_count,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "BookDepthAudit":
        payload = _object(value, context="book depth audit")
        fields = {
            "input_logical_ply",
            "input_prefix_count",
            "available_input_count",
            "book_miss_input_count",
            "terminal_input_count",
            "candidate_edge_count",
            "unique_child_history_count",
            "compound_edge_count",
        }
        _fields(payload, required=fields, context="book depth audit")
        return cls(
            **{
                field: _integer(payload[field], context=f"depth audit {field}")
                for field in fields
            }
        )


@dataclass(frozen=True)
class BookPathCorpus:
    generator_commit: str
    sanmill: Mapping[str, Any]
    source_identity: Mapping[str, Any]
    depth_audit: tuple[BookDepthAudit, ...]
    incomplete_leaves: tuple[IncompleteBookLeaf, ...]
    paths: tuple[CompleteBookPath, ...]
    corpus_identity: str = ""

    def __post_init__(self) -> None:
        _sha40(self.generator_commit, context="generator commit")
        if len(self.depth_audit) != PREFIX_LOGICAL_PLIES:
            raise SanmillBookPathError("book corpus needs eight depth audits")
        if [item.input_logical_ply for item in self.depth_audit] != list(
            range(PREFIX_LOGICAL_PLIES)
        ):
            raise SanmillBookPathError("book corpus depth audits are not contiguous")
        if self.depth_audit[0].input_prefix_count != 1:
            raise SanmillBookPathError("book corpus must start with one prefix")
        for index in range(1, len(self.depth_audit)):
            if (
                self.depth_audit[index].input_prefix_count
                != self.depth_audit[index - 1].unique_child_history_count
            ):
                raise SanmillBookPathError("book corpus frontier counts drifted")
        if (
            self.depth_audit[-1].unique_child_history_count
            != len(self.paths)
        ):
            raise SanmillBookPathError("book corpus final path count drifted")
        expected_incomplete = sum(
            item.book_miss_input_count + item.terminal_input_count
            for item in self.depth_audit
        )
        if expected_incomplete != len(self.incomplete_leaves):
            raise SanmillBookPathError("book corpus incomplete-leaf count drifted")
        path_histories = [path.action_tokens for path in self.paths]
        if path_histories != sorted(path_histories):
            raise SanmillBookPathError("book corpus paths are not sorted")
        if len(path_histories) != len(set(path_histories)):
            raise SanmillBookPathError("book corpus contains duplicate paths")
        leaf_order = [
            (leaf.logical_ply_count, leaf.kind, leaf.action_tokens)
            for leaf in self.incomplete_leaves
        ]
        if leaf_order != sorted(leaf_order):
            raise SanmillBookPathError("book corpus leaves are not sorted")
        leaf_histories = [leaf.action_tokens for leaf in self.incomplete_leaves]
        if len(leaf_histories) != len(set(leaf_histories)):
            raise SanmillBookPathError("book corpus contains duplicate leaves")
        if set(path_histories) & set(leaf_histories):
            raise SanmillBookPathError("complete and incomplete histories overlap")
        _validate_sanmill_record(self.sanmill)
        _validate_book_source_identity(self.source_identity)
        expected = canonical_sha256(self._identity_body())
        if self.corpus_identity and self.corpus_identity != expected:
            raise SanmillBookPathError("book corpus identity mismatch")
        object.__setattr__(self, "corpus_identity", expected)

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema_version": BOOK_PATH_CORPUS_SCHEMA,
            "generator": {
                "nmm_llm_commit": self.generator_commit,
                "algorithm": BOOK_ENUMERATION_ALGORITHM,
            },
            "sanmill": dict(self.sanmill),
            "source_identity": dict(self.source_identity),
            "logical_ply_count": PREFIX_LOGICAL_PLIES,
            "logical_plies_by_side": [4, 4],
            "path_equivalence": "exact-chronological-action-token-sequence",
            "path_order": "lexicographic-action-token-sequence",
            "fallback": "none",
            "policy_status": "inventory-only-unfrozen-for-evaluation",
            "summary": {
                "complete_path_count": len(self.paths),
                "incomplete_leaf_count": len(self.incomplete_leaves),
                "book_miss_leaf_count": sum(
                    leaf.kind == "book_miss" for leaf in self.incomplete_leaves
                ),
                "terminal_leaf_count": sum(
                    leaf.kind == "terminal" for leaf in self.incomplete_leaves
                ),
                "compound_edge_count": sum(
                    item.compound_edge_count for item in self.depth_audit
                ),
            },
            "depth_audit": [item.to_dict() for item in self.depth_audit],
            "incomplete_leaves": [
                item.to_dict() for item in self.incomplete_leaves
            ],
            "paths": [item.to_dict() for item in self.paths],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_body(), "corpus_identity": self.corpus_identity}

    @classmethod
    def from_dict(cls, value: Any) -> "BookPathCorpus":
        payload = _object(value, context="book path corpus")
        _fields(
            payload,
            required={
                "schema_version",
                "generator",
                "sanmill",
                "source_identity",
                "logical_ply_count",
                "logical_plies_by_side",
                "path_equivalence",
                "path_order",
                "fallback",
                "policy_status",
                "summary",
                "depth_audit",
                "incomplete_leaves",
                "paths",
                "corpus_identity",
            },
            context="book path corpus",
        )
        if payload["schema_version"] != BOOK_PATH_CORPUS_SCHEMA:
            raise SanmillBookPathError("unsupported book-corpus schema")
        if (
            payload["logical_ply_count"] != PREFIX_LOGICAL_PLIES
            or payload["logical_plies_by_side"] != [4, 4]
            or payload["path_equivalence"]
            != "exact-chronological-action-token-sequence"
            or payload["path_order"] != "lexicographic-action-token-sequence"
            or payload["fallback"] != "none"
            or payload["policy_status"]
            != "inventory-only-unfrozen-for-evaluation"
        ):
            raise SanmillBookPathError("book-corpus fixed contract changed")
        generator = _object(payload["generator"], context="book corpus generator")
        _fields(
            generator,
            required={"nmm_llm_commit", "algorithm"},
            context="book corpus generator",
        )
        if generator["algorithm"] != BOOK_ENUMERATION_ALGORITHM:
            raise SanmillBookPathError("book-corpus algorithm changed")
        for field in ("depth_audit", "incomplete_leaves", "paths"):
            if not isinstance(payload[field], list):
                raise SanmillBookPathError(f"book corpus {field} must be an array")
        corpus = cls(
            generator_commit=_sha40(
                generator["nmm_llm_commit"],
                context="book corpus generator commit",
            ),
            sanmill=_object(payload["sanmill"], context="book corpus Sanmill"),
            source_identity=_object(
                payload["source_identity"],
                context="book corpus source identity",
            ),
            depth_audit=tuple(
                BookDepthAudit.from_dict(item) for item in payload["depth_audit"]
            ),
            incomplete_leaves=tuple(
                IncompleteBookLeaf.from_dict(item)
                for item in payload["incomplete_leaves"]
            ),
            paths=tuple(
                CompleteBookPath.from_dict(item) for item in payload["paths"]
            ),
            corpus_identity=_sha256(
                payload["corpus_identity"],
                context="book corpus identity",
            ),
        )
        summary = _object(payload["summary"], context="book corpus summary")
        _fields(
            summary,
            required={
                "complete_path_count",
                "incomplete_leaf_count",
                "book_miss_leaf_count",
                "terminal_leaf_count",
                "compound_edge_count",
            },
            context="book corpus summary",
        )
        expected_summary = corpus._identity_body()["summary"]
        if summary != expected_summary:
            raise SanmillBookPathError("book corpus summary does not recompute")
        return corpus


@dataclass(frozen=True)
class _FrontierNode:
    actions: tuple[str, ...]
    steps: tuple[BookPathStep, ...]
    state: DataQueryState | None


def _request_id(
    generator_commit: str,
    depth: int,
    actions: Sequence[str],
    purpose: str,
) -> str:
    digest = canonical_sha256(
        {
            "schema": BOOK_PATH_CORPUS_SCHEMA,
            "generator_commit": generator_commit,
            "depth": depth,
            "actions": list(actions),
            "purpose": purpose,
        }
    )
    return f"book-corpus-{depth:02d}-{purpose}-{digest[:20]}"


def _validate_state(
    state: DataQueryState,
    *,
    actions: Sequence[str],
    logical_ply_count: int,
    previous: DataQueryState | None,
) -> None:
    if state.pending_removal or sum(state.pending_removals) != 0:
        raise SanmillBookPathError("book corpus boundary has pending removal")
    if state.action_token_count != len(actions):
        raise SanmillBookPathError("book corpus action count drifted")
    if state.logical_ply_count != logical_ply_count:
        raise SanmillBookPathError("book corpus logical count drifted")
    if state.logical_plies_by_side != _expected_side_counts(logical_ply_count):
        raise SanmillBookPathError("book corpus side counts drifted")
    if previous is not None and state != previous:
        raise SanmillBookPathError(
            "book query state differs from preceding history summary"
        )
    if state.outcome.terminal:
        if state.side_to_move is not None:
            raise SanmillBookPathError("terminal book state retains a side")
    else:
        expected_side = "white" if logical_ply_count % 2 == 0 else "black"
        if state.side_to_move != expected_side:
            raise SanmillBookPathError("book corpus side to move drifted")


def _leaf(
    kind: str,
    node: _FrontierNode,
    state: DataQueryState,
) -> IncompleteBookLeaf:
    return IncompleteBookLeaf(
        kind=kind,
        logical_ply_count=state.logical_ply_count,
        action_tokens=node.actions,
        fen=state.current_fen,
        history_sha256=state.history_sha256,
        outcome_kind=state.outcome.kind,
        winner=state.outcome.winner,
        reason=state.outcome.reason,
    )


def _step(
    depth: int,
    candidate: DataQueryCandidate,
    before: DataQueryState,
    after: DataQueryState,
) -> BookPathStep:
    if candidate.source_group_id is None or candidate.source_rank is None:
        raise SanmillBookPathError("book candidate lacks source rank metadata")
    if candidate.remaining_actions != candidate.full_turn_actions:
        raise SanmillBookPathError(
            "stable book boundary returned a partial logical turn"
        )
    if candidate.logical_ply_delta != 1 or not candidate.turn_prefix_complete:
        raise SanmillBookPathError("book candidate is not one complete logical ply")
    return BookPathStep(
        logical_ply=depth,
        side=before.side_to_move or "",
        candidate_logical_move_id=candidate.logical_move_id,
        candidate_stable_index=candidate.stable_index,
        candidate_source_group_id=candidate.source_group_id,
        candidate_source_rank=candidate.source_rank,
        action_tokens=candidate.full_turn_actions,
        compound_turn=candidate.contains_removal,
        input_fen=before.current_fen,
        input_history_sha256=before.history_sha256,
        output_fen=after.current_fen,
        output_history_sha256=after.history_sha256,
        output_action_token_count=after.action_token_count,
        output_logical_ply_count=after.logical_ply_count,
        output_logical_plies_by_side=after.logical_plies_by_side,
        output_terminal=after.outcome.terminal,
        output_outcome_kind=after.outcome.kind,
        output_winner=after.outcome.winner,
        output_reason=after.outcome.reason,
    )


def enumerate_complete_book_paths(
    session: SanmillDataQuerySession,
    installation: SanmillInstallation,
    *,
    generator_commit: str,
) -> BookPathCorpus:
    """Enumerate every exact book history that reaches eight logical plies."""
    if not _SHA40.fullmatch(generator_commit):
        raise SanmillBookPathError("generator_commit must be a full Git SHA")
    frontier = [_FrontierNode(actions=(), steps=(), state=None)]
    audits: list[BookDepthAudit] = []
    leaves: list[IncompleteBookLeaf] = []
    bound_source_identity: dict[str, Any] | None = None

    for depth in range(PREFIX_LOGICAL_PLIES):
        next_frontier: list[_FrontierNode] = []
        available_count = 0
        miss_count = 0
        terminal_count = 0
        edge_count = 0
        compound_count = 0
        seen_children: set[tuple[str, ...]] = set()

        for node in sorted(frontier, key=lambda item: item.actions):
            response = session.query_book(
                node.actions,
                request_id=_request_id(
                    generator_commit,
                    depth,
                    node.actions,
                    "query",
                ),
                expected_current_fen=(
                    node.state.current_fen if node.state is not None else None
                ),
            )
            if response.state is None:
                raise SanmillBookPathError("book response lacks state")
            _validate_state(
                response.state,
                actions=node.actions,
                logical_ply_count=depth,
                previous=node.state,
            )
            if response.status == "terminal":
                terminal_count += 1
                leaves.append(_leaf("terminal", node, response.state))
                continue
            if response.source is None:
                raise SanmillBookPathError("non-terminal book response lacks source")
            source_identity = portable_source_identity(response)
            if bound_source_identity is None:
                bound_source_identity = source_identity
            elif source_identity != bound_source_identity:
                raise SanmillBookPathError(
                    "book source identity changed during enumeration"
                )
            if response.status == "book_miss":
                miss_count += 1
                leaves.append(_leaf("book_miss", node, response.state))
                continue
            if response.status != "available" or response.candidates is None:
                raise SanmillBookPathError(
                    f"unsupported book enumeration status {response.status!r}"
                )
            available_count += 1
            for candidate in response.candidates:
                edge_count += 1
                compound_count += int(candidate.contains_removal)
                child_actions = node.actions + candidate.full_turn_actions
                if child_actions in seen_children:
                    raise SanmillBookPathError(
                        "duplicate child history during book enumeration"
                    )
                seen_children.add(child_actions)
                summary = session.history_summary(
                    child_actions,
                    request_id=_request_id(
                        generator_commit,
                        depth + 1,
                        child_actions,
                        "summary",
                    ),
                    count_mode="logical",
                )
                if summary.state is None or summary.status not in {
                    "available",
                    "terminal",
                }:
                    raise SanmillBookPathError(
                        "book candidate replay did not return an authoritative state"
                    )
                _validate_state(
                    summary.state,
                    actions=child_actions,
                    logical_ply_count=depth + 1,
                    previous=None,
                )
                next_frontier.append(
                    _FrontierNode(
                        actions=child_actions,
                        steps=node.steps
                        + (_step(depth, candidate, response.state, summary.state),),
                        state=summary.state,
                    )
                )
        audits.append(
            BookDepthAudit(
                input_logical_ply=depth,
                input_prefix_count=len(frontier),
                available_input_count=available_count,
                book_miss_input_count=miss_count,
                terminal_input_count=terminal_count,
                candidate_edge_count=edge_count,
                unique_child_history_count=len(seen_children),
                compound_edge_count=compound_count,
            )
        )
        frontier = sorted(next_frontier, key=lambda item: item.actions)

    if bound_source_identity is None:
        raise SanmillBookPathError("book enumeration never bound a source")
    if any(node.state is None for node in frontier):
        raise SanmillBookPathError("book enumeration retained a state-less path")
    paths = tuple(
        CompleteBookPath(
            action_tokens=node.actions,
            steps=node.steps,
            final_fen=node.state.current_fen,
            final_history_sha256=node.state.history_sha256,
            final_action_token_count=node.state.action_token_count,
            final_terminal=node.state.outcome.terminal,
            final_outcome_kind=node.state.outcome.kind,
            final_winner=node.state.outcome.winner,
            final_reason=node.state.outcome.reason,
        )
        for node in frontier
        if node.state is not None
    )
    leaves.sort(
        key=lambda item: (
            item.logical_ply_count,
            item.kind,
            item.action_tokens,
        )
    )
    return BookPathCorpus(
        generator_commit=generator_commit,
        sanmill=installation.portable_record(),
        source_identity=bound_source_identity,
        depth_audit=tuple(audits),
        incomplete_leaves=tuple(leaves),
        paths=paths,
    )


def freeze_book_path_corpus(path: str | Path, corpus: BookPathCorpus) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"book-path corpus already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"book-path temporary file exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(corpus.to_dict()))
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise FileExistsError(f"book-path corpus appeared during write: {target}")
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def load_book_path_corpus(path: str | Path) -> BookPathCorpus:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SanmillBookPathError("cannot read strict book-path corpus JSON") from exc
    return BookPathCorpus.from_dict(payload)

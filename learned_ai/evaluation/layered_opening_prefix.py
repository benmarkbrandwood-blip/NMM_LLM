"""Source-neutral twelve-logical-ply opening-prefix records.

The v1 paired-prefix sampler remains an eight-ply diagnostic.  This module
defines a separate v2 identity domain and validates complete histories selected
by source-specific Book, HumanDB, or Perfect DB audits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

from game.board import BoardState
from learned_ai.evaluation.oracle_corpus import ring16_canonical_fen
from learned_ai.evaluation.sanmill_data_query import (
    DataQueryState,
    SanmillDataQuerySession,
)
from learned_ai.evaluation.sanmill_uci import (
    SanmillInstallation,
    project_stable_sanmill_fen,
    validate_uci_action_token,
)
from learned_ai.training.run_contract import canonical_sha256


LAYERED_PREFIX_SCHEMA = "nmm.layered-opening-prefix.v2"
PREFIX_LOGICAL_PLIES_V2 = 12
PREFIX_LOGICAL_PLIES_BY_SIDE_V2 = (6, 6)
_STRATA = frozenset({"book", "human_db", "perfect_db"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LayeredPrefixError(ValueError):
    """Raised when a v2 prefix history, source, or identity is invalid."""


def _fields(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise LayeredPrefixError(f"{context} fields: {'; '.join(details)}")


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LayeredPrefixError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise LayeredPrefixError(f"{context} has a non-string field name")
    return dict(value)


def _string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise LayeredPrefixError(f"{context} must be non-empty text")
    return value


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise LayeredPrefixError(
            f"{context} must be an integer of at least {minimum}"
        )
    return value


def _sha256(value: Any, *, context: str) -> str:
    digest = _string(value, context=context)
    if not _SHA256.fullmatch(digest):
        raise LayeredPrefixError(f"{context} must be a lowercase SHA-256")
    return digest


def _two_counts(value: Any, *, context: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise LayeredPrefixError(f"{context} must contain two counts")
    return (
        _integer(value[0], context=f"{context}[0]"),
        _integer(value[1], context=f"{context}[1]"),
    )


def _expected_side_counts(logical_ply_count: int) -> tuple[int, int]:
    return ((logical_ply_count + 1) // 2, logical_ply_count // 2)


def _assert_portable(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LayeredPrefixError(
                    f"{context} has a non-string field name"
                )
            _assert_portable(item, context=f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_portable(item, context=f"{context}[{index}]")
        return
    if isinstance(value, str):
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise LayeredPrefixError(
                f"{context} contains a machine-specific absolute path"
            )
        return
    if value is not None and not isinstance(value, (bool, int, float)):
        raise LayeredPrefixError(f"{context} is not JSON-portable")


def _state_record(state: DataQueryState) -> dict[str, Any]:
    return {
        "fen": state.current_fen,
        "side_to_move": state.side_to_move,
        "phase": state.phase,
        "history_sha256": state.history_sha256,
        "action_token_count": state.action_token_count,
        "logical_ply_count": state.logical_ply_count,
        "logical_plies_by_side": list(state.logical_plies_by_side),
    }


def _validate_boundary(
    state: DataQueryState,
    *,
    logical_ply_count: int,
    action_token_count: int,
) -> None:
    if state.outcome.terminal:
        raise LayeredPrefixError("prefix reached a terminal boundary")
    if state.pending_removal or sum(state.pending_removals) != 0:
        raise LayeredPrefixError("prefix boundary has a pending removal")
    if state.logical_ply_count != logical_ply_count:
        raise LayeredPrefixError("prefix logical-ply count drifted")
    expected_counts = _expected_side_counts(logical_ply_count)
    if state.logical_plies_by_side != expected_counts:
        raise LayeredPrefixError("prefix per-side logical-ply counts drifted")
    if state.action_token_count != action_token_count:
        raise LayeredPrefixError("prefix action-token count drifted")
    if state.snapshot_history_len != action_token_count:
        raise LayeredPrefixError("prefix snapshot-history length drifted")
    expected_side = "white" if logical_ply_count % 2 == 0 else "black"
    if state.side_to_move != expected_side:
        raise LayeredPrefixError("prefix side to move drifted")
    _sha256(state.history_sha256, context="prefix history identity")


def _validate_logical_turn(
    actions: Sequence[str],
    *,
    logical_ply: int,
) -> tuple[str, ...]:
    if (
        not isinstance(actions, Sequence)
        or isinstance(actions, (str, bytes))
        or not actions
        or len(actions) > 2
    ):
        raise LayeredPrefixError(
            f"logical ply {logical_ply} must contain one primary action and "
            "at most one removal"
        )
    tokens = tuple(validate_uci_action_token(token) for token in actions)
    if tokens[0].startswith("x"):
        raise LayeredPrefixError(
            f"logical ply {logical_ply} starts with a removal"
        )
    if len(tokens) == 2 and not tokens[1].startswith("x"):
        raise LayeredPrefixError(
            f"logical ply {logical_ply} has a non-removal second action"
        )
    return tokens


@dataclass(frozen=True)
class LayeredPrefixStepV2:
    """One complete primary-plus-optional-removal logical turn."""

    logical_ply: int
    side: str
    action_tokens: tuple[str, ...]
    input_state: Mapping[str, Any]
    output_state: Mapping[str, Any]
    source_evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not 0 <= self.logical_ply < PREFIX_LOGICAL_PLIES_V2:
            raise LayeredPrefixError("v2 prefix step index is out of range")
        if self.side not in {"white", "black"}:
            raise LayeredPrefixError("v2 prefix step has an unknown side")
        object.__setattr__(
            self,
            "action_tokens",
            _validate_logical_turn(
                self.action_tokens,
                logical_ply=self.logical_ply,
            ),
        )
        object.__setattr__(
            self,
            "input_state",
            _parse_state_record(
                self.input_state,
                context="v2 prefix step input",
            ),
        )
        object.__setattr__(
            self,
            "output_state",
            _parse_state_record(
                self.output_state,
                context="v2 prefix step output",
            ),
        )
        evidence = _mapping(
            self.source_evidence,
            context="v2 prefix step source evidence",
        )
        _assert_portable(evidence, context="v2 prefix step source evidence")
        object.__setattr__(self, "source_evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_ply": self.logical_ply,
            "side": self.side,
            "action_tokens": list(self.action_tokens),
            "input": dict(self.input_state),
            "output": dict(self.output_state),
            "source_evidence": dict(self.source_evidence),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "LayeredPrefixStepV2":
        payload = _mapping(value, context="v2 prefix step")
        _fields(
            payload,
            required={
                "logical_ply",
                "side",
                "action_tokens",
                "input",
                "output",
                "source_evidence",
            },
            context="v2 prefix step",
        )
        actions = payload["action_tokens"]
        if not isinstance(actions, list):
            raise LayeredPrefixError("v2 prefix step actions must be an array")
        logical_ply = _integer(
            payload["logical_ply"],
            context="v2 prefix step logical_ply",
        )
        return cls(
            logical_ply=logical_ply,
            side=_string(payload["side"], context="v2 prefix step side"),
            action_tokens=_validate_logical_turn(
                actions,
                logical_ply=logical_ply,
            ),
            input_state=_parse_state_record(
                payload["input"],
                context="v2 prefix step input",
            ),
            output_state=_parse_state_record(
                payload["output"],
                context="v2 prefix step output",
            ),
            source_evidence=_mapping(
                payload["source_evidence"],
                context="v2 prefix step source evidence",
            ),
        )


def _parse_state_record(value: Any, *, context: str) -> dict[str, Any]:
    payload = _mapping(value, context=context)
    _fields(
        payload,
        required={
            "fen",
            "side_to_move",
            "phase",
            "history_sha256",
            "action_token_count",
            "logical_ply_count",
            "logical_plies_by_side",
        },
        context=context,
    )
    side = payload["side_to_move"]
    if side not in {"white", "black"}:
        raise LayeredPrefixError(f"{context} has an unknown side")
    counts = _two_counts(
        payload["logical_plies_by_side"],
        context=f"{context} logical_plies_by_side",
    )
    logical_count = _integer(
        payload["logical_ply_count"],
        context=f"{context} logical_ply_count",
    )
    if sum(counts) != logical_count:
        raise LayeredPrefixError(f"{context} logical counts disagree")
    return {
        "fen": _string(payload["fen"], context=f"{context} fen"),
        "side_to_move": side,
        "phase": _string(payload["phase"], context=f"{context} phase"),
        "history_sha256": _sha256(
            payload["history_sha256"],
            context=f"{context} history_sha256",
        ),
        "action_token_count": _integer(
            payload["action_token_count"],
            context=f"{context} action_token_count",
        ),
        "logical_ply_count": logical_count,
        "logical_plies_by_side": list(counts),
    }


@dataclass(frozen=True)
class LayeredOpeningPrefixV2:
    """Immutable, source-labelled twelve-logical-ply history."""

    stratum: str
    source_subtype: str
    source_history_id: str
    source_identity: Mapping[str, Any]
    source_evidence: Mapping[str, Any]
    sanmill: Mapping[str, Any]
    action_tokens: tuple[str, ...]
    steps: tuple[LayeredPrefixStepV2, ...]
    final_sanmill_fen: str
    final_nmm_fen: str
    final_ring16_fen: str
    final_history_sha256: str
    prefix_identity: str = ""

    def __post_init__(self) -> None:
        if self.stratum not in _STRATA:
            raise LayeredPrefixError(f"unknown prefix stratum {self.stratum!r}")
        _string(self.source_subtype, context="source subtype")
        _sha256(self.source_history_id, context="source history identity")
        _assert_portable(self.source_identity, context="source identity")
        _assert_portable(self.source_evidence, context="source evidence")
        _assert_portable(self.sanmill, context="Sanmill identity")
        if self.source_identity.get("kind") != self.stratum:
            raise LayeredPrefixError(
                "source identity kind differs from the prefix stratum"
            )
        _sha256(
            self.source_identity.get("identity_sha256"),
            context="source portable identity",
        )
        if len(self.steps) != PREFIX_LOGICAL_PLIES_V2:
            raise LayeredPrefixError("v2 prefix does not have twelve steps")
        if [step.logical_ply for step in self.steps] != list(
            range(PREFIX_LOGICAL_PLIES_V2)
        ):
            raise LayeredPrefixError("v2 prefix step indices are not contiguous")
        expected_sides = [
            "white" if index % 2 == 0 else "black"
            for index in range(PREFIX_LOGICAL_PLIES_V2)
        ]
        if [step.side for step in self.steps] != expected_sides:
            raise LayeredPrefixError("v2 prefix sides do not alternate")
        flattened = tuple(
            token for step in self.steps for token in step.action_tokens
        )
        if flattened != self.action_tokens:
            raise LayeredPrefixError("v2 prefix action history is inconsistent")
        consumed_actions = 0
        for index, step in enumerate(self.steps):
            input_state = step.input_state
            output_state = step.output_state
            if input_state["logical_ply_count"] != index:
                raise LayeredPrefixError("v2 prefix input count is inconsistent")
            if output_state["logical_ply_count"] != index + 1:
                raise LayeredPrefixError("v2 prefix output count is inconsistent")
            if tuple(output_state["logical_plies_by_side"]) != (
                _expected_side_counts(index + 1)
            ):
                raise LayeredPrefixError(
                    "v2 prefix output side counts are inconsistent"
                )
            if input_state["action_token_count"] != consumed_actions:
                raise LayeredPrefixError(
                    "v2 prefix input action count is inconsistent"
                )
            consumed_actions += len(step.action_tokens)
            if output_state["action_token_count"] != consumed_actions:
                raise LayeredPrefixError(
                    "v2 prefix output action count is inconsistent"
                )
            if (
                input_state["side_to_move"] != expected_sides[index]
                or step.side != input_state["side_to_move"]
            ):
                raise LayeredPrefixError(
                    "v2 prefix input side is inconsistent"
                )
            if index > 0 and input_state != self.steps[index - 1].output_state:
                raise LayeredPrefixError("v2 prefix state chain is inconsistent")
        final_output = self.steps[-1].output_state
        if (
            final_output["fen"] != self.final_sanmill_fen
            or final_output["history_sha256"] != self.final_history_sha256
            or final_output["logical_ply_count"] != PREFIX_LOGICAL_PLIES_V2
            or tuple(final_output["logical_plies_by_side"])
            != PREFIX_LOGICAL_PLIES_BY_SIDE_V2
        ):
            raise LayeredPrefixError("v2 prefix final state is inconsistent")
        _string(self.final_nmm_fen, context="final NMM FEN")
        _string(self.final_ring16_fen, context="final ring16 FEN")
        try:
            final_board = BoardState.from_fen_string(self.final_nmm_fen)
        except (TypeError, ValueError) as exc:
            raise LayeredPrefixError("final NMM FEN is invalid") from exc
        if final_board.to_fen_string() != self.final_nmm_fen:
            raise LayeredPrefixError("final NMM FEN does not round-trip")
        if ring16_canonical_fen(self.final_nmm_fen) != self.final_ring16_fen:
            raise LayeredPrefixError("final ring16 FEN is inconsistent")
        _sha256(self.final_history_sha256, context="final history identity")
        expected = canonical_sha256(self._identity_body())
        if self.prefix_identity and self.prefix_identity != expected:
            raise LayeredPrefixError("v2 prefix identity mismatch")
        object.__setattr__(self, "prefix_identity", expected)

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema_version": LAYERED_PREFIX_SCHEMA,
            "logical_ply_count": PREFIX_LOGICAL_PLIES_V2,
            "logical_plies_by_side": list(PREFIX_LOGICAL_PLIES_BY_SIDE_V2),
            "stratum": self.stratum,
            "source_subtype": self.source_subtype,
            "source_history_id": self.source_history_id,
            "source_identity": dict(self.source_identity),
            "source_evidence": dict(self.source_evidence),
            "sanmill": dict(self.sanmill),
            "action_tokens": list(self.action_tokens),
            "steps": [step.to_dict() for step in self.steps],
            "final": {
                "sanmill_fen": self.final_sanmill_fen,
                "nmm_fen": self.final_nmm_fen,
                "ring16_canonical_fen": self.final_ring16_fen,
                "history_sha256": self.final_history_sha256,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_body(),
            "prefix_identity": self.prefix_identity,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "LayeredOpeningPrefixV2":
        payload = _mapping(value, context="v2 layered prefix")
        _fields(
            payload,
            required={
                "schema_version",
                "logical_ply_count",
                "logical_plies_by_side",
                "stratum",
                "source_subtype",
                "source_history_id",
                "source_identity",
                "source_evidence",
                "sanmill",
                "action_tokens",
                "steps",
                "final",
                "prefix_identity",
            },
            context="v2 layered prefix",
        )
        if payload["schema_version"] != LAYERED_PREFIX_SCHEMA:
            raise LayeredPrefixError("unknown layered-prefix schema")
        if (
            payload["logical_ply_count"] != PREFIX_LOGICAL_PLIES_V2
            or _two_counts(
                payload["logical_plies_by_side"],
                context="v2 prefix side counts",
            )
            != PREFIX_LOGICAL_PLIES_BY_SIDE_V2
        ):
            raise LayeredPrefixError("v2 prefix target length is invalid")
        raw_steps = payload["steps"]
        raw_actions = payload["action_tokens"]
        if not isinstance(raw_steps, list) or not isinstance(raw_actions, list):
            raise LayeredPrefixError("v2 prefix steps and actions must be arrays")
        final = _mapping(payload["final"], context="v2 prefix final")
        _fields(
            final,
            required={
                "sanmill_fen",
                "nmm_fen",
                "ring16_canonical_fen",
                "history_sha256",
            },
            context="v2 prefix final",
        )
        return cls(
            stratum=_string(payload["stratum"], context="v2 prefix stratum"),
            source_subtype=_string(
                payload["source_subtype"],
                context="v2 prefix source subtype",
            ),
            source_history_id=_sha256(
                payload["source_history_id"],
                context="v2 prefix source history identity",
            ),
            source_identity=_mapping(
                payload["source_identity"],
                context="v2 prefix source identity",
            ),
            source_evidence=_mapping(
                payload["source_evidence"],
                context="v2 prefix source evidence",
            ),
            sanmill=_mapping(payload["sanmill"], context="v2 prefix Sanmill"),
            action_tokens=tuple(
                validate_uci_action_token(token) for token in raw_actions
            ),
            steps=tuple(
                LayeredPrefixStepV2.from_dict(step) for step in raw_steps
            ),
            final_sanmill_fen=_string(
                final["sanmill_fen"],
                context="v2 prefix final Sanmill FEN",
            ),
            final_nmm_fen=_string(
                final["nmm_fen"],
                context="v2 prefix final NMM FEN",
            ),
            final_ring16_fen=_string(
                final["ring16_canonical_fen"],
                context="v2 prefix final ring16 FEN",
            ),
            final_history_sha256=_sha256(
                final["history_sha256"],
                context="v2 prefix final history identity",
            ),
            prefix_identity=_sha256(
                payload["prefix_identity"],
                context="v2 prefix identity",
            ),
        )


def build_layered_prefix_v2(
    session: SanmillDataQuerySession,
    installation: SanmillInstallation,
    *,
    stratum: str,
    source_subtype: str,
    source_history_id: str,
    source_identity: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    logical_turns: Sequence[Sequence[str]],
    step_evidence: Sequence[Mapping[str, Any]],
) -> LayeredOpeningPrefixV2:
    """Replay and content-identify one source-selected twelve-ply history."""
    if len(logical_turns) != PREFIX_LOGICAL_PLIES_V2:
        raise LayeredPrefixError("v2 history must contain twelve logical turns")
    if len(step_evidence) != PREFIX_LOGICAL_PLIES_V2:
        raise LayeredPrefixError("v2 history must contain twelve step proofs")
    turns = tuple(
        _validate_logical_turn(turn, logical_ply=index)
        for index, turn in enumerate(logical_turns)
    )
    history_id = _sha256(source_history_id, context="source history identity")
    request_prefix = f"prefix-v2-{history_id[:20]}"
    initial = session.history_summary(
        (),
        request_id=f"{request_prefix}-start",
        count_mode="logical",
    )
    if initial.status != "available" or initial.state is None:
        raise LayeredPrefixError("Sanmill did not return an available start state")
    _validate_boundary(initial.state, logical_ply_count=0, action_token_count=0)

    actions: tuple[str, ...] = ()
    before = initial.state
    steps: list[LayeredPrefixStepV2] = []
    for logical_ply, turn in enumerate(turns):
        actions += turn
        response = session.history_summary(
            actions,
            request_id=f"{request_prefix}-{logical_ply:02d}",
            count_mode="logical",
        )
        if response.status != "available" or response.state is None:
            raise LayeredPrefixError(
                f"Sanmill rejected source history at logical ply {logical_ply}"
            )
        after = response.state
        _validate_boundary(
            after,
            logical_ply_count=logical_ply + 1,
            action_token_count=len(actions),
        )
        if after.history_sha256 == before.history_sha256:
            raise LayeredPrefixError("a v2 logical turn did not change history")
        evidence = _mapping(
            step_evidence[logical_ply],
            context=f"v2 step {logical_ply} source evidence",
        )
        _assert_portable(evidence, context=f"v2 step {logical_ply} evidence")
        steps.append(
            LayeredPrefixStepV2(
                logical_ply=logical_ply,
                side=before.side_to_move or "",
                action_tokens=turn,
                input_state=_state_record(before),
                output_state=_state_record(after),
                source_evidence=evidence,
            )
        )
        before = after

    board = project_stable_sanmill_fen(before.current_fen)
    nmm_fen = board.to_fen_string()
    return LayeredOpeningPrefixV2(
        stratum=stratum,
        source_subtype=source_subtype,
        source_history_id=history_id,
        source_identity=dict(source_identity),
        source_evidence=dict(source_evidence),
        sanmill=installation.portable_record(),
        action_tokens=actions,
        steps=tuple(steps),
        final_sanmill_fen=before.current_fen,
        final_nmm_fen=nmm_fen,
        final_ring16_fen=ring16_canonical_fen(nmm_fen),
        final_history_sha256=before.history_sha256,
    )

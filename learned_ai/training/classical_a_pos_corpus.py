"""Frozen offline corpus contract for the classical D9 ``A_pos`` student.

The loader deliberately replays every complete game prefix and compares the
stored full legal action inventory with the rules engine's exact atomic order.
It performs no teacher queries and cannot accept a partially labelled prefix as
the production corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase
from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_RULES_IDENTITY_SHA256,
    SANMILL_BINARY_RELATIVE,
)
from learned_ai.training.sanmill_referee import (
    TRAINING_REFEREE_FORMAT,
    TRAINING_REFEREE_PROFILE,
    TRAINING_REFEREE_SEMANTIC_DIGEST,
    TRAINING_REPETITION_OBSERVATION,
    TRAINING_SANMILL_BINARY_SHA256,
    TRAINING_SANMILL_BINARY_SIZE,
    TRAINING_SANMILL_COMMIT,
    TRAINING_SANMILL_TREE,
)
from learned_ai.training.run_contract import canonical_sha256


CORPUS_SCHEMA = "nmm.classical-a-pos-corpus.v1"
CORPUS_RECORD_SCHEMA = "nmm.classical-a-pos-corpus-record.v1"
STATE_RECORD_SCHEMA = "nmm.classical-a-pos-state-record.v1"
STATE_SPLIT_ARTIFACT_SCHEMA = "nmm.classical-a-pos-state-split-artifact.v1"
SINGLETON_LEDGER_SCHEMA = "nmm.classical-a-pos-singleton-ledger.v1"
TRAINING_PROFILE = "classical-a-pos-distillation-v1"

RESOURCE_IDENTITY_KEYS = frozenset(
    {
        "evolved_weights",
        "fullgame_db",
        "endgame_db",
        "phase_value_place",
        "phase_value_move",
        "phase_value_fly",
        "gap_net",
        "malom_manifest",
        "malom_content",
    }
)
IMPLEMENTATION_IDENTITY_KEYS = frozenset(
    {
        "corpus_builder",
        "scaffolded_encoder",
        "scaffolded_net",
        "game_ai",
        "heuristics",
        "native_extension",
        "positional_safety_gate",
    }
)
ORACLE_IDENTITY_KEYS = frozenset(
    {
        "label_version",
        "manifest_sha256",
        "content_sha256",
    }
)


class CorpusContractError(RuntimeError):
    """A frozen corpus or its provenance failed a fail-closed check."""


@dataclass(frozen=True)
class CorpusLayout:
    """Expected split/phase/colour cardinalities."""

    counts: Mapping[str, Mapping[str, Mapping[str, int]]]

    def to_dict(self) -> dict[str, dict[str, dict[str, int]]]:
        expected_splits = {"train", "dev"}
        expected_strata = {"placement", "movement", "flying"}
        expected_colours = {"W", "B"}
        if set(self.counts) != expected_splits:
            raise CorpusContractError("corpus layout split keys differ")
        copied: dict[str, dict[str, dict[str, int]]] = {}
        for split in ("train", "dev"):
            strata = self.counts[split]
            if set(strata) != expected_strata:
                raise CorpusContractError("corpus layout stratum keys differ")
            copied[split] = {}
            for stratum in ("placement", "movement", "flying"):
                colours = strata[stratum]
                if set(colours) != expected_colours:
                    raise CorpusContractError("corpus layout colour keys differ")
                values: dict[str, int] = {}
                for colour in ("W", "B"):
                    value = colours[colour]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise CorpusContractError(
                            "corpus layout counts must be non-negative integers"
                        )
                    values[colour] = value
                if values["W"] != values["B"]:
                    raise CorpusContractError(
                        "each corpus stratum must be colour balanced"
                    )
                copied[split][stratum] = values
        return copied

    @property
    def total(self) -> int:
        return sum(
            count
            for strata in self.to_dict().values()
            for colours in strata.values()
            for count in colours.values()
        )


FROZEN_CORPUS_LAYOUT = CorpusLayout(
    counts={
        "train": {
            "placement": {"W": 1_792, "B": 1_792},
            "movement": {"W": 3_584, "B": 3_584},
            "flying": {"W": 1_792, "B": 1_792},
        },
        "dev": {
            "placement": {"W": 256, "B": 256},
            "movement": {"W": 512, "B": 512},
            "flying": {"W": 256, "B": 256},
        },
    }
)


@dataclass(frozen=True)
class FrozenCorpusExample:
    state_record_identity: str
    example_id: str
    game_id: str
    game_index: int
    split: str
    stratum: str
    candidate_color: str
    logical_ply: int
    history_moves: tuple[dict[str, Any], ...]
    history_sha256: str
    sanmill_history_sha256: str
    board: BoardState
    legal_actions: tuple[dict[str, Any], ...]
    a_pos_mask: tuple[bool, ...]
    teacher_action: dict[str, Any]
    teacher_index: int
    teacher_evidence: Mapping[str, Any]
    record_sha256: str


@dataclass(frozen=True)
class FrozenCorpus:
    manifest: Mapping[str, Any]
    examples: tuple[FrozenCorpusExample, ...]
    corpus_identity: str
    split_identity: str


_STATE_RECORD_FIELDS = {
    "schema_version",
    "state_record_identity",
    "example_id",
    "game_id",
    "game_index",
    "split",
    "stratum",
    "candidate_color",
    "logical_ply",
    "history_moves",
    "history_sha256",
    "sanmill_history_sha256",
    "board_fen",
    "legal_actions",
    "a_pos_mask",
    "a_pos_verification",
}


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object_pairs)
    except CorpusContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusContractError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise CorpusContractError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CorpusContractError(f"{field} must be a lowercase SHA-256")
    return value


def _git_identity(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CorpusContractError(f"{field} must be a 40-character Git identity")
    return value


def _action(move: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(move, Mapping) or set(move) != {"from", "to", "capture"}:
        raise CorpusContractError(f"{field} must be one atomic from/to/capture action")
    source = move["from"]
    target = move["to"]
    capture = move["capture"]
    if source is not None and not isinstance(source, str):
        raise CorpusContractError(f"{field}.from is invalid")
    if not isinstance(target, str) or not target:
        raise CorpusContractError(f"{field}.to is invalid")
    if capture is not None and not isinstance(capture, str):
        raise CorpusContractError(f"{field}.capture is invalid")
    return {"from": source, "to": target, "capture": capture}


def action_key(move: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return move.get("from"), move.get("to"), move.get("capture")


def seal_singleton_ledger(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": SINGLETON_LEDGER_SCHEMA,
        "count": len(entries),
        "entries": [dict(entry) for entry in entries],
    }
    payload["identity"] = canonical_sha256(payload)
    return payload


def _validate_string_identities(
    value: Any,
    *,
    keys: frozenset[str],
    field: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise CorpusContractError(f"{field} identity keys differ")
    return {key: _sha256(value[key], field=f"{field}.{key}") for key in sorted(keys)}


def _validate_loader_oracle_identity(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != ORACLE_IDENTITY_KEYS:
        raise CorpusContractError("loader A_pos oracle identity keys differ")
    if value.get("label_version") != CURRENT_MALOM_LABEL_VERSION:
        raise CorpusContractError(
            "loader A_pos oracle label version must be sector-corrected-v1"
        )
    return {
        "label_version": CURRENT_MALOM_LABEL_VERSION,
        "manifest_sha256": _sha256(
            value.get("manifest_sha256"),
            field="loader A_pos oracle manifest_sha256",
        ),
        "content_sha256": _sha256(
            value.get("content_sha256"),
            field="loader A_pos oracle content_sha256",
        ),
    }


def _validate_source(source: Any) -> dict[str, Any]:
    if not isinstance(source, Mapping) or set(source) != {
        "git_commit",
        "git_tree",
        "implementation_sha256",
    }:
        raise CorpusContractError("corpus source keys differ")
    return {
        "git_commit": _git_identity(source["git_commit"], field="git_commit"),
        "git_tree": _git_identity(source["git_tree"], field="git_tree"),
        "implementation_sha256": _validate_string_identities(
            source["implementation_sha256"],
            keys=IMPLEMENTATION_IDENTITY_KEYS,
            field="implementation_sha256",
        ),
    }


def a_pos_verifier_identity(
    source: Mapping[str, Any], resources: Mapping[str, str]
) -> str:
    """Bind the build-time inventory verifier to code and Malom bytes."""
    return canonical_sha256(
        {
            "contract": "real-PositionalSafetyFilter-inventory-v1",
            "source": _validate_source(source),
            "resources": _validate_string_identities(
                resources,
                keys=RESOURCE_IDENTITY_KEYS,
                field="a_pos_verifier.resources",
            ),
        }
    )


class CorpusRecordSealer:
    """Seal records only after an injected real inventory verifier succeeds.

    Production corpus generation supplies a callback backed by
    ``PositionalSafetyFilter.inspect_moves``. The verifier identity is derived
    from the tracked builder/gate code plus immutable Malom resource hashes.
    This build-time chain is provenance only: the production loader separately
    re-queries the real positional-safety oracle for every frozen state.
    """

    _UNSEALED_STATE_FIELDS = {
        "example_id",
        "game_id",
        "game_index",
        "split",
        "stratum",
        "candidate_color",
        "logical_ply",
        "history_moves",
        "history_sha256",
        "sanmill_history_sha256",
        "board_fen",
    }

    def __init__(
        self,
        *,
        source: Mapping[str, Any],
        resources: Mapping[str, str],
        inventory_verifier: Callable[
            [BoardState, Sequence[Mapping[str, Any]]], Sequence[bool]
        ],
    ) -> None:
        if not callable(inventory_verifier):
            raise CorpusContractError("A_pos inventory verifier is not callable")
        self.verifier_identity = a_pos_verifier_identity(source, resources)
        self._inventory_verifier = inventory_verifier
        self._previous_verification_sha256 = "0" * 64

    @property
    def chain_head(self) -> str:
        return self._previous_verification_sha256

    def freeze_state(self, record: Mapping[str, Any]) -> dict[str, Any]:
        copied = dict(record)
        if set(copied) != self._UNSEALED_STATE_FIELDS:
            raise CorpusContractError("unsealed state record keys differ")
        history_raw = copied["history_moves"]
        if not isinstance(history_raw, list):
            raise CorpusContractError("history_moves must be a list")
        history = [
            _action(move, field=f"history_moves[{index}]")
            for index, move in enumerate(history_raw)
        ]
        board = BoardState.new_game()
        for index, move in enumerate(history):
            if move not in get_all_legal_moves(board):
                raise CorpusContractError(
                    f"cannot seal illegal full-history action at ply {index}"
                )
            board = board.apply_move(move)
        if copied["board_fen"] != board.to_fen_string():
            raise CorpusContractError("cannot seal a mismatched board FEN")
        legal = [dict(move) for move in get_all_legal_moves(board)]
        try:
            raw_mask = tuple(self._inventory_verifier(board, legal))
        except Exception as exc:
            raise CorpusContractError(
                "real A_pos inventory verification failed"
            ) from exc
        if (
            len(raw_mask) != len(legal)
            or any(not isinstance(value, bool) for value in raw_mask)
            or sum(raw_mask) <= 1
        ):
            raise CorpusContractError(
                "verified inventory is not one informative full legal A_pos mask"
            )
        inventory_payload = {
            "board_fen": board.to_fen_string(),
            "legal_actions": legal,
            "a_pos_mask": list(raw_mask),
        }
        verification = {
            "verifier_identity": self.verifier_identity,
            "inventory_sha256": canonical_sha256(inventory_payload),
            "previous_verification_sha256": self._previous_verification_sha256,
        }
        verification["verification_sha256"] = canonical_sha256(verification)
        copied.update(
            {
                "schema_version": STATE_RECORD_SCHEMA,
                "legal_actions": legal,
                "a_pos_mask": list(raw_mask),
                "a_pos_verification": verification,
            }
        )
        copied["state_record_identity"] = canonical_sha256(copied)
        self._previous_verification_sha256 = verification["verification_sha256"]
        return copied

    @staticmethod
    def seal_teacher_label(
        state_record: Mapping[str, Any],
        *,
        teacher_action: Mapping[str, Any],
        teacher_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = dict(state_record)
        observed_state_identity = state.pop("state_record_identity", None)
        if _sha256(
            observed_state_identity, field="state_record_identity"
        ) != canonical_sha256(state):
            raise CorpusContractError("cannot label a changed frozen state record")
        teacher = _action(teacher_action, field="teacher_action")
        legal = state_record.get("legal_actions")
        mask = state_record.get("a_pos_mask")
        if not isinstance(legal, list) or not isinstance(mask, list):
            raise CorpusContractError("frozen state lacks legal/A_pos inventory")
        matching = [index for index, move in enumerate(legal) if move == teacher]
        if len(matching) != 1 or not mask[matching[0]]:
            raise CorpusContractError("teacher action is not inside frozen A_pos")
        evidence = dict(teacher_evidence)
        if (
            evidence.get("gate_status") != "applied"
            or evidence.get("gate_selection_error") is not None
            or evidence.get("selected_action") != teacher
        ):
            raise CorpusContractError(
                "teacher evidence is not an applied post-gate selection"
            )
        record = {
            **{
                key: value
                for key, value in state_record.items()
                if key not in {"schema_version", "state_record_identity"}
            },
            "schema_version": CORPUS_RECORD_SCHEMA,
            "state_record_identity": observed_state_identity,
            "teacher_action": teacher,
            "teacher_index": matching[0],
            "teacher_evidence": evidence,
        }
        record["record_sha256"] = canonical_sha256(record)
        return record


def positional_safety_inventory_verifier(
    safety_filter: Any,
) -> Callable[[BoardState, Sequence[Mapping[str, Any]]], tuple[bool, ...]]:
    """Adapt the real ``PositionalSafetyFilter`` inventory to the sealer."""

    if not callable(getattr(safety_filter, "inspect_moves", None)):
        raise CorpusContractError("positional safety filter is unavailable")

    def verify(
        board: BoardState, legal: Sequence[Mapping[str, Any]]
    ) -> tuple[bool, ...]:
        inventory = safety_filter.inspect_moves(board, legal)
        observed = tuple(dict(move) for move in inventory.legal_moves)
        expected = tuple(dict(move) for move in legal)
        if observed != expected:
            raise CorpusContractError(
                "PositionalSafetyFilter changed the full legal action order"
            )
        safe = set(inventory.safe_indices)
        return tuple(index in safe for index in range(len(expected)))

    return verify


def _file_record(path: Path, *, count: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise CorpusContractError(f"required corpus file is missing: {path.name}")
    record: dict[str, Any] = {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if count is not None:
        record["count"] = count
    return record


def _validate_a_pos_verification(
    record: Mapping[str, Any],
    *,
    verifier_identity: str,
    previous_sha256: str,
) -> str:
    verification = record.get("a_pos_verification")
    if not isinstance(verification, Mapping) or set(verification) != {
        "verifier_identity",
        "inventory_sha256",
        "previous_verification_sha256",
        "verification_sha256",
    }:
        raise CorpusContractError("A_pos verification keys differ")
    if verification["verifier_identity"] != verifier_identity:
        raise CorpusContractError("A_pos verification identity differs")
    if verification["previous_verification_sha256"] != previous_sha256:
        raise CorpusContractError("A_pos verification hash chain differs")
    payload = dict(verification)
    observed = payload.pop("verification_sha256")
    if _sha256(observed, field="A_pos verification SHA-256") != canonical_sha256(
        payload
    ):
        raise CorpusContractError("A_pos verification SHA-256 differs")
    inventory = {
        "board_fen": record.get("board_fen"),
        "legal_actions": record.get("legal_actions"),
        "a_pos_mask": record.get("a_pos_mask"),
    }
    if verification["inventory_sha256"] != canonical_sha256(inventory):
        raise CorpusContractError("A_pos verification inventory differs")
    return observed


def _inspect_state_artifact(
    state_split_path: Path, *, verifier_identity: str
) -> tuple[int, str, tuple[str, ...], tuple[dict[str, Any], ...]]:
    previous = "0" * 64
    count = 0
    identities: list[str] = []
    records: list[dict[str, Any]] = []
    with state_split_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise CorpusContractError("state artifact contains a blank line")
            record = _strict_json_bytes(raw, label=f"state line {line_number}")
            if set(record) != _STATE_RECORD_FIELDS:
                raise CorpusContractError(
                    "frozen state artifact record keys differ or contain teacher data"
                )
            state_payload = dict(record)
            identity = state_payload.pop("state_record_identity", None)
            if _sha256(identity, field="state_record_identity") != canonical_sha256(
                state_payload
            ):
                raise CorpusContractError("frozen state record identity differs")
            if record.get("schema_version") != STATE_RECORD_SCHEMA:
                raise CorpusContractError("frozen state record schema differs")
            previous = _validate_a_pos_verification(
                record,
                verifier_identity=verifier_identity,
                previous_sha256=previous,
            )
            identities.append(identity)
            records.append(record)
            count += 1
    return count, previous, tuple(identities), tuple(records)


def _inspect_teacher_ledger(
    examples_path: Path,
) -> tuple[int, int, int, int, tuple[str, ...]]:
    count = 0
    main_nodes = 0
    rerank_nodes = 0
    positive_labels = 0
    state_identities: list[str] = []
    with examples_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise CorpusContractError("examples JSONL contains a blank line")
            record = _strict_json_bytes(raw, label=f"example line {line_number}")
            if set(record) != _RECORD_FIELDS:
                raise CorpusContractError("corpus record keys differ")
            evidence = record.get("teacher_evidence")
            if not isinstance(evidence, Mapping):
                raise CorpusContractError("teacher evidence is missing")
            main = evidence.get("main_search_nodes")
            rerank = evidence.get("restricted_rerank_nodes")
            if (
                isinstance(main, bool)
                or not isinstance(main, int)
                or not 0 <= main <= 13_887_000
                or isinstance(rerank, bool)
                or not isinstance(rerank, int)
                or rerank < 0
            ):
                raise CorpusContractError("teacher node evidence is invalid")
            main_nodes += main
            rerank_nodes += rerank
            positive_labels += int(main + rerank > 0)
            state_identities.append(
                _sha256(
                    record.get("state_record_identity"),
                    field="state_record_identity",
                )
            )
            count += 1
    return count, main_nodes, rerank_nodes, positive_labels, tuple(state_identities)


def _generator_contract(initial_policy_state_sha256: str) -> dict[str, Any]:
    return {
        "policy": "scratch-ScaffoldedPolicyNet-base62",
        "policy_hidden": [128, 64],
        "value_hidden": [],
        "dropout": 0.0,
        "sampling": "A_pos-multinomial",
        "temperature": 1.0,
        "cpu_generator_seed": 2026083090,
        "model_init_seed": 2026083090,
        "initial_policy_state_sha256": initial_policy_state_sha256,
        "initial_state": "standard",
        "candidate_colours": "alternate-W-B-by-game",
        "max_games": 1_024,
        "informative_only": True,
    }


def _referee_contract(runtime_identity: str) -> dict[str, Any]:
    return {
        "contract_id": "nmm.sanmill-training-runtime.v1",
        "sanmill_commit": TRAINING_SANMILL_COMMIT,
        "sanmill_tree": TRAINING_SANMILL_TREE,
        "binary_relative_path": SANMILL_BINARY_RELATIVE.as_posix(),
        "binary_sha256": TRAINING_SANMILL_BINARY_SHA256,
        "binary_size": TRAINING_SANMILL_BINARY_SIZE,
        "runtime_identity": _sha256(
            runtime_identity,
            field="referee.runtime_identity",
        ),
        "rules_identity_sha256": EXPECTED_RULES_IDENTITY_SHA256,
        "node_budget": 100_000,
        "threads": 1,
        "seed": 42,
        "book": False,
        "strict_complete_history": True,
        "strict_referee": {
            "format": TRAINING_REFEREE_FORMAT,
            "profile": TRAINING_REFEREE_PROFILE,
            "repetition_observation": TRAINING_REPETITION_OBSERVATION,
            "origin_counted": True,
            "semantic_digest": TRAINING_REFEREE_SEMANTIC_DIGEST,
        },
    }


def _selection_contract() -> dict[str, Any]:
    return {
        "informative_a_pos_minimum": 2,
        "singleton_policy": "encounter-ledger-only",
        "full_history_states": True,
        "atomic_action_identity": ["from", "to", "capture"],
        "stratum_selection_order": (
            "canonical-sha256(selection_seed,history_sha256)-ascending-v1"
        ),
        "selection_seed": 2026083091,
        "deduplicate_history_occurrences_by": (
            "canonical-sha256(game_id,logical_ply)-minimum"
        ),
        "selection_cells": 12,
        "teacher_results_visible_to_selection": False,
    }


def _split_contract(layout: CorpusLayout) -> dict[str, Any]:
    return {
        "strategy": "whole-game",
        "algorithm": {
            "game_index_origin": 0,
            "block_size_games": 16,
            "candidate_colour_by_parity": {"even": "W", "odd": "B"},
            "dev_remainders": [0, 1],
            "train_remainders": list(range(2, 16)),
            "stop_only_after_complete_block": True,
            "maximum_games": 1_024,
        },
        "counts": layout.to_dict(),
    }


def state_split_artifact_identity(
    *,
    state_record_identities: Sequence[str],
    split_identity: str,
    verifier_identity: str,
) -> str:
    """Return the ordered frozen state/split artifact identity.

    This is a pure identity helper.  It does not inspect, seal, or authorize an
    artifact, and the order of ``state_record_identities`` is semantic.
    """
    if not isinstance(state_record_identities, Sequence) or isinstance(
        state_record_identities,
        (str, bytes, bytearray),
    ):
        raise CorpusContractError("state record identities must be an ordered array")
    checked_state_identities = [
        _sha256(identity, field=f"state_record_identities[{index}]")
        for index, identity in enumerate(state_record_identities)
    ]
    checked_split_identity = _sha256(split_identity, field="split_identity")
    checked_verifier_identity = _sha256(
        verifier_identity,
        field="a_pos_verifier_identity",
    )
    return canonical_sha256(
        {
            "schema_version": STATE_SPLIT_ARTIFACT_SCHEMA,
            "state_record_schema": STATE_RECORD_SCHEMA,
            "state_record_identities": checked_state_identities,
            "split_identity": checked_split_identity,
            "a_pos_verifier_identity": checked_verifier_identity,
        }
    )


def _build_frozen_corpus_manifest(
    *,
    corpus_id: str,
    source: Mapping[str, Any],
    resources_before: Mapping[str, str],
    resources_after: Mapping[str, str],
    state_split_path: Path,
    examples_path: Path,
    singleton_ledger_path: Path,
    collection_games: int,
    teacher_active_seconds: float,
    initial_policy_state_sha256: str,
    sanmill_runtime_identity: str,
    layout: CorpusLayout,
) -> dict[str, Any]:
    """Build a sealed manifest; production callers use the frozen layout."""
    if not isinstance(corpus_id, str) or not corpus_id:
        raise CorpusContractError("corpus_id must be non-empty")
    checked_source = _validate_source(source)
    before = _validate_string_identities(
        resources_before,
        keys=RESOURCE_IDENTITY_KEYS,
        field="resources_before",
    )
    after = _validate_string_identities(
        resources_after,
        keys=RESOURCE_IDENTITY_KEYS,
        field="resources_after",
    )
    if before != after:
        raise CorpusContractError("teacher resources changed during labeling")
    verifier_identity = a_pos_verifier_identity(checked_source, before)
    (
        state_count,
        verification_chain_head,
        state_record_identities,
        _state_records,
    ) = _inspect_state_artifact(
        state_split_path,
        verifier_identity=verifier_identity,
    )
    (
        labelled_count,
        teacher_main_search_nodes,
        teacher_rerank_nodes,
        teacher_positive_search_labels,
        labelled_state_identities,
    ) = _inspect_teacher_ledger(
        examples_path,
    )
    if state_count != layout.total or labelled_count != layout.total:
        raise CorpusContractError("verified A_pos record count differs")
    if state_record_identities != labelled_state_identities:
        raise CorpusContractError(
            "teacher ledger does not bind the frozen state/split artifact in order"
        )
    if (
        isinstance(collection_games, bool)
        or not isinstance(collection_games, int)
        or not 0 < collection_games <= 1_024
        or collection_games % 16 != 0
    ):
        raise CorpusContractError(
            "collection must stop after a complete 16-game block within 1,024 games"
        )
    if (
        not isinstance(teacher_active_seconds, (int, float))
        or isinstance(teacher_active_seconds, bool)
        or not math.isfinite(float(teacher_active_seconds))
        or not 0.0 <= float(teacher_active_seconds) <= 4 * 3600
    ):
        raise CorpusContractError("teacher active-time bound is invalid")
    teacher_aggregate_nodes = teacher_main_search_nodes + teacher_rerank_nodes
    if not 0 <= teacher_aggregate_nodes <= 4_000_000_000:
        raise CorpusContractError("teacher aggregate-node bound is invalid")
    if not 0 <= teacher_positive_search_labels <= 1_280:
        raise CorpusContractError("positive-search label bound is invalid")
    initial_policy_hash = _sha256(
        initial_policy_state_sha256,
        field="generator.initial_policy_state_sha256",
    )

    split_contract = _split_contract(layout)
    split_identity = canonical_sha256(split_contract)
    state_identity = state_split_artifact_identity(
        state_record_identities=state_record_identities,
        split_identity=split_identity,
        verifier_identity=verifier_identity,
    )
    manifest: dict[str, Any] = {
        "schema_version": CORPUS_SCHEMA,
        "corpus_id": corpus_id,
        "profile": TRAINING_PROFILE,
        "status": "complete",
        "source": checked_source,
        "generator": _generator_contract(initial_policy_hash),
        "referee": _referee_contract(sanmill_runtime_identity),
        "teacher": {
            "route": "current-head-exact-D9-post-ProductPositionalSafetyGate",
            "identity_claim": "fresh-instance-current-D9-positional-teacher",
            "product_tt_trajectory_equivalent": False,
            "fresh_instance_per_label": True,
            "difficulty": 9,
            "depth": 14,
            "threads": 1,
            "node_cap_per_label": 13_887_000,
            "label": "one-hot-atomic-action-inside-A_pos",
            "online_queries_during_training": False,
            "active_seconds": float(teacher_active_seconds),
            "active_seconds_limit": 4 * 3600,
            "main_search_nodes": teacher_main_search_nodes,
            "restricted_rerank_nodes": teacher_rerank_nodes,
            "aggregate_nodes": teacher_aggregate_nodes,
            "aggregate_nodes_limit": 4_000_000_000,
            "positive_search_labels": teacher_positive_search_labels,
            "positive_search_labels_limit": 1_280,
            "completed_labels": layout.total,
        },
        "selection": _selection_contract(),
        "a_pos_verifier": {
            "boundary": ("build-time-chain-plus-load-time-real-PositionalSafetyFilter"),
            "identity": verifier_identity,
            "record_count": state_count,
            "verification_chain_head": verification_chain_head,
            "loader_requeries_malom": True,
        },
        "resources_before": before,
        "resources_after": after,
        "state_split_artifact": {
            "schema_version": STATE_SPLIT_ARTIFACT_SCHEMA,
            "file": _file_record(state_split_path, count=layout.total),
            "identity": state_identity,
            "teacher_fields_present": False,
            "controller_freeze_event_required_before_teacher": True,
        },
        "examples": _file_record(examples_path, count=layout.total),
        "singleton_ledger": _file_record(singleton_ledger_path),
        "collection_games": collection_games,
        "split": split_contract,
        "split_identity": split_identity,
        "heldout": {"consumed": False, "sources": []},
    }
    manifest["corpus_identity"] = canonical_sha256(manifest)
    return manifest


def build_frozen_corpus_manifest(**kwargs: Any) -> dict[str, Any]:
    """Build the production manifest with the non-overridable 16,384 layout."""
    return _build_frozen_corpus_manifest(**kwargs, layout=FROZEN_CORPUS_LAYOUT)


def _validate_file_ref(
    root: Path,
    record: Any,
    *,
    field: str,
    require_count: bool,
) -> Path:
    expected_keys = {"path", "size", "sha256"} | ({"count"} if require_count else set())
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise CorpusContractError(f"{field} file record keys differ")
    relative = record["path"]
    if not isinstance(relative, str) or not relative or Path(relative).name != relative:
        raise CorpusContractError(f"{field} path must be one sibling file")
    path = root / relative
    if not path.is_file():
        raise CorpusContractError(f"{field} file is missing")
    if path.stat().st_size != record["size"]:
        raise CorpusContractError(f"{field} size differs")
    if _sha256_file(path) != _sha256(record["sha256"], field=f"{field}.sha256"):
        raise CorpusContractError(f"{field} SHA-256 differs")
    return path


def _validate_singleton_ledger(path: Path, expected: Mapping[str, Any]) -> None:
    ledger = _strict_json_bytes(path.read_bytes(), label="singleton ledger")
    if set(ledger) != {"schema_version", "count", "entries", "identity"}:
        raise CorpusContractError("singleton ledger keys differ")
    identity = ledger.pop("identity")
    if _sha256(identity, field="singleton ledger identity") != canonical_sha256(ledger):
        raise CorpusContractError("singleton ledger identity differs")
    if ledger["schema_version"] != SINGLETON_LEDGER_SCHEMA:
        raise CorpusContractError("singleton ledger schema differs")
    entries = ledger["entries"]
    if not isinstance(entries, list) or ledger["count"] != len(entries):
        raise CorpusContractError("singleton ledger count differs")
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("a_pos_count") != 1:
            raise CorpusContractError(
                "singleton ledger may contain only |A_pos|=1 encounters"
            )
        _sha256(entry.get("history_sha256"), field="singleton history")
    if expected.get("size") != path.stat().st_size:
        raise CorpusContractError("singleton ledger size differs")


_RECORD_FIELDS = {
    "schema_version",
    "state_record_identity",
    "example_id",
    "game_id",
    "game_index",
    "split",
    "stratum",
    "candidate_color",
    "logical_ply",
    "history_moves",
    "history_sha256",
    "sanmill_history_sha256",
    "board_fen",
    "legal_actions",
    "a_pos_mask",
    "teacher_action",
    "teacher_index",
    "teacher_evidence",
    "a_pos_verification",
    "record_sha256",
}
_TEACHER_FIELDS = {
    "difficulty",
    "depth",
    "threads",
    "node_cap",
    "main_search_nodes",
    "restricted_rerank_nodes",
    "positive_search",
    "teacher_instance_id",
    "gate_status",
    "gate_source",
    "gate_selection_rule",
    "gate_selection_error",
    "original_action",
    "selected_action",
}


def _validate_record(record: Mapping[str, Any]) -> FrozenCorpusExample:
    if set(record) != _RECORD_FIELDS:
        raise CorpusContractError("corpus record keys differ")
    sealed = dict(record)
    observed_hash = sealed.pop("record_sha256")
    if _sha256(observed_hash, field="record_sha256") != canonical_sha256(sealed):
        raise CorpusContractError("corpus record SHA-256 differs")
    if record["schema_version"] != CORPUS_RECORD_SCHEMA:
        raise CorpusContractError("corpus record schema differs")
    state_record_identity = _sha256(
        record["state_record_identity"],
        field="state_record_identity",
    )
    example_id = _sha256(record["example_id"], field="example_id")
    game_id = record["game_id"]
    if not isinstance(game_id, str) or not game_id:
        raise CorpusContractError("game_id must be non-empty")
    game_index = record["game_index"]
    if (
        isinstance(game_index, bool)
        or not isinstance(game_index, int)
        or game_index < 0
    ):
        raise CorpusContractError("game_index must be a non-negative integer")
    split = record["split"]
    stratum = record["stratum"]
    colour = record["candidate_color"]
    if split not in {"train", "dev"}:
        raise CorpusContractError("record split is invalid")
    if stratum not in {"placement", "movement", "flying"}:
        raise CorpusContractError("record stratum is invalid")
    if colour not in {"W", "B"}:
        raise CorpusContractError("candidate colour is invalid")
    history_raw = record["history_moves"]
    if not isinstance(history_raw, list):
        raise CorpusContractError("history_moves must be a list")
    history = tuple(
        _action(move, field=f"history_moves[{index}]")
        for index, move in enumerate(history_raw)
    )
    if record["logical_ply"] != len(history):
        raise CorpusContractError("logical ply differs from full history")
    if _sha256(record["history_sha256"], field="history_sha256") != canonical_sha256(
        list(history)
    ):
        raise CorpusContractError("full-history SHA-256 differs")
    sanmill_history = _sha256(
        record["sanmill_history_sha256"], field="sanmill_history_sha256"
    )

    board = BoardState.new_game()
    for index, move in enumerate(history):
        legal = get_all_legal_moves(board)
        if move not in legal:
            raise CorpusContractError(
                f"full history contains an illegal atomic action at ply {index}"
            )
        board = board.apply_move(move)
    if record["board_fen"] != board.to_fen_string():
        raise CorpusContractError("board FEN differs from replayed full history")
    if board.turn != colour:
        raise CorpusContractError("candidate colour differs from side to move")
    expected_phase = {
        "place": "placement",
        "move": "movement",
        "fly": "flying",
    }[get_game_phase(board, colour)]
    if stratum != expected_phase:
        raise CorpusContractError("record stratum differs from replayed phase")
    expected_example_id = canonical_sha256(
        {
            "history_sha256": record["history_sha256"],
            "board_fen": record["board_fen"],
        }
    )
    if example_id != expected_example_id:
        raise CorpusContractError("example identity differs from full-history state")

    raw_legal = record["legal_actions"]
    if not isinstance(raw_legal, list):
        raise CorpusContractError("legal_actions must be a list")
    actions = tuple(
        _action(move, field=f"legal_actions[{index}]")
        for index, move in enumerate(raw_legal)
    )
    expected_actions = tuple(dict(move) for move in get_all_legal_moves(board))
    if actions != expected_actions:
        raise CorpusContractError(
            "stored legal action order differs from the encoder/rules order"
        )
    if len({action_key(move) for move in actions}) != len(actions):
        raise CorpusContractError("legal action inventory contains duplicates")
    mask_raw = record["a_pos_mask"]
    if (
        not isinstance(mask_raw, list)
        or len(mask_raw) != len(actions)
        or any(not isinstance(value, bool) for value in mask_raw)
    ):
        raise CorpusContractError("A_pos mask differs from legal action order")
    mask = tuple(mask_raw)
    if sum(mask) <= 1:
        raise CorpusContractError("training corpus may contain only |A_pos|>1 states")
    teacher_index = record["teacher_index"]
    if (
        isinstance(teacher_index, bool)
        or not isinstance(teacher_index, int)
        or not 0 <= teacher_index < len(actions)
        or not mask[teacher_index]
    ):
        raise CorpusContractError("teacher index is not inside A_pos")
    teacher_action = _action(record["teacher_action"], field="teacher_action")
    if teacher_action != actions[teacher_index]:
        raise CorpusContractError(
            "teacher action/index differs from legal action order"
        )

    evidence = record["teacher_evidence"]
    if not isinstance(evidence, Mapping) or set(evidence) != _TEACHER_FIELDS:
        raise CorpusContractError("teacher evidence keys differ")
    exact = {
        "difficulty": 9,
        "depth": 14,
        "threads": 1,
        "node_cap": 13_887_000,
        "gate_status": "applied",
        "gate_source": "classical-coordinator",
        "gate_selection_error": None,
    }
    if any(evidence.get(key) != value for key, value in exact.items()):
        raise CorpusContractError("teacher D9 post-gate contract differs")
    if evidence.get("gate_selection_rule") not in {
        "original-already-in-A_pos",
        "restricted-root-research",
    }:
        raise CorpusContractError("teacher gate used a fallback selection rule")
    if _action(evidence["selected_action"], field="selected_action") != teacher_action:
        raise CorpusContractError("teacher selected action differs")
    original = _action(evidence["original_action"], field="original_action")
    if action_key(original) not in {action_key(move) for move in actions}:
        raise CorpusContractError("teacher original action is not legal")
    main_nodes = evidence["main_search_nodes"]
    if (
        isinstance(main_nodes, bool)
        or not isinstance(main_nodes, int)
        or not 0 <= main_nodes <= 13_887_000
    ):
        raise CorpusContractError("teacher main-search node count is invalid")
    rerank_nodes = evidence["restricted_rerank_nodes"]
    if (
        isinstance(rerank_nodes, bool)
        or not isinstance(rerank_nodes, int)
        or rerank_nodes < 0
    ):
        raise CorpusContractError("teacher rerank node count is invalid")
    positive = evidence["positive_search"]
    if not isinstance(positive, bool) or positive != (main_nodes + rerank_nodes > 0):
        raise CorpusContractError("teacher positive-search evidence differs")
    _sha256(evidence["teacher_instance_id"], field="teacher_instance_id")

    return FrozenCorpusExample(
        state_record_identity=state_record_identity,
        example_id=example_id,
        game_id=game_id,
        game_index=game_index,
        split=split,
        stratum=stratum,
        candidate_color=colour,
        logical_ply=len(history),
        history_moves=history,
        history_sha256=record["history_sha256"],
        sanmill_history_sha256=sanmill_history,
        board=board,
        legal_actions=actions,
        a_pos_mask=mask,
        teacher_action=teacher_action,
        teacher_index=teacher_index,
        teacher_evidence=dict(evidence),
        record_sha256=observed_hash,
    )


def _assert_teacher_record_binds_state(
    teacher_record: Mapping[str, Any],
    state_record: Mapping[str, Any],
) -> None:
    if teacher_record.get("state_record_identity") != state_record.get(
        "state_record_identity"
    ):
        raise CorpusContractError("teacher record state identity differs")
    shared_fields = _STATE_RECORD_FIELDS - {
        "schema_version",
        "state_record_identity",
    }
    if any(
        teacher_record.get(field) != state_record.get(field) for field in shared_fields
    ):
        raise CorpusContractError(
            "teacher record changed the independently frozen state/split record"
        )


def _requery_a_pos_inventory(
    example: FrozenCorpusExample,
    *,
    inventory_verifier: Callable[
        [BoardState, Sequence[Mapping[str, Any]]], Sequence[bool]
    ],
) -> None:
    try:
        observed_raw = tuple(inventory_verifier(example.board, example.legal_actions))
    except CorpusContractError:
        raise
    except Exception as exc:
        raise CorpusContractError(
            "independent A_pos inventory re-query failed"
        ) from exc
    if len(observed_raw) != len(example.legal_actions) or any(
        not isinstance(value, bool) for value in observed_raw
    ):
        raise CorpusContractError(
            "independent A_pos inventory has an invalid legal-order mask"
        )
    if observed_raw != example.a_pos_mask:
        raise CorpusContractError(
            "independent A_pos inventory differs from the frozen mask"
        )


def _load_frozen_corpus(
    manifest_path: str | Path,
    *,
    layout: CorpusLayout,
    inventory_verifier: Callable[
        [BoardState, Sequence[Mapping[str, Any]]], Sequence[bool]
    ],
    oracle_identity: Mapping[str, Any],
    sanmill_runtime_identity: str,
) -> FrozenCorpus:
    if not callable(inventory_verifier):
        raise CorpusContractError(
            "strict corpus loading requires an independent A_pos inventory verifier"
        )
    checked_oracle_identity = _validate_loader_oracle_identity(oracle_identity)
    expected_runtime_identity = _sha256(
        sanmill_runtime_identity,
        field="expected Sanmill runtime identity",
    )
    path = Path(manifest_path).resolve(strict=True)
    manifest = _strict_json_bytes(path.read_bytes(), label="corpus manifest")
    required = {
        "schema_version",
        "corpus_id",
        "profile",
        "status",
        "source",
        "generator",
        "referee",
        "teacher",
        "selection",
        "a_pos_verifier",
        "resources_before",
        "resources_after",
        "state_split_artifact",
        "examples",
        "singleton_ledger",
        "collection_games",
        "split",
        "split_identity",
        "heldout",
        "corpus_identity",
    }
    if set(manifest) != required:
        raise CorpusContractError("corpus manifest keys differ")
    identity_payload = dict(manifest)
    observed_identity = identity_payload.pop("corpus_identity")
    if _sha256(observed_identity, field="corpus_identity") != canonical_sha256(
        identity_payload
    ):
        raise CorpusContractError("corpus manifest identity differs")
    if (
        manifest["schema_version"] != CORPUS_SCHEMA
        or manifest["profile"] != TRAINING_PROFILE
        or manifest["status"] != "complete"
    ):
        raise CorpusContractError("corpus schema/profile/completion differs")
    if layout.total == 16_384 and (
        not isinstance(manifest.get("examples"), Mapping)
        or manifest["examples"].get("count") != 16_384
    ):
        raise CorpusContractError(
            "production corpus must contain the exact 16,384-entry split"
        )
    _validate_source(manifest["source"])
    before = _validate_string_identities(
        manifest["resources_before"],
        keys=RESOURCE_IDENTITY_KEYS,
        field="resources_before",
    )
    after = _validate_string_identities(
        manifest["resources_after"],
        keys=RESOURCE_IDENTITY_KEYS,
        field="resources_after",
    )
    if before != after:
        raise CorpusContractError("teacher resources changed during labeling")
    if checked_oracle_identity["manifest_sha256"] != before["malom_manifest"]:
        raise CorpusContractError("loader A_pos oracle Malom manifest identity differs")
    if checked_oracle_identity["content_sha256"] != before["malom_content"]:
        raise CorpusContractError("loader A_pos oracle Malom content identity differs")
    generator = manifest["generator"]
    if not isinstance(generator, Mapping):
        raise CorpusContractError("generator contract is invalid")
    initial_policy_hash = _sha256(
        generator.get("initial_policy_state_sha256"),
        field="generator.initial_policy_state_sha256",
    )
    if dict(generator) != _generator_contract(initial_policy_hash):
        raise CorpusContractError("frozen generator contract differs")
    if manifest["referee"] != _referee_contract(expected_runtime_identity):
        raise CorpusContractError("frozen referee contract differs")
    if manifest["selection"] != _selection_contract():
        raise CorpusContractError("frozen selection contract differs")
    teacher = manifest["teacher"]
    teacher_fields = {
        "route",
        "identity_claim",
        "product_tt_trajectory_equivalent",
        "fresh_instance_per_label",
        "difficulty",
        "depth",
        "threads",
        "node_cap_per_label",
        "label",
        "online_queries_during_training",
        "active_seconds",
        "active_seconds_limit",
        "main_search_nodes",
        "restricted_rerank_nodes",
        "aggregate_nodes",
        "aggregate_nodes_limit",
        "positive_search_labels",
        "positive_search_labels_limit",
        "completed_labels",
    }
    if not isinstance(teacher, Mapping) or set(teacher) != teacher_fields:
        raise CorpusContractError("teacher manifest keys differ")
    fixed_teacher = {
        "route": "current-head-exact-D9-post-ProductPositionalSafetyGate",
        "identity_claim": "fresh-instance-current-D9-positional-teacher",
        "product_tt_trajectory_equivalent": False,
        "fresh_instance_per_label": True,
        "difficulty": 9,
        "depth": 14,
        "threads": 1,
        "node_cap_per_label": 13_887_000,
        "label": "one-hot-atomic-action-inside-A_pos",
        "online_queries_during_training": False,
        "active_seconds_limit": 4 * 3600,
        "aggregate_nodes_limit": 4_000_000_000,
        "positive_search_labels_limit": 1_280,
        "completed_labels": layout.total,
    }
    if any(teacher.get(key) != value for key, value in fixed_teacher.items()):
        raise CorpusContractError("frozen teacher contract differs")
    active_seconds = teacher["active_seconds"]
    if (
        isinstance(active_seconds, bool)
        or not isinstance(active_seconds, (int, float))
        or not math.isfinite(float(active_seconds))
        or not 0.0 <= float(active_seconds) <= 4 * 3600
    ):
        raise CorpusContractError("teacher active-time ledger differs")
    expected_verifier_identity = a_pos_verifier_identity(manifest["source"], before)
    verifier = manifest["a_pos_verifier"]
    if not isinstance(verifier, Mapping) or set(verifier) != {
        "boundary",
        "identity",
        "record_count",
        "verification_chain_head",
        "loader_requeries_malom",
    }:
        raise CorpusContractError("A_pos verifier manifest keys differ")
    if (
        verifier.get("boundary")
        != "build-time-chain-plus-load-time-real-PositionalSafetyFilter"
        or verifier.get("identity") != expected_verifier_identity
        or verifier.get("record_count") != layout.total
        or verifier.get("loader_requeries_malom") is not True
    ):
        raise CorpusContractError("A_pos verifier trust boundary differs")
    _sha256(
        verifier.get("verification_chain_head"),
        field="verification_chain_head",
    )
    if manifest["heldout"] != {"consumed": False, "sources": []}:
        raise CorpusContractError("consumed heldout data is forbidden")
    expected_counts = layout.to_dict()
    split_contract = manifest["split"]
    if split_contract != _split_contract(layout):
        if layout.total == 16_384:
            raise CorpusContractError(
                "production corpus must contain the exact 16,384-entry split"
            )
        raise CorpusContractError("corpus split cardinalities differ")
    expected_split_identity = canonical_sha256(split_contract)
    if manifest["split_identity"] != expected_split_identity:
        raise CorpusContractError("corpus split identity differs")

    state_artifact = manifest["state_split_artifact"]
    if not isinstance(state_artifact, Mapping) or set(state_artifact) != {
        "schema_version",
        "file",
        "identity",
        "teacher_fields_present",
        "controller_freeze_event_required_before_teacher",
    }:
        raise CorpusContractError("state/split artifact manifest keys differ")
    if (
        state_artifact.get("schema_version") != STATE_SPLIT_ARTIFACT_SCHEMA
        or state_artifact.get("teacher_fields_present") is not False
        or state_artifact.get("controller_freeze_event_required_before_teacher")
        is not True
    ):
        raise CorpusContractError("state/split freeze contract differs")
    state_path = _validate_file_ref(
        path.parent,
        state_artifact["file"],
        field="state/split artifact",
        require_count=True,
    )
    if state_artifact["file"]["count"] != layout.total:
        raise CorpusContractError("state/split artifact count differs")
    (
        state_count,
        state_verification_head,
        state_record_identities,
        state_records,
    ) = _inspect_state_artifact(
        state_path,
        verifier_identity=expected_verifier_identity,
    )
    if state_count != layout.total:
        raise CorpusContractError("state/split artifact is an incomplete prefix")
    expected_state_identity = state_split_artifact_identity(
        state_record_identities=state_record_identities,
        split_identity=expected_split_identity,
        verifier_identity=expected_verifier_identity,
    )
    if (
        _sha256(state_artifact.get("identity"), field="state/split identity")
        != expected_state_identity
    ):
        raise CorpusContractError("state/split artifact identity differs")
    if state_verification_head != verifier["verification_chain_head"]:
        raise CorpusContractError("A_pos verification chain head differs")

    examples_path = _validate_file_ref(
        path.parent,
        manifest["examples"],
        field="examples",
        require_count=True,
    )
    if manifest["examples"]["count"] != layout.total:
        raise CorpusContractError("corpus example count differs")
    singleton_path = _validate_file_ref(
        path.parent,
        manifest["singleton_ledger"],
        field="singleton ledger",
        require_count=False,
    )
    _validate_singleton_ledger(singleton_path, manifest["singleton_ledger"])

    examples: list[FrozenCorpusExample] = []
    with examples_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise CorpusContractError("examples JSONL contains a blank line")
            record = _strict_json_bytes(raw, label=f"example line {line_number}")
            if line_number > len(state_records):
                raise CorpusContractError(
                    "teacher ledger has labels outside the frozen state artifact"
                )
            state_record = state_records[line_number - 1]
            _assert_teacher_record_binds_state(record, state_record)
            example = _validate_record(record)
            _requery_a_pos_inventory(
                example,
                inventory_verifier=inventory_verifier,
            )
            examples.append(example)
    if len(examples) != layout.total:
        raise CorpusContractError("corpus example file is an incomplete prefix")
    if len({item.example_id for item in examples}) != len(examples):
        raise CorpusContractError("corpus contains duplicate example identities")
    if tuple(item.state_record_identity for item in examples) != (
        state_record_identities
    ):
        raise CorpusContractError("teacher ledger state order differs")
    if len({item.history_sha256 for item in examples}) != len(examples):
        raise CorpusContractError(
            "corpus repeats one full-history state across game or split"
        )
    teacher_instances = [
        str(item.teacher_evidence["teacher_instance_id"]) for item in examples
    ]
    if len(set(teacher_instances)) != len(teacher_instances):
        raise CorpusContractError("teacher instance was reused across labels")

    observed_counts = {
        split: {
            stratum: {colour: 0 for colour in ("W", "B")}
            for stratum in ("placement", "movement", "flying")
        }
        for split in ("train", "dev")
    }
    game_splits: dict[str, str] = {}
    game_indices: dict[str, int] = {}
    index_games: dict[int, str] = {}
    main_nodes = 0
    rerank_nodes = 0
    positive_labels = 0
    for item in examples:
        expected_colour = "W" if item.game_index % 2 == 0 else "B"
        expected_split = "dev" if item.game_index % 16 in {0, 1} else "train"
        if item.candidate_color != expected_colour:
            raise CorpusContractError("candidate colour differs from game parity")
        if item.split != expected_split:
            raise CorpusContractError("record split differs from whole-game algorithm")
        if item.game_index >= manifest["collection_games"]:
            raise CorpusContractError("record game index exceeds collection ledger")
        observed_counts[item.split][item.stratum][item.candidate_color] += 1
        existing = game_splits.setdefault(item.game_id, item.split)
        if existing != item.split:
            raise CorpusContractError("one game crosses train/dev split")
        existing_index = game_indices.setdefault(item.game_id, item.game_index)
        if existing_index != item.game_index:
            raise CorpusContractError("one game_id has multiple game indices")
        existing_game = index_games.setdefault(item.game_index, item.game_id)
        if existing_game != item.game_id:
            raise CorpusContractError("one game index has multiple game IDs")
        main = int(item.teacher_evidence["main_search_nodes"])
        rerank = int(item.teacher_evidence["restricted_rerank_nodes"])
        main_nodes += main
        rerank_nodes += rerank
        positive_labels += int(main + rerank > 0)
    if observed_counts != expected_counts:
        raise CorpusContractError("observed split/stratum/colour counts differ")

    total_nodes = main_nodes + rerank_nodes
    if (
        teacher.get("completed_labels") != layout.total
        or teacher.get("main_search_nodes") != main_nodes
        or teacher.get("restricted_rerank_nodes") != rerank_nodes
        or teacher.get("aggregate_nodes") != total_nodes
        or teacher.get("positive_search_labels") != positive_labels
        or teacher.get("aggregate_nodes", 4_000_000_001) > 4_000_000_000
        or teacher.get("positive_search_labels", 1_281) > 1_280
        or teacher.get("active_seconds", 14_401) > 4 * 3600
    ):
        raise CorpusContractError("teacher completion/resource ledger differs")
    collection_games = manifest["collection_games"]
    if (
        isinstance(collection_games, bool)
        or not isinstance(collection_games, int)
        or not 0 < collection_games <= 1_024
        or collection_games % 16 != 0
        or len(game_splits) > collection_games
    ):
        raise CorpusContractError("collection game block ledger differs")
    return FrozenCorpus(
        manifest=manifest,
        examples=tuple(examples),
        corpus_identity=observed_identity,
        split_identity=expected_split_identity,
    )


def load_frozen_corpus(
    manifest_path: str | Path,
    *,
    safety_filter: Any | None = None,
    sanmill_runtime_identity: str | None = None,
) -> FrozenCorpus:
    """Load the production corpus after independent live trust checks.

    The build-time proof chain is provenance, not authentication. Production
    loading therefore requires an actual ``PositionalSafetyFilter`` instance
    and re-queries all 16,384 masks. The preflight also supplies the identity
    of the independently inspected pinned Sanmill runtime.
    """
    from learned_ai.agents.positional_safety import PositionalSafetyFilter

    if not isinstance(safety_filter, PositionalSafetyFilter):
        raise CorpusContractError(
            "production loader requires a real PositionalSafetyFilter oracle"
        )
    if sanmill_runtime_identity is None:
        raise CorpusContractError(
            "production loader requires the inspected Sanmill runtime identity"
        )
    return _load_frozen_corpus(
        manifest_path,
        layout=FROZEN_CORPUS_LAYOUT,
        inventory_verifier=positional_safety_inventory_verifier(safety_filter),
        oracle_identity={
            "label_version": getattr(safety_filter, "label_version", None),
            "manifest_sha256": getattr(
                safety_filter,
                "manifest_sha256",
                None,
            ),
            "content_sha256": getattr(safety_filter, "content_sha256", None),
        },
        sanmill_runtime_identity=sanmill_runtime_identity,
    )

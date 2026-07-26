"""Source-only audit for the maintainer's expert-curated Book plays."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.layered_book_audit import (
    _expand_named_variation,
)
from learned_ai.evaluation.layered_opening_prefix import (
    LAYERED_PREFIX_SCHEMA,
    PREFIX_LOGICAL_PLIES_BY_SIDE_V2,
    PREFIX_LOGICAL_PLIES_V2,
    LayeredOpeningPrefixV2,
    LayeredPrefixError,
    build_layered_prefix_v2,
)
from learned_ai.evaluation.layered_perfect_audit import (
    SourceOverlapIndex,
    load_source_overlap_index,
    verify_layered_perfect_audit,
)
from learned_ai.evaluation.oracle_corpus import ring16_canonical_fen
from learned_ai.evaluation.sanmill_data_query import SanmillDataQuerySession
from learned_ai.evaluation.sanmill_uci import (
    SanmillInstallation,
    nmm_move_base,
    project_stable_sanmill_fen,
)
from learned_ai.training.run_contract import canonical_sha256


EXPERT_BOOK_SOURCE_SCHEMA = "nmm.maintainer-expert-book-play-source.v1"
EXPERT_BOOK_AUDIT_SCHEMA = "nmm.layered-expert-book-source-audit.v1"
EXPERT_BOOK_SOURCE_SUBTYPE = "maintainer_expert_curated_play"
_EVIDENCE_BASES = frozenset(
    {
        "typed_text",
        "typed_text_explicit_alternative",
        "typed_text_plus_embedded_move_list",
    }
)
_SHA256_CHARS = frozenset("0123456789abcdef")
_SHA40_CHARS = _SHA256_CHARS


class LayeredExpertBookAuditError(LayeredPrefixError):
    """Raised when expert Book source or audit evidence is inconsistent."""


@dataclass(frozen=True)
class ExpertBookSource:
    """Validated portable transcription and its content identity."""

    payload: Mapping[str, Any]
    file_record: Mapping[str, Any]
    source_identity: Mapping[str, Any]


@dataclass(frozen=True)
class ExpertBookCandidate:
    """One fully resolved project-rules history from one source variation."""

    entry_id: str
    source_row: int
    variation_id: str
    label: str
    evidence_basis: str
    normalization_notes: tuple[str, ...]
    image_sha256: str
    author_tokens: tuple[str, ...]
    logical_turns: tuple[tuple[str, ...], ...]
    action_tokens: tuple[str, ...]
    exact_history_sha256: str
    parent8_action_tokens: tuple[str, ...]
    parent8_exact_history_sha256: str
    parent8_nmm_fen: str
    parent8_ring16_fen: str
    final_nmm_fen: str
    final_ring16_fen: str


@dataclass(frozen=True)
class ExpertSourceOverlapIndex:
    """Verified prior-source sets and matching HumanDB frequency records."""

    evidence: Mapping[str, Any]
    sanmill_book_exact: frozenset[str]
    sanmill_book_fen: frozenset[str]
    sanmill_book_orbit: frozenset[str]
    human_exact: frozenset[str]
    human_fen: frozenset[str]
    human_orbit: frozenset[str]
    perfect_exact: frozenset[str]
    perfect_fen: frozenset[str]
    perfect_orbit: frozenset[str]
    human_exact_support: Mapping[str, Mapping[str, Any]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _SHA256_CHARS for char in value)
    ):
        raise LayeredExpertBookAuditError(
            f"{context} must be a lowercase SHA-256"
        )
    return value


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LayeredExpertBookAuditError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise LayeredExpertBookAuditError(
            f"{context} has a non-string field"
        )
    return dict(value)


def _string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise LayeredExpertBookAuditError(f"{context} must be non-empty text")
    return value


def _portable_relative_path(value: Any, *, context: str) -> str:
    text = _string(value, context=context)
    if Path(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise LayeredExpertBookAuditError(f"{context} must be relative")
    return Path(text).as_posix()


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayeredExpertBookAuditError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise LayeredExpertBookAuditError(f"{label} is not a JSON object")
    return value


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as exc:
        raise LayeredExpertBookAuditError(
            "expert Book source must be inside the repository"
        ) from exc


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "status",
        "candidate_loaded",
        "games_played",
        "delivery",
        "expert_context",
        "normalization_contract",
        "entries",
        "transcription_identity",
    }
    if set(payload) != expected:
        raise LayeredExpertBookAuditError(
            "expert Book source top-level fields drifted"
        )
    if (
        payload["schema_version"] != EXPERT_BOOK_SOURCE_SCHEMA
        or payload["status"] != "source-only-audit-candidate"
        or payload["candidate_loaded"] is not False
        or payload["games_played"] != 0
    ):
        raise LayeredExpertBookAuditError(
            "expert Book source scope boundary drifted"
        )
    body = dict(payload)
    identity = _sha256(
        body.pop("transcription_identity"),
        context="expert Book transcription identity",
    )
    if canonical_sha256(body) != identity:
        raise LayeredExpertBookAuditError(
            "expert Book transcription identity mismatch"
        )

    delivery = _mapping(payload["delivery"], context="expert Book delivery")
    archive = _mapping(
        delivery.get("archive"),
        context="expert Book archive record",
    )
    _portable_relative_path(
        archive.get("relative_path"),
        context="expert Book archive path",
    )
    if (
        not isinstance(archive.get("byte_length"), int)
        or archive["byte_length"] <= 0
    ):
        raise LayeredExpertBookAuditError(
            "expert Book archive byte length is invalid"
        )
    _sha256(archive.get("sha256"), context="expert Book document identity")

    contract = payload["normalization_contract"]
    if contract != {
        "target_prefix_schema": LAYERED_PREFIX_SCHEMA,
        "logical_ply_count": PREFIX_LOGICAL_PLIES_V2,
        "logical_plies_by_side": list(PREFIX_LOGICAL_PLIES_BY_SIDE_V2),
        "mill_and_required_capture_count_as_one_logical_ply": True,
        "source_rows_preserved_before_deduplication": True,
        "row_1_explicit_alternatives_preserved": True,
        "row_11_missing_typed_token_policy": (
            "The final black c5 is transcribed from the embedded move-list "
            "screenshot and remains explicitly marked as visual "
            "interpretation."
        ),
        "selection_status": "audit_candidate_not_frozen",
        "final_corpus_membership_frozen": False,
    }:
        raise LayeredExpertBookAuditError(
            "expert Book normalization contract drifted"
        )

    entries = payload["entries"]
    if not isinstance(entries, list) or not entries:
        raise LayeredExpertBookAuditError("expert Book entries are absent")
    if delivery.get("table_row_count") != len(entries):
        raise LayeredExpertBookAuditError(
            "expert Book table-row count drifted"
        )
    if delivery.get("embedded_image_count") != len(entries):
        raise LayeredExpertBookAuditError(
            "expert Book image count drifted"
        )

    expected_rows = list(range(1, len(entries) + 1))
    if [entry.get("source_row") for entry in entries] != expected_rows:
        raise LayeredExpertBookAuditError(
            "expert Book source rows are not contiguous"
        )
    entry_ids: set[str] = set()
    variation_ids: set[str] = set()
    media_names: set[str] = set()
    for entry in entries:
        record = _mapping(entry, context="expert Book entry")
        if set(record) != {
            "entry_id",
            "source_row",
            "source_text",
            "embedded_image",
            "normalization_notes",
            "variations",
        }:
            raise LayeredExpertBookAuditError(
                "expert Book entry fields drifted"
            )
        entry_id = _string(
            record["entry_id"],
            context="expert Book entry id",
        )
        if entry_id in entry_ids:
            raise LayeredExpertBookAuditError(
                "expert Book entry id is duplicated"
            )
        entry_ids.add(entry_id)
        _string(record["source_text"], context="expert Book source text")
        image = _mapping(
            record["embedded_image"],
            context="expert Book embedded image",
        )
        if set(image) != {"media_name", "sha256"}:
            raise LayeredExpertBookAuditError(
                "expert Book embedded-image fields drifted"
            )
        media_name = _string(
            image["media_name"],
            context="expert Book media name",
        )
        _sha256(image["sha256"], context="expert Book image identity")
        if media_name in media_names:
            raise LayeredExpertBookAuditError(
                "expert Book media name is duplicated"
            )
        media_names.add(media_name)
        notes = record["normalization_notes"]
        if not isinstance(notes, list) or any(
            not isinstance(note, str) or not note for note in notes
        ):
            raise LayeredExpertBookAuditError(
                "expert Book normalization notes are invalid"
            )
        variations = record["variations"]
        if not isinstance(variations, list) or not variations:
            raise LayeredExpertBookAuditError(
                "expert Book entry has no variation"
            )
        for variation in variations:
            item = _mapping(
                variation,
                context="expert Book variation",
            )
            if set(item) != {
                "variation_id",
                "author_tokens",
                "evidence_basis",
                "label",
            }:
                raise LayeredExpertBookAuditError(
                    "expert Book variation fields drifted"
                )
            variation_id = _string(
                item["variation_id"],
                context="expert Book variation id",
            )
            if variation_id in variation_ids:
                raise LayeredExpertBookAuditError(
                    "expert Book variation id is duplicated"
                )
            variation_ids.add(variation_id)
            tokens = item["author_tokens"]
            if (
                not isinstance(tokens, list)
                or len(tokens) != PREFIX_LOGICAL_PLIES_V2
                or any(not isinstance(token, str) or not token for token in tokens)
            ):
                raise LayeredExpertBookAuditError(
                    "expert Book variation is not twelve typed tokens"
                )
            if item["evidence_basis"] not in _EVIDENCE_BASES:
                raise LayeredExpertBookAuditError(
                    "expert Book evidence basis is unsupported"
                )
            _string(item["label"], context="expert Book variation label")


def load_expert_book_source(path: str | Path) -> ExpertBookSource:
    """Load and content-identify the tracked expert Book transcription."""
    source_path = Path(path).resolve()
    payload = _load_json(source_path, label="expert Book transcription")
    _validate_source_payload(payload)
    delivery = _mapping(payload["delivery"], context="expert Book delivery")
    archive = _mapping(
        delivery["archive"],
        context="expert Book archive record",
    )
    file_record = {
        "relative_path": _repo_relative(source_path),
        "byte_length": source_path.stat().st_size,
        "sha256": _file_sha256(source_path),
        "schema_version": payload["schema_version"],
        "transcription_identity": payload["transcription_identity"],
        "document": {
            "relative_archive_path": archive["relative_path"],
            "byte_length": archive["byte_length"],
            "sha256": archive["sha256"],
            "source_row_count": delivery["table_row_count"],
            "embedded_image_count": delivery["embedded_image_count"],
        },
    }
    identity_body = {
        "kind": "book",
        "source_subtype": EXPERT_BOOK_SOURCE_SUBTYPE,
        "document_sha256": archive["sha256"],
        "document_byte_length": archive["byte_length"],
        "transcription_schema": payload["schema_version"],
        "transcription_identity": payload["transcription_identity"],
    }
    source_identity = {
        "kind": "book",
        "identity": identity_body,
        "identity_sha256": canonical_sha256(identity_body),
    }
    return ExpertBookSource(
        payload=payload,
        file_record=file_record,
        source_identity=source_identity,
    )


def _turn_actions(move: Mapping[str, Any]) -> tuple[str, ...]:
    primary = nmm_move_base(move)
    capture = move.get("capture")
    return (primary,) if capture is None else (primary, f"x{capture}")


def _replay_logical_turns(
    turns: Sequence[Sequence[str]],
) -> BoardState:
    board = BoardState.new_game()
    for turn in turns:
        expected = tuple(turn)
        matching = [
            move
            for move in get_all_legal_moves(board)
            if _turn_actions(move) == expected
        ]
        if len(matching) != 1:
            raise LayeredExpertBookAuditError(
                "resolved expert Book turn is not uniquely legal"
            )
        board = board.apply_move(matching[0])
    return board


def prepare_expert_book_candidates(
    source: ExpertBookSource,
) -> tuple[ExpertBookCandidate, ...]:
    """Resolve every source variation under project rules without fallback."""
    candidates: list[ExpertBookCandidate] = []
    for raw_entry in source.payload["entries"]:
        entry = _mapping(raw_entry, context="expert Book entry")
        image = _mapping(
            entry["embedded_image"],
            context="expert Book embedded image",
        )
        notes = tuple(str(note) for note in entry["normalization_notes"])
        for raw_variation in entry["variations"]:
            variation = _mapping(
                raw_variation,
                context="expert Book variation",
            )
            author_tokens = tuple(str(token) for token in variation["author_tokens"])
            expansion = _expand_named_variation(
                {
                    "id": variation["variation_id"],
                    "lineMoves": list(author_tokens),
                }
            )
            histories = expansion["expanded_histories"]
            if expansion["status"] != "complete" or len(histories) != 1:
                raise LayeredExpertBookAuditError(
                    "expert Book variation is illegal or capture-ambiguous"
                )
            logical_turns, final_board = histories[0]
            if len(logical_turns) != PREFIX_LOGICAL_PLIES_V2:
                raise LayeredExpertBookAuditError(
                    "expert Book variation did not resolve to twelve turns"
                )
            replayed_final = _replay_logical_turns(logical_turns)
            if replayed_final != final_board:
                raise LayeredExpertBookAuditError(
                    "expert Book project replays disagree"
                )
            parent_board = _replay_logical_turns(logical_turns[:8])
            actions = tuple(
                token for turn in logical_turns for token in turn
            )
            parent_actions = tuple(
                token for turn in logical_turns[:8] for token in turn
            )
            final_fen = final_board.to_fen_string()
            parent_fen = parent_board.to_fen_string()
            candidates.append(
                ExpertBookCandidate(
                    entry_id=str(entry["entry_id"]),
                    source_row=int(entry["source_row"]),
                    variation_id=str(variation["variation_id"]),
                    label=str(variation["label"]),
                    evidence_basis=str(variation["evidence_basis"]),
                    normalization_notes=notes,
                    image_sha256=str(image["sha256"]),
                    author_tokens=author_tokens,
                    logical_turns=tuple(
                        tuple(turn) for turn in logical_turns
                    ),
                    action_tokens=actions,
                    exact_history_sha256=canonical_sha256(list(actions)),
                    parent8_action_tokens=parent_actions,
                    parent8_exact_history_sha256=canonical_sha256(
                        list(parent_actions)
                    ),
                    parent8_nmm_fen=parent_fen,
                    parent8_ring16_fen=ring16_canonical_fen(parent_fen),
                    final_nmm_fen=final_fen,
                    final_ring16_fen=ring16_canonical_fen(final_fen),
                )
            )
    return tuple(candidates)


def _perfect_sets(
    payload: Mapping[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    exact: set[str] = set()
    fens: set[str] = set()
    orbits: set[str] = set()
    for route in payload["routes"]:
        exact.add(str(route["exact_history_sha256"]))
        final = route["prefix_record"]["final"]
        fens.add(str(final["nmm_fen"]))
        orbits.add(str(final["ring16_canonical_fen"]))
    return exact, fens, orbits


def _human_support_records(
    ledger_path: Path,
    targets: frozenset[str],
) -> dict[str, Mapping[str, Any]]:
    matches: dict[str, Mapping[str, Any]] = {}
    try:
        with ledger_path.open("r", encoding="utf-8") as handle:
            next(handle)
            for raw_line in handle:
                record = json.loads(raw_line)
                identity = str(record.get("history_identity", ""))
                if identity not in targets:
                    continue
                if canonical_sha256(record.get("action_tokens")) != identity:
                    raise LayeredExpertBookAuditError(
                        "matching HumanDB history identity drifted"
                    )
                matches[identity] = {
                    "history_identity": identity,
                    "occurrence_count": int(record["occurrence_count"]),
                    "distinct_game_count": int(record["distinct_game_count"]),
                    "results": dict(record["results"]),
                    "final": dict(record["final"]),
                }
    except (OSError, StopIteration, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayeredExpertBookAuditError(
            "cannot read matching HumanDB support records"
        ) from exc
    return matches


def load_expert_source_overlap_index(
    *,
    book_audit_path: str | Path,
    human_audit_path: str | Path,
    human_ledger_path: str | Path,
    perfect_audit_path: str | Path,
    expert_exact_history_ids: Sequence[str],
) -> ExpertSourceOverlapIndex:
    """Verify all prior audits and collect support for expert exact histories."""
    base: SourceOverlapIndex = load_source_overlap_index(
        book_audit_path=book_audit_path,
        human_audit_path=human_audit_path,
        human_ledger_path=human_ledger_path,
    )
    perfect_path = Path(perfect_audit_path).resolve()
    perfect = _load_json(perfect_path, label="Perfect DB audit evidence")
    verify_layered_perfect_audit(perfect)
    if perfect["overlap"]["comparison_inputs"] != base.evidence:
        raise LayeredExpertBookAuditError(
            "Perfect DB audit used different Book or HumanDB evidence"
        )
    perfect_exact, perfect_fen, perfect_orbit = _perfect_sets(perfect)
    ledger_path = Path(human_ledger_path).resolve()
    targets = frozenset(
        _sha256(value, context="expert exact history")
        for value in expert_exact_history_ids
    )
    support = _human_support_records(ledger_path, targets)
    evidence = {
        **dict(base.evidence),
        "perfect_audit": {
            "relative_path": _repo_relative(perfect_path),
            "byte_length": perfect_path.stat().st_size,
            "sha256": _file_sha256(perfect_path),
            "audit_identity": perfect["audit_identity"],
        },
    }
    return ExpertSourceOverlapIndex(
        evidence=evidence,
        sanmill_book_exact=base.book_exact,
        sanmill_book_fen=base.book_fen,
        sanmill_book_orbit=base.book_orbit,
        human_exact=base.human_exact,
        human_fen=base.human_fen,
        human_orbit=base.human_orbit,
        perfect_exact=frozenset(perfect_exact),
        perfect_fen=frozenset(perfect_fen),
        perfect_orbit=frozenset(perfect_orbit),
        human_exact_support=support,
    )


def _step_evidence(
    candidate: ExpertBookCandidate,
) -> list[dict[str, Any]]:
    return [
        {
            "entry_id": candidate.entry_id,
            "source_row": candidate.source_row,
            "variation_id": candidate.variation_id,
            "author_token_index": index,
            "author_token": author_token,
            "resolved_action_tokens": list(actions),
            "evidence_basis": candidate.evidence_basis,
            "visual_interpretation": (
                candidate.evidence_basis
                == "typed_text_plus_embedded_move_list"
                and index == PREFIX_LOGICAL_PLIES_V2 - 1
            ),
        }
        for index, (author_token, actions) in enumerate(
            zip(
                candidate.author_tokens,
                candidate.logical_turns,
                strict=True,
            )
        )
    ]


def _record_overlap(
    candidate: ExpertBookCandidate,
    overlap: ExpertSourceOverlapIndex,
) -> dict[str, dict[str, bool]]:
    return {
        "sanmill_book": {
            "exact_history": (
                candidate.exact_history_sha256 in overlap.sanmill_book_exact
            ),
            "final_fen": candidate.final_nmm_fen in overlap.sanmill_book_fen,
            "ring16_orbit": (
                candidate.final_ring16_fen in overlap.sanmill_book_orbit
            ),
        },
        "human_db": {
            "exact_history": (
                candidate.exact_history_sha256 in overlap.human_exact
            ),
            "final_fen": candidate.final_nmm_fen in overlap.human_fen,
            "ring16_orbit": (
                candidate.final_ring16_fen in overlap.human_orbit
            ),
        },
        "perfect_db": {
            "exact_history": (
                candidate.exact_history_sha256 in overlap.perfect_exact
            ),
            "final_fen": candidate.final_nmm_fen in overlap.perfect_fen,
            "ring16_orbit": (
                candidate.final_ring16_fen in overlap.perfect_orbit
            ),
        },
    }


def _references(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "entry_id": record["entry_id"],
            "source_row": record["source_row"],
            "variation_id": record["variation_id"],
        }
        for record in sorted(
            records,
            key=lambda item: (
                int(item["source_row"]),
                str(item["variation_id"]),
            ),
        )
    ]


def _multiplicity(values: Sequence[str]) -> list[dict[str, int]]:
    groups = Counter(Counter(values).values())
    return [
        {
            "multiplicity": multiplicity,
            "value_count": groups[multiplicity],
        }
        for multiplicity in sorted(groups)
    ]


def _duplicate_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[value_key])].append(record)
    return [
        {
            "value": value,
            "candidate_record_count": len(group),
            "source_references": _references(group),
        }
        for value, group in sorted(groups.items())
        if len(group) > 1
    ]


def _parent_groups(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["parent8"]["ring16_canonical_fen"])].append(record)
    result = []
    for orbit, group in sorted(groups.items()):
        result.append(
            {
                "ring16_canonical_fen": orbit,
                "candidate_record_count": len(group),
                "unique_exact_history_count": len(
                    {
                        str(item["parent8"]["exact_history_sha256"])
                        for item in group
                    }
                ),
                "unique_exact_fen_count": len(
                    {str(item["parent8"]["nmm_fen"]) for item in group}
                ),
                "source_references": _references(group),
            }
        )
    return result


def _source_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact = [str(record["exact_history_sha256"]) for record in records]
    fens = [str(record["prefix_record"]["final"]["nmm_fen"]) for record in records]
    orbits = [
        str(record["prefix_record"]["final"]["ring16_canonical_fen"])
        for record in records
    ]
    parent_exact = [
        str(record["parent8"]["exact_history_sha256"]) for record in records
    ]
    parent_fens = [str(record["parent8"]["nmm_fen"]) for record in records]
    parent_orbits = [
        str(record["parent8"]["ring16_canonical_fen"]) for record in records
    ]
    return {
        "source_row_count": len(
            {int(record["source_row"]) for record in records}
        ),
        "source_variation_count": len(records),
        "legal_prefix_record_count": len(records),
        "visual_interpretation_record_count": sum(
            record["evidence_basis"]
            == "typed_text_plus_embedded_move_list"
            for record in records
        ),
        "unique_exact_history_count": len(set(exact)),
        "unique_exact_final_fen_count": len(set(fens)),
        "unique_ring16_final_orbit_count": len(set(orbits)),
        "exact_history_multiplicity": _multiplicity(exact),
        "exact_final_fen_multiplicity": _multiplicity(fens),
        "ring16_final_orbit_multiplicity": _multiplicity(orbits),
        "duplicate_exact_history_groups": _duplicate_groups(
            records,
            value_key="exact_history_sha256",
        ),
        "duplicate_exact_final_fen_groups": _duplicate_groups(
            records,
            value_key="_final_nmm_fen",
        ),
        "duplicate_ring16_final_orbit_groups": _duplicate_groups(
            records,
            value_key="_final_ring16_fen",
        ),
        "parent8": {
            "logical_ply_count": 8,
            "unique_exact_history_count": len(set(parent_exact)),
            "unique_exact_final_fen_count": len(set(parent_fens)),
            "unique_ring16_final_orbit_count": len(set(parent_orbits)),
            "exact_history_multiplicity": _multiplicity(parent_exact),
            "exact_final_fen_multiplicity": _multiplicity(parent_fens),
            "ring16_final_orbit_multiplicity": _multiplicity(parent_orbits),
            "ring16_groups": _parent_groups(records),
        },
    }


def _one_source_overlap(
    records: Sequence[Mapping[str, Any]],
    *,
    source_key: str,
) -> dict[str, Any]:
    exact_all = [
        str(record["exact_history_sha256"])
        for record in records
        if record["overlap"][source_key]["exact_history"]
    ]
    fen_all = [
        str(record["_final_nmm_fen"])
        for record in records
        if record["overlap"][source_key]["final_fen"]
    ]
    orbit_all = [
        str(record["_final_ring16_fen"])
        for record in records
        if record["overlap"][source_key]["ring16_orbit"]
    ]
    return {
        "candidate_record_counts": {
            "exact_history": len(exact_all),
            "final_fen": len(fen_all),
            "ring16_orbit": len(orbit_all),
        },
        "unique_value_counts": {
            "exact_history": len(set(exact_all)),
            "final_fen": len(set(fen_all)),
            "ring16_orbit": len(set(orbit_all)),
        },
    }


def _human_support_summary(
    records: Sequence[Mapping[str, Any]],
    support: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        identity = str(record["exact_history_sha256"])
        if identity in support:
            groups[identity].append(record)
    matches = []
    for identity, group in sorted(groups.items()):
        item = support[identity]
        matches.append(
            {
                "history_identity": identity,
                "source_references": _references(group),
                "occurrence_count": int(item["occurrence_count"]),
                "distinct_game_count": int(item["distinct_game_count"]),
                "results": dict(item["results"]),
                "final": dict(item["final"]),
            }
        )
    return {
        "matched_unique_history_count": len(matches),
        "matched_candidate_record_count": sum(
            len(item["source_references"]) for item in matches
        ),
        "total_distinct_game_support": sum(
            int(item["distinct_game_count"]) for item in matches
        ),
        "maximum_distinct_game_support": max(
            (int(item["distinct_game_count"]) for item in matches),
            default=0,
        ),
        "matches": matches,
        "interpretation": (
            "Observed support in the frozen current PlayOK sample; outcome "
            "counts do not certify opening quality or causal strength."
        ),
    }


def _overlap_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    comparison_inputs: Mapping[str, Any],
    human_support: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "comparison_inputs": dict(comparison_inputs),
        "with_sanmill_book": _one_source_overlap(
            records,
            source_key="sanmill_book",
        ),
        "with_human_db": {
            **_one_source_overlap(records, source_key="human_db"),
            "exact_history_support": _human_support_summary(
                records,
                human_support,
            ),
        },
        "with_perfect_db": _one_source_overlap(
            records,
            source_key="perfect_db",
        ),
    }


def build_layered_expert_book_audit(
    session: SanmillDataQuerySession,
    installation: SanmillInstallation,
    *,
    source: ExpertBookSource,
    candidates: Sequence[ExpertBookCandidate],
    overlap: ExpertSourceOverlapIndex,
    generator_commit: str,
    fresh_processes: int = 2,
) -> dict[str, Any]:
    """Replay and audit all expert Book candidates without source fallback."""
    if (
        len(generator_commit) != 40
        or any(char not in _SHA40_CHARS for char in generator_commit)
    ):
        raise LayeredExpertBookAuditError(
            "generator commit must be a full Git SHA"
        )
    if fresh_processes < 2:
        raise LayeredExpertBookAuditError(
            "expert Book audit requires at least two fresh processes"
        )
    if not candidates:
        raise LayeredExpertBookAuditError(
            "expert Book audit has no source candidates"
        )

    records: list[dict[str, Any]] = []
    for candidate in candidates:
        source_history_id = canonical_sha256(
            {
                "schema": EXPERT_BOOK_AUDIT_SCHEMA,
                "source_identity_sha256": source.source_identity[
                    "identity_sha256"
                ],
                "entry_id": candidate.entry_id,
                "variation_id": candidate.variation_id,
                "action_tokens": list(candidate.action_tokens),
            }
        )
        prefix = build_layered_prefix_v2(
            session,
            installation,
            stratum="book",
            source_subtype=EXPERT_BOOK_SOURCE_SUBTYPE,
            source_history_id=source_history_id,
            source_identity=source.source_identity,
            source_evidence={
                "entry_id": candidate.entry_id,
                "source_row": candidate.source_row,
                "variation_id": candidate.variation_id,
                "label": candidate.label,
                "evidence_basis": candidate.evidence_basis,
                "normalization_notes": list(
                    candidate.normalization_notes
                ),
                "embedded_image_sha256": candidate.image_sha256,
                "document_sha256": source.file_record["document"]["sha256"],
                "transcription_identity": source.file_record[
                    "transcription_identity"
                ],
                "selection_status": "audit_candidate_not_frozen",
            },
            logical_turns=candidate.logical_turns,
            step_evidence=_step_evidence(candidate),
        )
        if (
            prefix.final_nmm_fen != candidate.final_nmm_fen
            or prefix.final_ring16_fen != candidate.final_ring16_fen
        ):
            raise LayeredExpertBookAuditError(
                "Sanmill and project final states disagree"
            )
        parent_board = project_stable_sanmill_fen(
            prefix.steps[7].output_state["fen"]
        )
        parent_nmm_fen = parent_board.to_fen_string()
        if (
            parent_nmm_fen != candidate.parent8_nmm_fen
            or ring16_canonical_fen(parent_nmm_fen)
            != candidate.parent8_ring16_fen
        ):
            raise LayeredExpertBookAuditError(
                "Sanmill and project eight-ply parent states disagree"
            )
        record = {
            "entry_id": candidate.entry_id,
            "source_row": candidate.source_row,
            "variation_id": candidate.variation_id,
            "label": candidate.label,
            "evidence_basis": candidate.evidence_basis,
            "author_tokens": list(candidate.author_tokens),
            "resolved_logical_turns": [
                list(turn) for turn in candidate.logical_turns
            ],
            "exact_history_sha256": candidate.exact_history_sha256,
            "parent8": {
                "action_tokens": list(candidate.parent8_action_tokens),
                "exact_history_sha256": (
                    candidate.parent8_exact_history_sha256
                ),
                "nmm_fen": candidate.parent8_nmm_fen,
                "ring16_canonical_fen": candidate.parent8_ring16_fen,
            },
            "prefix_record": prefix.to_dict(),
            "overlap": _record_overlap(candidate, overlap),
            "_final_nmm_fen": candidate.final_nmm_fen,
            "_final_ring16_fen": candidate.final_ring16_fen,
        }
        records.append(record)

    records.sort(key=lambda item: (item["source_row"], item["variation_id"]))
    summary = _source_summary(records)
    overlap_summary = _overlap_summary(
        records,
        comparison_inputs=overlap.evidence,
        human_support=overlap.human_exact_support,
    )
    for record in records:
        record.pop("_final_nmm_fen")
        record.pop("_final_ring16_fen")
    body = {
        "schema_version": EXPERT_BOOK_AUDIT_SCHEMA,
        "status": "source-only-needs-decision",
        "candidate_loaded": False,
        "games_played": 0,
        "fallback": "none",
        "target": {
            "prefix_schema": LAYERED_PREFIX_SCHEMA,
            "logical_ply_count": PREFIX_LOGICAL_PLIES_V2,
            "logical_plies_by_side": list(
                PREFIX_LOGICAL_PLIES_BY_SIDE_V2
            ),
        },
        "generator": {
            "algorithm": "maintainer-expert-book-play-audit-v1",
            "nmm_llm_commit": generator_commit,
            "fresh_processes": fresh_processes,
            "byte_identical_runs_required": True,
        },
        "sanmill": installation.portable_record(),
        "source": dict(source.file_record),
        "source_identity": dict(source.source_identity),
        "summary": summary,
        "overlap": overlap_summary,
        "records": records,
        "decision": {
            "final_corpus_frozen": False,
            "book_quota_frozen": False,
            "expert_book_membership_frozen": False,
            "row_11_visual_completion_confirmed": False,
            "selection_status": (
                "fixed source-audit candidates only; no corpus membership "
                "frozen"
            ),
            "next_gate": (
                "review parent-family coverage, duplicate and cross-source "
                "evidence before revising the corpus decision brief"
            ),
        },
    }
    return {**body, "audit_identity": canonical_sha256(body)}


def _records_with_private_final_fields(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for raw_record in records:
        record = dict(raw_record)
        final = record["prefix_record"]["final"]
        record["_final_nmm_fen"] = final["nmm_fen"]
        record["_final_ring16_fen"] = final["ring16_canonical_fen"]
        result.append(record)
    return result


def verify_layered_expert_book_audit(
    payload: Mapping[str, Any],
) -> dict[str, int]:
    """Verify scope, identities, prefix records, summaries, and decisions."""
    expected = {
        "schema_version",
        "status",
        "candidate_loaded",
        "games_played",
        "fallback",
        "target",
        "generator",
        "sanmill",
        "source",
        "source_identity",
        "summary",
        "overlap",
        "records",
        "decision",
        "audit_identity",
    }
    if set(payload) != expected:
        raise LayeredExpertBookAuditError(
            "expert Book audit top-level fields drifted"
        )
    if (
        payload["schema_version"] != EXPERT_BOOK_AUDIT_SCHEMA
        or payload["status"] != "source-only-needs-decision"
        or payload["candidate_loaded"] is not False
        or payload["games_played"] != 0
        or payload["fallback"] != "none"
    ):
        raise LayeredExpertBookAuditError(
            "expert Book audit scope boundary drifted"
        )
    if payload["target"] != {
        "prefix_schema": LAYERED_PREFIX_SCHEMA,
        "logical_ply_count": PREFIX_LOGICAL_PLIES_V2,
        "logical_plies_by_side": list(PREFIX_LOGICAL_PLIES_BY_SIDE_V2),
    }:
        raise LayeredExpertBookAuditError(
            "expert Book audit target drifted"
        )
    body = dict(payload)
    identity = body.pop("audit_identity")
    if canonical_sha256(body) != identity:
        raise LayeredExpertBookAuditError(
            "expert Book audit identity mismatch"
        )
    source_identity = payload["source_identity"]
    if (
        source_identity.get("kind") != "book"
        or source_identity.get("identity", {}).get("source_subtype")
        != EXPERT_BOOK_SOURCE_SUBTYPE
        or canonical_sha256(source_identity.get("identity"))
        != source_identity.get("identity_sha256")
    ):
        raise LayeredExpertBookAuditError(
            "expert Book portable source identity drifted"
        )
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise LayeredExpertBookAuditError(
            "expert Book audit records are absent"
        )

    variation_ids: set[str] = set()
    for index, record in enumerate(records):
        if index and (
            record["source_row"],
            record["variation_id"],
        ) <= (
            records[index - 1]["source_row"],
            records[index - 1]["variation_id"],
        ):
            raise LayeredExpertBookAuditError(
                "expert Book record ordering drifted"
            )
        variation_id = str(record["variation_id"])
        if variation_id in variation_ids:
            raise LayeredExpertBookAuditError(
                "expert Book audit variation is duplicated"
            )
        variation_ids.add(variation_id)
        prefix = LayeredOpeningPrefixV2.from_dict(record["prefix_record"])
        exact = canonical_sha256(list(prefix.action_tokens))
        parent_actions = tuple(
            token
            for step in prefix.steps[:8]
            for token in step.action_tokens
        )
        parent = record["parent8"]
        if (
            prefix.stratum != "book"
            or prefix.source_subtype != EXPERT_BOOK_SOURCE_SUBTYPE
            or prefix.source_identity != source_identity
            or record["exact_history_sha256"] != exact
            or list(prefix.action_tokens)
            != [
                token
                for turn in record["resolved_logical_turns"]
                for token in turn
            ]
            or parent["action_tokens"] != list(parent_actions)
            or parent["exact_history_sha256"]
            != canonical_sha256(list(parent_actions))
            or prefix.source_evidence["entry_id"] != record["entry_id"]
            or prefix.source_evidence["source_row"] != record["source_row"]
            or prefix.source_evidence["variation_id"] != variation_id
        ):
            raise LayeredExpertBookAuditError(
                "expert Book prefix record is inconsistent"
            )
        parent_board = project_stable_sanmill_fen(
            prefix.steps[7].output_state["fen"]
        )
        parent_fen = parent_board.to_fen_string()
        if (
            parent["nmm_fen"] != parent_fen
            or parent["ring16_canonical_fen"]
            != ring16_canonical_fen(parent_fen)
        ):
            raise LayeredExpertBookAuditError(
                "expert Book parent state is inconsistent"
            )

    private_records = _records_with_private_final_fields(records)
    if _source_summary(private_records) != payload["summary"]:
        raise LayeredExpertBookAuditError(
            "expert Book source summary drifted"
        )
    human_support = {
        item["history_identity"]: item
        for item in payload["overlap"]["with_human_db"][
            "exact_history_support"
        ]["matches"]
    }
    if _overlap_summary(
        private_records,
        comparison_inputs=payload["overlap"]["comparison_inputs"],
        human_support=human_support,
    ) != payload["overlap"]:
        raise LayeredExpertBookAuditError(
            "expert Book overlap summary drifted"
        )
    if payload["decision"] != {
        "final_corpus_frozen": False,
        "book_quota_frozen": False,
        "expert_book_membership_frozen": False,
        "row_11_visual_completion_confirmed": False,
        "selection_status": (
            "fixed source-audit candidates only; no corpus membership frozen"
        ),
        "next_gate": (
            "review parent-family coverage, duplicate and cross-source "
            "evidence before revising the corpus decision brief"
        ),
    }:
        raise LayeredExpertBookAuditError(
            "expert Book decision boundary drifted"
        )
    return {
        "source_rows": payload["summary"]["source_row_count"],
        "source_variations": payload["summary"]["source_variation_count"],
        "legal_records": payload["summary"]["legal_prefix_record_count"],
        "unique_histories": payload["summary"]["unique_exact_history_count"],
        "unique_final_fens": payload["summary"][
            "unique_exact_final_fen_count"
        ],
        "unique_ring16_orbits": payload["summary"][
            "unique_ring16_final_orbit_count"
        ],
        "parent8_ring16_orbits": payload["summary"]["parent8"][
            "unique_ring16_final_orbit_count"
        ],
        "human_exact_matches": payload["overlap"]["with_human_db"][
            "exact_history_support"
        ]["matched_unique_history_count"],
    }

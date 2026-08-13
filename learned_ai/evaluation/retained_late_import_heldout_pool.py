"""Candidate-blind held-out source pool from HumanDB-unseen PlayOK imports."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ai.human_db import HumanDB
from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.evaluation.oracle_corpus import ring16_canonical_fen
from learned_ai.evaluation.phase_corpus import _phase
from learned_ai.evaluation.sanmill_uci import nmm_move_base
from learned_ai.training.generalist_preflight import (
    _probe_human_db,
    _probe_specialist_db,
)
from learned_ai.training.run_contract import canonical_sha256
from learned_ai.training.sanmill_referee import (
    SanmillTrainingGame,
    nmm_move_actions,
    training_installation_record,
)


POOL_SCHEMA = "nmm.retained-late-import-heldout-pool.v1"
POOL_ID = "sanmill-retained-v3-v4-late-import-heldout-pool-v1"
POOL_STATUS = "frozen_source_only_awaiting_precision_plan_and_authorization"
SELECTION_DOMAIN = f"{POOL_ID}:selection:v1"
MINIMUM_LOGICAL_PLY = 12
PHASE_ORDER = ("placement", "movement", "flying")
GAME_FILE_PATTERN = re.compile(r"human_(ml\d+)\.jsonl$")
FEN_PATTERN = re.compile(r"^[.WB]{24}\|[WB]\|\d+\|\d+$")

EXPECTED_IMPORTED_GAMES = 94_540
EXPECTED_PROCESSED_ROWS = 94_983
EXPECTED_PROCESSED_UNIQUE_GAMES = 94_134
EXPECTED_LATE_IMPORT_GAMES = 406
EXPECTED_LOCALLY_VALID_GAMES = 395

PRIOR_CORPUS_PATHS = (
    "docs/experiments/dev-v4-formal-paired-eval-v1-corpus-draft.json",
    "docs/experiments/dev-v4-formal-paired-eval-v1-oracle-corpus.json",
    "docs/experiments/dev-v4-phase-covered-corpus-v1-start-positions.json",
    "docs/experiments/dev-v4-phase-covered-corpus-v1.json",
    "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json",
    "docs/experiments/sanmill-book-path-corpus-v1.json",
    (
        "docs/experiments/"
        "sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.json"
    ),
    "docs/experiments/sanmill-retained-v3-v4-phase-process-corpus-v1.json",
)


class RetainedLateImportPoolError(RuntimeError):
    """Raised when the source-only held-out pool cannot be reproduced."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(*parts: object) -> str:
    text = "|".join((SELECTION_DOMAIN, *(str(part) for part in parts)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RetainedLateImportPoolError(f"{context} must be an object")
    return value


def _canonical_identity(payload: Mapping[str, Any], field: str) -> str:
    identity = payload.get(field)
    if not isinstance(identity, str):
        raise RetainedLateImportPoolError(f"{field} is absent")
    body = {key: value for key, value in payload.items() if key != field}
    if canonical_sha256(body) != identity:
        raise RetainedLateImportPoolError(f"{field} differs")
    return identity


def _game_id_from_path(path: str) -> str | None:
    match = GAME_FILE_PATTERN.search(path.replace("\\", "/"))
    return match.group(1) if match else None


def _processed_game_ids(human_db_path: Path) -> tuple[int, set[str]]:
    uri = human_db_path.resolve().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = [
            str(row[0])
            for row in connection.execute(
                "SELECT file_path FROM processed_files ORDER BY file_path"
            )
        ]
    finally:
        connection.close()
    game_ids = {
        game_id
        for row in rows
        if (game_id := _game_id_from_path(row)) is not None
    }
    return len(rows), game_ids


def discover_late_imports(
    imported_manifest: Mapping[str, Any],
    processed_game_ids: set[str],
) -> list[str]:
    """Return imported PlayOK IDs that are absent from the active HumanDB."""
    imported_ids = set()
    for game_id, imported_at in imported_manifest.items():
        if not isinstance(game_id, str) or not re.fullmatch(r"ml\d+", game_id):
            raise RetainedLateImportPoolError("imported manifest game ID differs")
        if not isinstance(imported_at, str) or not imported_at:
            raise RetainedLateImportPoolError("imported timestamp is absent")
        imported_ids.add(game_id)
    if processed_game_ids - imported_ids:
        raise RetainedLateImportPoolError(
            "active HumanDB contains normalized IDs outside the import manifest"
        )
    return sorted(imported_ids - processed_game_ids)


def _logical_turn(move: Mapping[str, Any]) -> list[str]:
    actions = [nmm_move_base(move)]
    capture = move.get("capture")
    if capture is not None:
        actions.append(f"x{capture}")
    return actions


def _notation(move: Mapping[str, Any]) -> str:
    base = nmm_move_base(move)
    capture = move.get("capture")
    return base if capture is None else f"{base}x{capture}"


def _matching_raw_move(board: BoardState, raw: Mapping[str, Any]) -> Any:
    expected = {key: raw.get(key) for key in ("from", "to", "capture")}
    matches = [
        move
        for move in get_all_legal_moves(board)
        if all(move.get(key) == value for key, value in expected.items())
    ]
    if len(matches) != 1:
        raise RetainedLateImportPoolError("source move is illegal or ambiguous")
    return matches[0]


def extract_source_candidates(
    raw: bytes,
    *,
    relative_path: str,
    imported_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay one source game and return blind board-before-move candidates."""
    file_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetainedLateImportPoolError("source game is invalid JSON") from exc
    record = _mapping(payload, context="source game")
    moves = record.get("moves")
    if not isinstance(moves, list):
        raise RetainedLateImportPoolError("source moves are not an array")
    session_id = record.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RetainedLateImportPoolError("source session ID is absent")
    if (
        record.get("source") != "playok"
        or record.get("source_type") != "human_vs_human"
    ):
        raise RetainedLateImportPoolError("source provenance differs")

    source = {
        "relative_path": relative_path,
        "byte_length": len(raw),
        "file_sha256": file_sha256,
        "imported_at": imported_at,
        "session_sha256": canonical_sha256(
            {"domain": "playok-session-id-v1", "session_id": session_id}
        ),
        "source": "playok",
        "source_type": "human_vs_human",
    }
    source["source_identity"] = canonical_sha256(source)

    board = BoardState.new_game()
    turns: list[list[str]] = []
    candidates: list[dict[str, Any]] = []
    for index, raw_move in enumerate(moves):
        move_record = _mapping(raw_move, context=f"source move {index}")
        if move_record.get("board_fen_before") != board.to_fen_string():
            raise RetainedLateImportPoolError("source board history differs")
        if move_record.get("color") != board.turn:
            raise RetainedLateImportPoolError("source move colour differs")
        move = _matching_raw_move(board, move_record)
        if move_record.get("notation") != _notation(move):
            raise RetainedLateImportPoolError("source move notation differs")

        phase = _phase(board)
        if index >= MINIMUM_LOGICAL_PLY and phase is not None:
            fen = board.to_fen_string()
            body = {
                "source": dict(source),
                "logical_ply_count": index,
                "logical_turns": [list(turn) for turn in turns],
                "action_history": [token for turn in turns for token in turn],
                "fen": fen,
                "ring16_canonical_fen": ring16_canonical_fen(fen),
                "phase": phase,
                "turn": board.turn,
            }
            candidates.append(
                {
                    **body,
                    "candidate_identity": canonical_sha256(body),
                    "selection_rank": _rank(
                        "state",
                        source["source_identity"],
                        phase,
                        index,
                        fen,
                    ),
                }
            )

        turns.append(_logical_turn(move))
        board = board.apply_move(move)
        if index + 1 < len(moves) and is_terminal(board)[0]:
            raise RetainedLateImportPoolError("source continues after local terminal")
    return source, candidates


def _walk_fens(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_fens(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _walk_fens(item)
    elif isinstance(value, str) and FEN_PATTERN.fullmatch(value):
        yield value


def load_prior_corpus_orbits(
    repository_root: Path,
    relative_paths: Sequence[str] = PRIOR_CORPUS_PATHS,
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    exact: set[str] = set()
    ring16: set[str] = set()
    files = []
    for relative in relative_paths:
        path = repository_root / relative
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise RetainedLateImportPoolError(
                f"cannot read prior corpus {relative}"
            ) from exc
        for fen in _walk_fens(payload):
            exact.add(fen)
            ring16.add(ring16_canonical_fen(fen))
        files.append(
            {
                "path": relative,
                "byte_length": path.stat().st_size,
                "file_sha256": sha256_file(path),
            }
        )
    return exact, ring16, files


def audit_training_exposure(
    candidates: Sequence[Mapping[str, Any]],
    *,
    human_db_path: Path,
    human_db_identity: str,
    specialist_paths: Mapping[str, tuple[Path, str]],
    prior_exact: set[str],
    prior_ring16: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject any state visible in a route database or earlier start corpus."""
    human_report = _probe_human_db(human_db_path, immutable=True)
    if human_report.get("error"):
        raise RetainedLateImportPoolError(str(human_report["error"]))
    if human_report.get("identity") != human_db_identity:
        raise RetainedLateImportPoolError("HumanDB identity differs")
    if human_report.get("malom_columns_policy") != "masked_historical_labels":
        raise RetainedLateImportPoolError("HumanDB label policy differs")

    for candidate_id, (path, identity) in specialist_paths.items():
        report = _probe_specialist_db(path)
        if report.get("error"):
            raise RetainedLateImportPoolError(str(report["error"]))
        if report.get("content_sha256") != identity:
            raise RetainedLateImportPoolError(
                f"{candidate_id} SpecialistDB identity differs"
            )
        if report.get("label_version") != "sector-corrected-v1":
            raise RetainedLateImportPoolError(
                f"{candidate_id} SpecialistDB labels are untrusted"
            )

    human = HumanDB(human_db_path, read_only=True, immutable=True)
    specialists = {
        candidate_id: SpecialistDB(path, read_only=True)
        for candidate_id, (path, _identity) in specialist_paths.items()
    }
    cache: dict[str, tuple[bool, dict[str, bool]]] = {}
    rejection_hits: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    try:
        for candidate in candidates:
            fen = str(candidate["fen"])
            if fen not in cache:
                board = BoardState.from_fen_string(fen)
                cache[fen] = (
                    human.query_position(board) is not None,
                    {
                        candidate_id: db.query_wdl_evidence(
                            board, min_samples=0
                        )
                        is not None
                        for candidate_id, db in specialists.items()
                    },
                )
            human_hit, specialist_hits = cache[fen]
            reasons = []
            if human_hit:
                reasons.append("human_db_d4")
            reasons.extend(
                f"{candidate_id}_specialist_db_d4"
                for candidate_id, hit in specialist_hits.items()
                if hit
            )
            if fen in prior_exact:
                reasons.append("prior_corpus_exact")
            if str(candidate["ring16_canonical_fen"]) in prior_ring16:
                reasons.append("prior_corpus_ring16")
            rejection_hits.update(reasons)
            if not reasons:
                eligible.append(dict(candidate))
    finally:
        human.close()
        for specialist in specialists.values():
            specialist.close()

    phase_games: dict[str, set[str]] = defaultdict(set)
    for candidate in eligible:
        phase_games[str(candidate["phase"])].add(
            str(candidate["source"]["source_identity"])
        )
    summary = {
        "candidate_state_count": len(candidates),
        "unique_candidate_fen_count": len(cache),
        "rejection_hits_nonexclusive": dict(sorted(rejection_hits.items())),
        "eligible_state_count": len(eligible),
        "eligible_source_game_count": len(
            {str(item["source"]["source_identity"]) for item in eligible}
        ),
        "eligible_state_phase_counts": dict(
            sorted(Counter(str(item["phase"]) for item in eligible).items())
        ),
        "eligible_source_game_phase_counts": {
            phase: len(game_ids) for phase, game_ids in sorted(phase_games.items())
        },
        "human_db_identity": human_db_identity,
        "specialist_db_identities": {
            candidate_id: identity
            for candidate_id, (_path, identity) in specialist_paths.items()
        },
        "eligible_records_identity": canonical_sha256(
            [str(item["candidate_identity"]) for item in eligible]
        ),
    }
    return eligible, summary


def _interleave_phase_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    pending = {phase: list(groups.get(phase, ())) for phase in PHASE_ORDER}
    ordered = []
    while any(pending.values()):
        for phase in PHASE_ORDER:
            if pending[phase]:
                ordered.append(dict(pending[phase].pop(0)))
    return ordered


def select_independent_records(
    eligible: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose one ring16-unique state per source and create nested phase order."""
    by_game_phase: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for candidate in eligible:
        game_id = str(candidate["source"]["source_identity"])
        by_game_phase[game_id][str(candidate["phase"])].append(dict(candidate))
    for phases in by_game_phase.values():
        for rows in phases.values():
            rows.sort(key=lambda item: str(item["selection_rank"]))

    flying_games = {
        game_id for game_id, phases in by_game_phase.items() if "flying" in phases
    }
    remaining = sorted(set(by_game_phase) - flying_games)
    placement_only = [
        game_id
        for game_id in remaining
        if set(by_game_phase[game_id]) == {"placement"}
    ]
    movement_only = [
        game_id
        for game_id in remaining
        if set(by_game_phase[game_id]) == {"movement"}
    ]
    dual = [
        game_id
        for game_id in remaining
        if {"placement", "movement"}.issubset(by_game_phase[game_id])
    ]
    unexpected = [
        game_id
        for game_id in remaining
        if game_id not in placement_only
        and game_id not in movement_only
        and game_id not in dual
    ]
    if unexpected:
        raise RetainedLateImportPoolError("eligible phase availability differs")

    target_placement = (len(remaining) + 1) // 2
    dual_for_placement = target_placement - len(placement_only)
    if not 0 <= dual_for_placement <= len(dual):
        raise RetainedLateImportPoolError("balanced phase assignment is infeasible")
    dual.sort(key=lambda game_id: _rank("phase-assignment", game_id))
    assigned: dict[str, str] = {
        **{game_id: "flying" for game_id in flying_games},
        **{game_id: "placement" for game_id in placement_only},
        **{game_id: "movement" for game_id in movement_only},
        **{
            game_id: (
                "placement" if index < dual_for_placement else "movement"
            )
            for index, game_id in enumerate(dual)
        },
    }

    game_groups: dict[str, list[dict[str, Any]]] = {}
    for phase in PHASE_ORDER:
        rows = [
            {
                "source_identity": game_id,
                "phase": phase,
                "order_rank": _rank("source-order", phase, game_id),
            }
            for game_id, assigned_phase in assigned.items()
            if assigned_phase == phase
        ]
        rows.sort(key=lambda item: str(item["order_rank"]))
        game_groups[phase] = rows
    source_order = _interleave_phase_groups(game_groups)

    selected = []
    ring16_seen = set()
    ring16_excluded_sources = []
    for source in source_order:
        game_id = str(source["source_identity"])
        phase = str(source["phase"])
        choice = next(
            (
                row
                for row in by_game_phase[game_id][phase]
                if row["ring16_canonical_fen"] not in ring16_seen
            ),
            None,
        )
        if choice is None:
            ring16_excluded_sources.append(game_id)
            continue
        ring16_seen.add(str(choice["ring16_canonical_fen"]))
        selected.append(choice)

    summary = {
        "eligible_source_count": len(by_game_phase),
        "flying_sources_reserved_for_flying": len(flying_games),
        "balanced_nonflying_assignment": {
            "placement": sum(value == "placement" for value in assigned.values()),
            "movement": sum(value == "movement" for value in assigned.values()),
        },
        "ring16_collision_excluded_source_count": len(ring16_excluded_sources),
        "ring16_collision_excluded_source_identity": canonical_sha256(
            ring16_excluded_sources
        ),
        "pre_strict_selected_count": len(selected),
        "pre_strict_phase_counts": dict(
            sorted(Counter(str(item["phase"]) for item in selected).items())
        ),
    }
    return selected, summary


def _matching_actions(board: BoardState, actions: Sequence[str]) -> Any:
    expected = tuple(str(action) for action in actions)
    matches = [
        move
        for move in get_all_legal_moves(board)
        if nmm_move_actions(move) == expected
    ]
    if len(matches) != 1:
        raise RetainedLateImportPoolError(
            f"source history selects {len(matches)} legal moves"
        )
    return matches[0]


def _strict_replay_observation(
    record: Mapping[str, Any],
    installation: Any,
) -> dict[str, Any]:
    board = BoardState.new_game()
    turns = record["logical_turns"]
    with SanmillTrainingGame(installation, seed=42) as game:
        for logical_ply, actions in enumerate(turns, 1):
            if game.state.terminal:
                return {
                    "candidate_identity": record["candidate_identity"],
                    "accepted": False,
                    "disposition": "strict_terminal_before_source_start",
                    "completed_logical_plies": logical_ply - 1,
                    "target_logical_plies": len(turns),
                    "outcome_reason": str(game.state.outcome_reason),
                    "history_sha256": str(game.state.history_sha256),
                }
            move = _matching_actions(board, actions)
            applied = game.apply_nmm_move(board, move)
            board = board.apply_move(move)
            if applied.state.logical_ply_count != logical_ply:
                raise RetainedLateImportPoolError(
                    "strict replay logical-ply count differs"
                )
        game.assert_current_board(board)
        if board.to_fen_string() != record.get("fen"):
            raise RetainedLateImportPoolError("strict replay final FEN differs")
        if game.state.terminal:
            return {
                "candidate_identity": record["candidate_identity"],
                "accepted": False,
                "disposition": "strict_terminal_at_source_start",
                "completed_logical_plies": len(turns),
                "target_logical_plies": len(turns),
                "outcome_reason": str(game.state.outcome_reason),
                "history_sha256": str(game.state.history_sha256),
            }
        return {
            "candidate_identity": record["candidate_identity"],
            "accepted": True,
            "disposition": "strict_nonterminal_source_start",
            "history_sha256": str(game.state.history_sha256),
            "sanmill_fen": str(game.state.fen),
            "logical_ply_count": int(game.state.logical_ply_count),
            "action_token_count": int(game.state.action_token_count),
            "logical_plies_by_side": list(game.state.logical_plies_by_side),
            "no_capture_count": int(game.state.no_capture_count),
            "repetition_current_count": int(game.state.repetition_current_count),
        }


def strict_replay_audit(
    records: Sequence[Mapping[str, Any]],
    installation: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay every selected source twice in fresh strict-referee processes."""
    passes = [
        [_strict_replay_observation(record, installation) for record in records]
        for _ in range(2)
    ]
    if passes[0] != passes[1]:
        raise RetainedLateImportPoolError("fresh strict replay passes differ")
    observations = passes[0]
    accepted = {
        str(item["candidate_identity"]): item
        for item in observations
        if item["accepted"] is True
    }
    accepted_records = [
        dict(record)
        for record in records
        if str(record["candidate_identity"]) in accepted
    ]
    body = {
        "runtime": training_installation_record(installation, seed=42),
        "repeat_passes": 2,
        "fresh_process_count": len(records) * 2,
        "repeat_passes_byte_equal": True,
        "selected_count": len(records),
        "accepted_count": len(accepted_records),
        "excluded_count": len(records) - len(accepted_records),
        "excluded_reason_counts": dict(
            sorted(
                Counter(
                    str(item["outcome_reason"])
                    for item in observations
                    if item["accepted"] is False
                ).items()
            )
        ),
        "observations": observations,
        "observations_identity": canonical_sha256(observations),
    }
    return accepted_records, {**body, "audit_identity": canonical_sha256(body)}


def _finalize_records(
    records: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    strict = {
        str(item["candidate_identity"]): item
        for item in observations
        if item["accepted"] is True
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for phase in PHASE_ORDER:
        rows = [dict(record) for record in records if record["phase"] == phase]
        rows.sort(
            key=lambda item: _rank(
                "final-order", phase, item["source"]["source_identity"]
            )
        )
        groups[phase] = rows
    ordered = _interleave_phase_groups(groups)
    final = []
    for index, raw in enumerate(ordered, 1):
        observation = strict[str(raw["candidate_identity"])]
        body = {
            "start_id": f"late-import-heldout-{index:03d}",
            **{
                key: value
                for key, value in raw.items()
                if key not in {"selection_rank", "candidate_identity"}
            },
            "source_candidate_identity": raw["candidate_identity"],
            "strict_start": {
                key: observation[key]
                for key in (
                    "history_sha256",
                    "sanmill_fen",
                    "logical_ply_count",
                    "action_token_count",
                    "logical_plies_by_side",
                    "no_capture_count",
                    "repetition_current_count",
                )
            },
            "training_db_d4_exposed": False,
            "prior_corpus_exact_overlap": False,
            "prior_corpus_ring16_overlap": False,
        }
        final.append({**body, "record_identity": canonical_sha256(body)})
    return final


def _prefix_profiles(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    profiles = []
    for starts in (64, 142, 253, 568):
        available = len(records) >= starts
        subset = records[:starts] if available else records
        profiles.append(
            {
                "target_starts": starts,
                "target_games": starts * 4,
                "available": available,
                "available_starts": len(subset),
                "phase_counts": dict(
                    sorted(Counter(str(row["phase"]) for row in subset).items())
                ),
                "records_identity": canonical_sha256(
                    [str(row["record_identity"]) for row in subset]
                ),
            }
        )
    return profiles


def build_retained_late_import_pool(
    *,
    repository_root: Path,
    human_games_root: Path,
    imported_manifest_path: Path,
    human_db_path: Path,
    human_db_identity: str,
    specialist_paths: Mapping[str, tuple[Path, str]],
    installation: Any,
    prior_corpus_paths: Sequence[str] = PRIOR_CORPUS_PATHS,
) -> dict[str, Any]:
    """Build the frozen source pool without loading either candidate policy."""
    imported_raw = imported_manifest_path.read_bytes()
    imported = _mapping(json.loads(imported_raw), context="imported manifest")
    processed_rows, processed_ids = _processed_game_ids(human_db_path)
    late_ids = discover_late_imports(imported, processed_ids)
    if len(imported) != EXPECTED_IMPORTED_GAMES:
        raise RetainedLateImportPoolError("imported game count differs")
    if processed_rows != EXPECTED_PROCESSED_ROWS:
        raise RetainedLateImportPoolError("processed-file row count differs")
    if len(processed_ids) != EXPECTED_PROCESSED_UNIQUE_GAMES:
        raise RetainedLateImportPoolError("processed unique game count differs")
    if len(late_ids) != EXPECTED_LATE_IMPORT_GAMES:
        raise RetainedLateImportPoolError("late import count differs")

    source_manifest = []
    all_candidates = []
    invalid_sources = []
    valid_source_count = 0
    for game_id in late_ids:
        relative_path = f"data/human_games/human_{game_id}.jsonl"
        path = human_games_root / f"human_{game_id}.jsonl"
        try:
            raw = path.read_bytes()
            source, candidates = extract_source_candidates(
                raw,
                relative_path=relative_path,
                imported_at=str(imported[game_id]),
            )
        except (OSError, RetainedLateImportPoolError) as exc:
            raw = path.read_bytes() if path.is_file() else b""
            invalid_sources.append(
                {
                    "relative_path": relative_path,
                    "byte_length": len(raw),
                    "file_sha256": hashlib.sha256(raw).hexdigest(),
                    "reason": str(exc),
                }
            )
            continue
        valid_source_count += 1
        source_manifest.append(source)
        all_candidates.extend(candidates)
    if valid_source_count != EXPECTED_LOCALLY_VALID_GAMES:
        raise RetainedLateImportPoolError("locally valid source count differs")

    prior_exact, prior_ring16, prior_files = load_prior_corpus_orbits(
        repository_root, prior_corpus_paths
    )
    eligible, exposure = audit_training_exposure(
        all_candidates,
        human_db_path=human_db_path,
        human_db_identity=human_db_identity,
        specialist_paths=specialist_paths,
        prior_exact=prior_exact,
        prior_ring16=prior_ring16,
    )
    selected, selection = select_independent_records(eligible)
    accepted, strict_audit = strict_replay_audit(selected, installation)
    records = _finalize_records(accepted, strict_audit["observations"])

    phase_counts = dict(
        sorted(Counter(str(record["phase"]) for record in records).items())
    )
    source_audit = {
        "imported_manifest": {
            "path": "data/human_games/imported.json",
            "byte_length": len(imported_raw),
            "file_sha256": hashlib.sha256(imported_raw).hexdigest(),
            "entry_count": len(imported),
        },
        "active_human_db": {
            "identity": human_db_identity,
            "processed_file_rows": processed_rows,
            "normalized_unique_game_ids": len(processed_ids),
        },
        "late_import_count": len(late_ids),
        "late_import_timestamp_range": [
            min(str(imported[game_id]) for game_id in late_ids),
            max(str(imported[game_id]) for game_id in late_ids),
        ],
        "late_import_source_set_identity": canonical_sha256(
            [
                {
                    "relative_path": f"data/human_games/human_{game_id}.jsonl",
                    "imported_at": str(imported[game_id]),
                }
                for game_id in late_ids
            ]
        ),
        "locally_valid_source_count": valid_source_count,
        "invalid_source_count": len(invalid_sources),
        "invalid_sources": invalid_sources,
        "valid_source_manifest_identity": canonical_sha256(source_manifest),
    }
    body: dict[str, Any] = {
        "schema_version": POOL_SCHEMA,
        "pool_id": POOL_ID,
        "status": POOL_STATUS,
        "source_audit": source_audit,
        "prior_corpus_audit": {
            "files": prior_files,
            "files_identity": canonical_sha256(prior_files),
            "exact_fen_count": len(prior_exact),
            "ring16_orbit_count": len(prior_ring16),
        },
        "exposure_audit": exposure,
        "selection_contract": {
            "candidate_policy_loaded": False,
            "candidate_outcome_rows_read": 0,
            "human_source_outcomes_used": False,
            "minimum_logical_ply": MINIMUM_LOGICAL_PLY,
            "one_start_per_source_game": True,
            "unique_ring16_orbit_per_start": True,
            "phase_assignment": (
                "reserve every flying-capable source for flying; balance the "
                "remaining sources between placement and movement; choose the "
                "SHA-256-ranked unexposed state and interleave phases"
            ),
            **selection,
            "strict_accepted_count": len(records),
            "strict_phase_counts": phase_counts,
        },
        "strict_replay_audit": strict_audit,
        "records": records,
        "records_identity": canonical_sha256(records),
        "nested_precision_prefixes": _prefix_profiles(records),
        "claim_boundaries": {
            "candidate_games_played": 0,
            "candidate_policy_loaded": False,
            "training_or_update": False,
            "heldout_evaluation_completed": False,
            "playing_strength_claim": False,
            "equivalence_claim": False,
            "refresh_causal_claim": False,
            "promotion": False,
            "publication": False,
            "release": False,
            "launch_authorized": False,
        },
    }
    payload = {**body, "pool_identity": canonical_sha256(body)}
    validate_retained_late_import_pool(payload)
    return payload


def validate_retained_late_import_pool(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate portable identities, independence, and every local history."""
    if payload.get("schema_version") != POOL_SCHEMA:
        raise RetainedLateImportPoolError("pool schema differs")
    if payload.get("pool_id") != POOL_ID:
        raise RetainedLateImportPoolError("pool id differs")
    if payload.get("status") != POOL_STATUS:
        raise RetainedLateImportPoolError("pool status differs")
    _canonical_identity(payload, "pool_identity")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RetainedLateImportPoolError("pool records are absent")
    if payload.get("records_identity") != canonical_sha256(records):
        raise RetainedLateImportPoolError("pool records identity differs")

    source_ids = set()
    exact_fens = set()
    ring16_fens = set()
    phases: Counter[str] = Counter()
    for index, raw in enumerate(records, 1):
        record = _mapping(raw, context="pool record")
        _canonical_identity(record, "record_identity")
        if record.get("start_id") != f"late-import-heldout-{index:03d}":
            raise RetainedLateImportPoolError("pool start order differs")
        source = _mapping(record.get("source"), context="record source")
        _canonical_identity(source, "source_identity")
        source_id = str(source.get("source_identity", ""))
        if not source_id or source_id in source_ids:
            raise RetainedLateImportPoolError("source game is duplicated")
        source_ids.add(source_id)
        fen = str(record.get("fen", ""))
        ring16 = str(record.get("ring16_canonical_fen", ""))
        if fen in exact_fens or ring16 in ring16_fens:
            raise RetainedLateImportPoolError("pool board orbit is duplicated")
        exact_fens.add(fen)
        ring16_fens.add(ring16)
        if ring16_canonical_fen(fen) != ring16:
            raise RetainedLateImportPoolError("pool ring16 value differs")
        if any(
            record.get(field) is not False
            for field in (
                "training_db_d4_exposed",
                "prior_corpus_exact_overlap",
                "prior_corpus_ring16_overlap",
            )
        ):
            raise RetainedLateImportPoolError("pool exposure flag differs")
        turns = record.get("logical_turns")
        tokens = record.get("action_history")
        if not isinstance(turns, list) or not isinstance(tokens, list):
            raise RetainedLateImportPoolError("pool history is incomplete")
        if [token for turn in turns for token in turn] != tokens:
            raise RetainedLateImportPoolError("pool history does not flatten")
        board = BoardState.new_game()
        for actions in turns:
            board = board.apply_move(_matching_actions(board, actions))
        if board.to_fen_string() != fen:
            raise RetainedLateImportPoolError("pool local replay differs")
        if record.get("logical_ply_count") != len(turns):
            raise RetainedLateImportPoolError("pool logical-ply count differs")
        if record.get("phase") != _phase(board):
            raise RetainedLateImportPoolError("pool phase differs")
        candidate_body = {
            key: record[key]
            for key in (
                "source",
                "logical_ply_count",
                "logical_turns",
                "action_history",
                "fen",
                "ring16_canonical_fen",
                "phase",
                "turn",
            )
        }
        if canonical_sha256(candidate_body) != record.get(
            "source_candidate_identity"
        ):
            raise RetainedLateImportPoolError("source candidate identity differs")
        strict = _mapping(record.get("strict_start"), context="strict start")
        if strict.get("logical_ply_count") != len(turns):
            raise RetainedLateImportPoolError("strict start ply count differs")
        phases[str(record["phase"])] += 1

    selection = _mapping(payload.get("selection_contract"), context="selection")
    if selection.get("candidate_policy_loaded") is not False:
        raise RetainedLateImportPoolError("selection loaded a candidate")
    if selection.get("candidate_outcome_rows_read") != 0:
        raise RetainedLateImportPoolError("selection read candidate outcomes")
    if selection.get("human_source_outcomes_used") is not False:
        raise RetainedLateImportPoolError("selection used human outcomes")
    if selection.get("strict_accepted_count") != len(records):
        raise RetainedLateImportPoolError("strict accepted count differs")
    if selection.get("strict_phase_counts") != dict(sorted(phases.items())):
        raise RetainedLateImportPoolError("strict phase counts differ")

    exposure = _mapping(payload.get("exposure_audit"), context="exposure audit")
    if exposure.get("eligible_source_game_count", 0) < len(records):
        raise RetainedLateImportPoolError("exposure source count differs")
    boundaries = _mapping(
        payload.get("claim_boundaries"), context="claim boundaries"
    )
    if boundaries != {
        "candidate_games_played": 0,
        "candidate_policy_loaded": False,
        "training_or_update": False,
        "heldout_evaluation_completed": False,
        "playing_strength_claim": False,
        "equivalence_claim": False,
        "refresh_causal_claim": False,
        "promotion": False,
        "publication": False,
        "release": False,
        "launch_authorized": False,
    }:
        raise RetainedLateImportPoolError("claim boundary differs")

    expected_profiles = _prefix_profiles(records)
    if payload.get("nested_precision_prefixes") != expected_profiles:
        raise RetainedLateImportPoolError("precision prefix profiles differ")
    return [dict(record) for record in records]

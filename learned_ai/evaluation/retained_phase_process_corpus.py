"""Candidate-blind phase-history corpus for retained-route process confirmation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai.human_db import HumanDB
from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.evaluation.heldout_exposure import validate_executable_corpus
from learned_ai.evaluation.phase_corpus import validate_phase_corpus
from learned_ai.evaluation.phase_replay_development_corpus import (
    _fixture_history,
    validate_phase_replay_development_corpus,
)
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


CORPUS_SCHEMA = "nmm.retained-phase-process-corpus.v1"
CORPUS_ID = "sanmill-retained-v3-v4-phase-process-corpus-v1"
CORPUS_STATUS = "frozen_source_only_awaiting_evaluation_plan_and_authorization"
PASSIVITY_PLAN_IDENTITY = (
    "035c68f80b94dddb8d139d56c38c86c4fde29fa13de5e19db1f4e1fe484c318e"
)
PHASE_CORPUS_IDENTITY = (
    "cc5477b777cf38ea59c5f20af4bb7b6019b0d12b328ac139b2a8e7d9f2df0bc8"
)
PRIOR_REPLAY_CORPUS_IDENTITY = (
    "ca4b410dd2913933d3ecbd8672fe274ea4a2f8ad42db3f039dabfa52af196aa4"
)
OPENING_CORPUS_IDENTITY = (
    "417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9"
)
EXPECTED_ELIGIBLE_COUNT = 42
EXPECTED_ACCEPTED_COUNT = 39
EXPECTED_PHASE_COUNTS = {"flying": 7, "movement": 14, "placement": 18}
EXPECTED_STRICT_EXCLUSIONS = (29, 31, 32)


class RetainedPhaseProcessCorpusError(RuntimeError):
    """Raised when the source-only confirmation corpus is not reproducible."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RetainedPhaseProcessCorpusError(f"{context} must be an object")
    return value


def _source(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = entry.get("sources")
    if (
        not isinstance(sources, Sequence)
        or isinstance(sources, (str, bytes))
        or len(sources) != 1
        or not isinstance(sources[0], Mapping)
    ):
        raise RetainedPhaseProcessCorpusError("phase entry source is incomplete")
    return sources[0]


def _canonical_identity(payload: Mapping[str, Any], field: str) -> str:
    identity = payload.get(field)
    if not isinstance(identity, str):
        raise RetainedPhaseProcessCorpusError(f"{field} is absent")
    body = {key: value for key, value in payload.items() if key != field}
    if canonical_sha256(body) != identity:
        raise RetainedPhaseProcessCorpusError(f"{field} differs")
    return identity


def select_candidate_blind_entries(
    phase_corpus: Mapping[str, Any],
    prior_replay_corpus: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select every identity-replayable entry not used by the prior probe."""
    validate_phase_corpus(phase_corpus)
    validate_phase_replay_development_corpus(
        prior_replay_corpus,
        phase_corpus=phase_corpus,
    )
    if phase_corpus.get("corpus_identity") != PHASE_CORPUS_IDENTITY:
        raise RetainedPhaseProcessCorpusError("phase corpus identity differs")
    if prior_replay_corpus.get("corpus_identity") != PRIOR_REPLAY_CORPUS_IDENTITY:
        raise RetainedPhaseProcessCorpusError("prior replay corpus identity differs")
    used = {
        int(record["source_entry_index"])
        for record in prior_replay_corpus["records"]
    }
    selected = [
        dict(entry)
        for entry in phase_corpus["entries"]
        if int(entry["index"]) not in used
        and _source(entry).get("color_transform") == "identity"
    ]
    if len(selected) != EXPECTED_ELIGIBLE_COUNT:
        raise RetainedPhaseProcessCorpusError("eligible phase-history count differs")
    return selected


def _matching_move(board: BoardState, actions: Sequence[str]) -> Any:
    expected = tuple(str(action) for action in actions)
    matches = [
        move
        for move in get_all_legal_moves(board)
        if nmm_move_actions(move) == expected
    ]
    if len(matches) != 1:
        raise RetainedPhaseProcessCorpusError(
            f"phase history selects {len(matches)} legal moves"
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
                    "source_entry_index": record["source_entry_index"],
                    "record_identity": record["record_identity"],
                    "accepted": False,
                    "disposition": "strict_terminal_before_source_start",
                    "completed_logical_plies": logical_ply - 1,
                    "target_logical_plies": len(turns),
                    "outcome_reason": str(game.state.outcome_reason),
                    "history_sha256": str(game.state.history_sha256),
                }
            move = _matching_move(board, actions)
            applied = game.apply_nmm_move(board, move)
            board = board.apply_move(move)
            if applied.state.logical_ply_count != logical_ply:
                raise RetainedPhaseProcessCorpusError(
                    "strict replay logical-ply count differs"
                )

        game.assert_current_board(board)
        if board.to_fen_string() != record.get("fen"):
            raise RetainedPhaseProcessCorpusError("strict replay final FEN differs")
        if game.state.terminal:
            return {
                "source_entry_index": record["source_entry_index"],
                "record_identity": record["record_identity"],
                "accepted": False,
                "disposition": "strict_terminal_at_source_start",
                "completed_logical_plies": len(turns),
                "target_logical_plies": len(turns),
                "outcome_reason": str(game.state.outcome_reason),
                "history_sha256": str(game.state.history_sha256),
            }
        return {
            "source_entry_index": record["source_entry_index"],
            "record_identity": record["record_identity"],
            "accepted": True,
            "disposition": "strict_nonterminal_source_start",
            "logical_ply_count": int(game.state.logical_ply_count),
            "action_token_count": int(game.state.action_token_count),
            "logical_plies_by_side": list(game.state.logical_plies_by_side),
            "no_capture_count": int(game.state.no_capture_count),
            "repetition_current_count": int(game.state.repetition_current_count),
            "history_sha256": str(game.state.history_sha256),
            "sanmill_fen": str(game.state.fen),
        }


def _candidate_records(
    entries: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for entry in entries:
        tokens, turns = _fixture_history(fixture, entry)
        body = {
            "source_entry_index": int(entry["index"]),
            "fen": str(entry["fen"]),
            "ring16_canonical_fen": str(entry["ring16_canonical_fen"]),
            "phase": str(entry["phase"]),
            "turn": str(entry["turn"]),
            "malom_wdl_for_side_to_move": str(
                entry["malom_wdl_for_side_to_move"]
            ),
            "source": dict(_source(entry)),
            "action_history": tokens,
            "logical_turns": turns,
            "action_token_count": len(tokens),
            "logical_ply_count": len(turns),
        }
        records.append({**body, "record_identity": canonical_sha256(body)})
    return records


def _strict_replay_audit(
    records: Sequence[Mapping[str, Any]],
    installation: Any,
) -> dict[str, Any]:
    passes = [
        [_strict_replay_observation(record, installation) for record in records]
        for _ in range(2)
    ]
    if passes[0] != passes[1]:
        raise RetainedPhaseProcessCorpusError("fresh strict replay passes differ")
    observations = passes[0]
    excluded = tuple(
        int(item["source_entry_index"])
        for item in observations
        if item["accepted"] is False
    )
    if excluded != EXPECTED_STRICT_EXCLUSIONS:
        raise RetainedPhaseProcessCorpusError("strict replay exclusion set differs")
    body = {
        "runtime": training_installation_record(installation, seed=42),
        "repeat_passes": 2,
        "fresh_process_count": len(records) * 2,
        "repeat_passes_byte_equal": True,
        "observations": observations,
        "observations_identity": canonical_sha256(observations),
    }
    return {**body, "audit_identity": canonical_sha256(body)}


def _db_exposure(
    records: Sequence[Mapping[str, Any]],
    *,
    human_db_path: Path,
    human_db_identity: str,
    specialist_paths: Mapping[str, tuple[Path, str]],
) -> dict[str, Any]:
    human_report = _probe_human_db(human_db_path)
    if human_report.get("error"):
        raise RetainedPhaseProcessCorpusError(str(human_report["error"]))
    if human_report.get("identity") != human_db_identity:
        raise RetainedPhaseProcessCorpusError("HumanDB identity differs")
    if human_report.get("malom_columns_policy") != "masked_historical_labels":
        raise RetainedPhaseProcessCorpusError("HumanDB label policy differs")

    for candidate_id, (path, identity) in specialist_paths.items():
        report = _probe_specialist_db(path)
        if report.get("error"):
            raise RetainedPhaseProcessCorpusError(str(report["error"]))
        if report.get("content_sha256") != identity:
            raise RetainedPhaseProcessCorpusError(
                f"{candidate_id} SpecialistDB identity differs"
            )
        if report.get("label_version") != "sector-corrected-v1":
            raise RetainedPhaseProcessCorpusError(
                f"{candidate_id} SpecialistDB labels are untrusted"
            )

    human = HumanDB(human_db_path, read_only=True)
    specialists = {
        candidate_id: SpecialistDB(path, read_only=True)
        for candidate_id, (path, _identity) in specialist_paths.items()
    }
    try:
        rows = []
        for record in records:
            board = BoardState.from_fen_string(str(record["fen"]))
            human_item = human.query_position(board)
            specialist_items = {
                candidate_id: db.query_wdl_evidence(board, min_samples=0)
                for candidate_id, db in specialists.items()
            }
            rows.append(
                {
                    "start_id": record["start_id"],
                    "source_entry_index": record["source_entry_index"],
                    "human_db_d4_exposed": human_item is not None,
                    "human_db_games": (
                        int(human_item.total_games) if human_item is not None else 0
                    ),
                    "specialist_db_d4_exposed": {
                        candidate_id: item is not None
                        for candidate_id, item in specialist_items.items()
                    },
                    "specialist_db_empirical_samples": {
                        candidate_id: (
                            sum(int(value) for value in item.empirical_counts)
                            if item is not None
                            else 0
                        )
                        for candidate_id, item in specialist_items.items()
                    },
                    "specialist_db_has_theoretical_label": {
                        candidate_id: bool(
                            item is not None and item.theoretical_wdl is not None
                        )
                        for candidate_id, item in specialist_items.items()
                    },
                }
            )
    finally:
        for db in specialists.values():
            db.close()
        human.close()

    exposed = [
        row
        for row in rows
        if row["human_db_d4_exposed"]
        or any(row["specialist_db_d4_exposed"].values())
    ]
    if exposed:
        raise RetainedPhaseProcessCorpusError(
            "candidate-blind phase corpus has trainer-visible D4 exposure"
        )
    body = {
        "human_db_identity": human_db_identity,
        "specialist_db_identities": {
            candidate_id: identity
            for candidate_id, (_path, identity) in specialist_paths.items()
        },
        "records": rows,
        "records_identity": canonical_sha256(rows),
        "summary": {
            "record_count": len(rows),
            "human_db_d4_exposed_count": 0,
            "specialist_db_d4_exposed_count": {
                candidate_id: 0 for candidate_id in specialist_paths
            },
            "strict_independence_count": len(rows),
        },
    }
    return {**body, "audit_identity": canonical_sha256(body)}


def build_retained_phase_process_corpus(
    *,
    phase_corpus: Mapping[str, Any],
    phase_corpus_file_sha256: str,
    prior_replay_corpus: Mapping[str, Any],
    prior_replay_file_sha256: str,
    fixture: Mapping[str, Any],
    fixture_source: Mapping[str, Any],
    opening_corpus: Mapping[str, Any],
    opening_corpus_file_sha256: str,
    passivity_plan: Mapping[str, Any],
    passivity_plan_file_sha256: str,
    installation: Any,
    human_db_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Build the frozen 39-start source artifact without loading a candidate."""
    if passivity_plan.get("plan_identity") != PASSIVITY_PLAN_IDENTITY:
        raise RetainedPhaseProcessCorpusError("passivity plan identity differs")
    _canonical_identity(passivity_plan, "plan_identity")
    if passivity_plan.get("corpus", {}).get("identity") != OPENING_CORPUS_IDENTITY:
        raise RetainedPhaseProcessCorpusError("prior opening corpus binding differs")
    opening_records = validate_executable_corpus(
        opening_corpus,
        expected_corpus_identity=OPENING_CORPUS_IDENTITY,
        expected_records_identity=str(passivity_plan["corpus"]["records_identity"]),
    )
    old_fens = {
        str(record["execution_record"]["final"]["nmm_fen"])
        for record in opening_records
    }
    old_orbits = {
        str(record["execution_record"]["final"]["ring16_canonical_fen"])
        for record in opening_records
    }

    eligible_entries = select_candidate_blind_entries(
        phase_corpus,
        prior_replay_corpus,
    )
    eligible_records = _candidate_records(eligible_entries, fixture)
    strict_audit = _strict_replay_audit(eligible_records, installation)
    accepted_observations = {
        int(item["source_entry_index"]): item
        for item in strict_audit["observations"]
        if item["accepted"] is True
    }
    accepted_raw = [
        record
        for record in eligible_records
        if int(record["source_entry_index"]) in accepted_observations
    ]
    if len(accepted_raw) != EXPECTED_ACCEPTED_COUNT:
        raise RetainedPhaseProcessCorpusError("accepted strict-replay count differs")

    records = []
    for record_index, raw in enumerate(accepted_raw, 1):
        observation = accepted_observations[int(raw["source_entry_index"])]
        if raw["fen"] in old_fens or raw["ring16_canonical_fen"] in old_orbits:
            raise RetainedPhaseProcessCorpusError(
                "phase start overlaps the prior opening corpus"
            )
        body = {
            "start_id": f"phase-process-{record_index:03d}",
            **{
                key: value
                for key, value in raw.items()
                if key != "record_identity"
            },
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
            "prior_opening_exact_overlap": False,
            "prior_opening_ring16_overlap": False,
        }
        records.append({**body, "record_identity": canonical_sha256(body)})

    candidates = passivity_plan.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise RetainedPhaseProcessCorpusError("passivity candidates differ")
    specialist_paths = {
        str(candidate["candidate_id"]): (
            (repository_root / str(candidate["specialist_db"]["path"])).resolve(),
            str(candidate["specialist_db"]["file_sha256"]),
        )
        for candidate in candidates
    }
    human_identity = str(passivity_plan["data"]["human_db_identity"])
    exposure = _db_exposure(
        records,
        human_db_path=human_db_path,
        human_db_identity=human_identity,
        specialist_paths=specialist_paths,
    )

    phase_counts = dict(sorted(Counter(record["phase"] for record in records).items()))
    if phase_counts != EXPECTED_PHASE_COUNTS:
        raise RetainedPhaseProcessCorpusError("accepted phase counts differ")
    prior_indices = sorted(
        int(record["source_entry_index"])
        for record in prior_replay_corpus["records"]
    )
    body: dict[str, Any] = {
        "schema_version": CORPUS_SCHEMA,
        "corpus_id": CORPUS_ID,
        "status": CORPUS_STATUS,
        "source_identities": {
            "phase_corpus": {
                "identity": PHASE_CORPUS_IDENTITY,
                "file_sha256": phase_corpus_file_sha256,
            },
            "prior_phase_replay_corpus": {
                "identity": PRIOR_REPLAY_CORPUS_IDENTITY,
                "file_sha256": prior_replay_file_sha256,
            },
            "prior_opening_corpus": {
                "identity": OPENING_CORPUS_IDENTITY,
                "file_sha256": opening_corpus_file_sha256,
            },
            "completed_passivity_plan": {
                "identity": PASSIVITY_PLAN_IDENTITY,
                "file_sha256": passivity_plan_file_sha256,
            },
            "fixture": dict(fixture_source),
        },
        "selection_contract": {
            "candidate_loaded": False,
            "candidate_outcome_rows_read": 0,
            "algorithm": (
                "all identity-history phase entries excluding the frozen 12-start "
                "development replay; require deterministic current strict-referee "
                "nonterminal replay and zero D4 exposure to HumanDB and both "
                "candidate-owned SpecialistDBs"
            ),
            "prior_development_source_entry_indices": prior_indices,
            "eligible_identity_history_count": len(eligible_records),
            "strict_replay_excluded_source_entry_indices": list(
                EXPECTED_STRICT_EXCLUSIONS
            ),
            "accepted_count": len(records),
            "phase_counts": phase_counts,
        },
        "strict_replay_audit": strict_audit,
        "exposure_audit": exposure,
        "records": records,
        "records_identity": canonical_sha256(records),
        "claim_boundaries": {
            "games_played": 0,
            "candidate_loaded": False,
            "training_or_update": False,
            "playing_strength_claim": False,
            "refresh_causal_claim": False,
            "promotion_or_publication": False,
            "process_generalization_launch_authorized": False,
        },
    }
    payload = {**body, "corpus_identity": canonical_sha256(body)}
    validate_retained_phase_process_corpus(payload)
    return payload


def validate_retained_phase_process_corpus(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the portable corpus, source audits, and legal local replays."""
    if payload.get("schema_version") != CORPUS_SCHEMA:
        raise RetainedPhaseProcessCorpusError("corpus schema differs")
    if payload.get("corpus_id") != CORPUS_ID:
        raise RetainedPhaseProcessCorpusError("corpus id differs")
    if payload.get("status") != CORPUS_STATUS:
        raise RetainedPhaseProcessCorpusError("corpus status differs")
    identity = _canonical_identity(payload, "corpus_identity")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_ACCEPTED_COUNT:
        raise RetainedPhaseProcessCorpusError("corpus record count differs")
    if payload.get("records_identity") != canonical_sha256(records):
        raise RetainedPhaseProcessCorpusError("corpus record identity differs")
    phase_counts: Counter[str] = Counter()
    start_ids = set()
    record_ids = set()
    source_indices = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, Mapping):
            raise RetainedPhaseProcessCorpusError("corpus record is not an object")
        record_identity = _canonical_identity(record, "record_identity")
        if record.get("start_id") != f"phase-process-{index:03d}":
            raise RetainedPhaseProcessCorpusError("corpus start order differs")
        if record["start_id"] in start_ids or record_identity in record_ids:
            raise RetainedPhaseProcessCorpusError("corpus record is duplicated")
        start_ids.add(str(record["start_id"]))
        record_ids.add(record_identity)
        source_index = int(record["source_entry_index"])
        if source_index in source_indices:
            raise RetainedPhaseProcessCorpusError("source entry is duplicated")
        source_indices.add(source_index)
        if source_index in EXPECTED_STRICT_EXCLUSIONS:
            raise RetainedPhaseProcessCorpusError("excluded source entry is present")
        if record.get("prior_opening_exact_overlap") is not False or record.get(
            "prior_opening_ring16_overlap"
        ) is not False:
            raise RetainedPhaseProcessCorpusError("prior opening overlap differs")
        turns = record.get("logical_turns")
        tokens = record.get("action_history")
        if not isinstance(turns, list) or not isinstance(tokens, list):
            raise RetainedPhaseProcessCorpusError("record history is incomplete")
        if [token for turn in turns for token in turn] != tokens:
            raise RetainedPhaseProcessCorpusError("record turns do not flatten")
        board = BoardState.new_game()
        for actions in turns:
            board = board.apply_move(_matching_move(board, actions))
        if board.to_fen_string() != record.get("fen"):
            raise RetainedPhaseProcessCorpusError("record local replay differs")
        strict_start = _mapping(record.get("strict_start"), context="strict start")
        if strict_start.get("logical_ply_count") != len(turns):
            raise RetainedPhaseProcessCorpusError("strict start ply count differs")
        phase_counts[str(record["phase"])] += 1
    if dict(sorted(phase_counts.items())) != EXPECTED_PHASE_COUNTS:
        raise RetainedPhaseProcessCorpusError("corpus phase counts differ")

    selection = _mapping(payload.get("selection_contract"), context="selection")
    if (
        selection.get("candidate_loaded") is not False
        or selection.get("candidate_outcome_rows_read") != 0
        or selection.get("eligible_identity_history_count") != EXPECTED_ELIGIBLE_COUNT
        or selection.get("accepted_count") != EXPECTED_ACCEPTED_COUNT
        or tuple(selection.get("strict_replay_excluded_source_entry_indices", []))
        != EXPECTED_STRICT_EXCLUSIONS
    ):
        raise RetainedPhaseProcessCorpusError("selection contract differs")
    exposure = _mapping(payload.get("exposure_audit"), context="exposure audit")
    _canonical_identity(exposure, "audit_identity")
    if exposure.get("summary", {}).get("strict_independence_count") != len(records):
        raise RetainedPhaseProcessCorpusError("exposure support differs")
    replay = _mapping(payload.get("strict_replay_audit"), context="replay audit")
    _canonical_identity(replay, "audit_identity")
    if replay.get("repeat_passes") != 2 or replay.get("fresh_process_count") != 84:
        raise RetainedPhaseProcessCorpusError("strict replay audit scope differs")
    claims = payload.get("claim_boundaries", {})
    if claims != {
        "games_played": 0,
        "candidate_loaded": False,
        "training_or_update": False,
        "playing_strength_claim": False,
        "refresh_causal_claim": False,
        "promotion_or_publication": False,
        "process_generalization_launch_authorized": False,
    }:
        raise RetainedPhaseProcessCorpusError("claim boundary differs")
    return {
        "corpus_identity": identity,
        "record_count": len(records),
        "phase_counts": dict(sorted(phase_counts.items())),
    }


def write_retained_phase_process_corpus(
    payload: Mapping[str, Any],
    output: str | Path,
) -> None:
    """Write one validated source artifact without overwriting evidence."""
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"phase process corpus already exists: {target}")
    validate_retained_phase_process_corpus(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

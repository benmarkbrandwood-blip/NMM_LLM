"""Candidate-blind, legally replayable phase starts for development probes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.evaluation.phase_corpus import (
    EXPECTED_SANMILL_COMMIT,
    EXPECTED_SOURCE_BLOB,
    EXPECTED_SOURCE_SHA256,
    PHASE_CORPUS_ID,
    SOURCE_ASSET,
    validate_phase_corpus,
)
from learned_ai.training.run_contract import canonical_sha256
from learned_ai.training.sanmill_referee import (
    SanmillTrainingGame,
    nmm_move_actions,
    training_installation_record,
)


CORPUS_SCHEMA = "nmm.phase-replay-development-corpus.v1"
CORPUS_ID = "dev-v4-phase-replay-development-corpus-v1"
CORPUS_STATUS = "frozen_development_measurement_only"
SANMILL_AUDIT_SCHEMA = "nmm.phase-replay-sanmill-audit.v1"
SOURCE_CORPUS_PATH = "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
SOURCE_CORPUS_SHA256 = (
    "cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e"
)
SOURCE_CORPUS_IDENTITY = (
    "cc5477b777cf38ea59c5f20af4bb7b6019b0d12b328ac139b2a8e7d9f2df0bc8"
)
PHASES = ("placement", "movement", "flying")
WDL_CLASSES = ("W", "D", "L")


class PhaseReplayCorpusError(RuntimeError):
    """Raised when a development start cannot be selected or replayed."""


def _json_document(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = entry.get("sources")
    if (
        not isinstance(sources, Sequence)
        or isinstance(sources, (str, bytes))
        or len(sources) != 1
        or not isinstance(sources[0], Mapping)
    ):
        raise PhaseReplayCorpusError("phase entry source is incomplete")
    return sources[0]


def _entry_order(entry: Mapping[str, Any]) -> tuple[int, str, int]:
    source = _source(entry)
    try:
        return (
            int(source["ply"]),
            str(source["source_identity"]),
            int(entry["index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PhaseReplayCorpusError("phase entry ordering fields are invalid") from exc


def select_replayable_phase_entries(
    phase_corpus: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select four identity-history starts per phase without model evidence."""
    validate_phase_corpus(phase_corpus)
    selected: list[dict[str, Any]] = []
    entries = phase_corpus["entries"]
    for phase in PHASES:
        eligible = sorted(
            (
                dict(entry)
                for entry in entries
                if entry.get("phase") == phase
                and _source(entry).get("color_transform") == "identity"
            ),
            key=_entry_order,
        )
        phase_selected: list[dict[str, Any]] = []
        for wdl in WDL_CLASSES:
            choice = next(
                (
                    entry
                    for entry in eligible
                    if entry["malom_wdl_for_side_to_move"] == wdl
                    and entry not in phase_selected
                ),
                None,
            )
            if choice is None:
                raise PhaseReplayCorpusError(
                    f"phase {phase} lacks an identity-history {wdl} start"
                )
            phase_selected.append(choice)

        used_trajectories = {
            int(_source(entry)["trajectory_index"]) for entry in phase_selected
        }
        remaining = [entry for entry in eligible if entry not in phase_selected]
        extra = next(
            (
                entry
                for entry in remaining
                if int(_source(entry)["trajectory_index"])
                not in used_trajectories
            ),
            remaining[0] if remaining else None,
        )
        if extra is None:
            raise PhaseReplayCorpusError(f"phase {phase} lacks a fourth start")
        phase_selected.append(extra)
        selected.extend(sorted(phase_selected, key=lambda item: int(item["index"])))
    return selected


def group_action_tokens(tokens: Sequence[str]) -> list[list[str]]:
    """Combine a primary TGF action and compulsory removal into one turn."""
    turns: list[list[str]] = []
    index = 0
    while index < len(tokens):
        primary = str(tokens[index])
        if not primary or primary.startswith("x"):
            raise PhaseReplayCorpusError("action history contains an orphan removal")
        turn = [primary]
        index += 1
        if index < len(tokens) and str(tokens[index]).startswith("x"):
            removal = str(tokens[index])
            if len(removal) <= 1:
                raise PhaseReplayCorpusError("action history contains an empty removal")
            turn.append(removal)
            index += 1
        turns.append(turn)
    return turns


def replay_logical_turns(turns: Sequence[Sequence[str]]) -> BoardState:
    """Replay complete logical turns through the independent NMM rules path."""
    board = BoardState.new_game()
    for logical_ply, actions in enumerate(turns, 1):
        expected = tuple(str(action) for action in actions)
        matches = [
            move
            for move in get_all_legal_moves(board)
            if nmm_move_actions(move) == expected
        ]
        if len(matches) != 1:
            raise PhaseReplayCorpusError(
                f"logical ply {logical_ply} has {len(matches)} matching legal moves"
            )
        board = board.apply_move(matches[0])
    return board


def replay_record_into_sanmill_game(
    record: Mapping[str, Any],
    game: SanmillTrainingGame,
) -> BoardState:
    """Replay one corpus record through local and strict Sanmill rules."""
    turns = record.get("logical_turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise PhaseReplayCorpusError("replay record logical turns are absent")
    board = BoardState.new_game()
    for logical_ply, actions in enumerate(turns, 1):
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise PhaseReplayCorpusError(
                f"logical ply {logical_ply} is not an action sequence"
            )
        expected = tuple(str(action) for action in actions)
        matches = [
            move
            for move in get_all_legal_moves(board)
            if nmm_move_actions(move) == expected
        ]
        if len(matches) != 1:
            raise PhaseReplayCorpusError(
                f"logical ply {logical_ply} has {len(matches)} matching legal moves"
            )
        game.apply_nmm_move(board, matches[0])
        board = board.apply_move(matches[0])
    game.assert_current_board(board)
    if board.to_fen_string() != record.get("fen"):
        raise PhaseReplayCorpusError("strict replay does not reach record FEN")
    if game.state.terminal:
        raise PhaseReplayCorpusError("strict referee makes replay record terminal")
    return board


def _fixture_history(
    fixture: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[list[str], list[list[str]]]:
    source = _source(entry)
    try:
        trajectory_index = int(source["trajectory_index"])
        step_index = int(source["step_index"])
        trajectories = fixture["trajectories"]
        trajectory = trajectories[trajectory_index - 1]
        steps = trajectory["steps"]
        step = steps[step_index - 1]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PhaseReplayCorpusError("phase entry fixture location is invalid") from exc
    if trajectory.get("seed") != source.get("trajectory_seed"):
        raise PhaseReplayCorpusError("phase entry trajectory seed differs")
    expected_step = {
        "ply": source.get("ply"),
        "fen": source.get("tgf_fen"),
        "side_to_move": source.get("side_to_move"),
        "phase_tag": source.get("phase_tag"),
        "picked_uci": source.get("picked_uci"),
    }
    observed_step = {
        "ply": step.get("ply"),
        "fen": step.get("fen"),
        "side_to_move": step.get("side_to_move"),
        "phase_tag": step.get("phase_tag"),
        "picked_uci": step.get("picked_uci"),
    }
    if observed_step != expected_step:
        raise PhaseReplayCorpusError("phase entry fixture step differs")
    tokens = [str(item.get("picked_uci", "")) for item in steps[: step_index - 1]]
    if any(not token for token in tokens):
        raise PhaseReplayCorpusError("fixture history contains an empty action")
    turns = group_action_tokens(tokens)
    replayed = replay_logical_turns(turns)
    if replayed.to_fen_string() != entry.get("fen"):
        raise PhaseReplayCorpusError("fixture history does not reach phase entry")
    return tokens, turns


def build_phase_replay_development_corpus(
    phase_corpus: Mapping[str, Any],
    fixture: Mapping[str, Any],
    fixture_source: Mapping[str, Any],
    *,
    source_corpus_sha256: str,
) -> dict[str, Any]:
    """Build the immutable 12-start development-only replay corpus."""
    if source_corpus_sha256 != SOURCE_CORPUS_SHA256:
        raise PhaseReplayCorpusError("source phase corpus file identity differs")
    if phase_corpus.get("corpus_id") != PHASE_CORPUS_ID:
        raise PhaseReplayCorpusError("source phase corpus id differs")
    if phase_corpus.get("corpus_identity") != SOURCE_CORPUS_IDENTITY:
        raise PhaseReplayCorpusError("source phase corpus identity differs")
    expected_fixture = {
        "sanmill_commit": EXPECTED_SANMILL_COMMIT,
        "asset_path": SOURCE_ASSET.as_posix(),
        "git_blob": EXPECTED_SOURCE_BLOB,
        "asset_sha256": EXPECTED_SOURCE_SHA256,
    }
    if any(fixture_source.get(key) != value for key, value in expected_fixture.items()):
        raise PhaseReplayCorpusError("pinned fixture identity differs")

    records: list[dict[str, Any]] = []
    for record_index, entry in enumerate(
        select_replayable_phase_entries(phase_corpus), 1
    ):
        tokens, turns = _fixture_history(fixture, entry)
        source = dict(_source(entry))
        body = {
            "record_index": record_index,
            "source_entry_index": int(entry["index"]),
            "fen": entry["fen"],
            "phase": entry["phase"],
            "turn": entry["turn"],
            "malom_wdl_for_side_to_move": entry["malom_wdl_for_side_to_move"],
            "ring16_canonical_fen": entry["ring16_canonical_fen"],
            "source": source,
            "action_history": tokens,
            "logical_turns": turns,
            "action_token_count": len(tokens),
            "logical_ply_count": len(turns),
        }
        records.append({**body, "record_identity": canonical_sha256(body)})

    payload: dict[str, Any] = {
        "schema_version": CORPUS_SCHEMA,
        "corpus_id": CORPUS_ID,
        "status": CORPUS_STATUS,
        "source_phase_corpus": {
            "path": SOURCE_CORPUS_PATH,
            "file_sha256": SOURCE_CORPUS_SHA256,
            "corpus_identity": SOURCE_CORPUS_IDENTITY,
        },
        "source_fixture": dict(fixture_source),
        "selection_contract": {
            "algorithm": "earliest-per-wdl-plus-new-trajectory-v1",
            "candidate_model_loaded": False,
            "color_transform": "identity-only",
            "phase_order": list(PHASES),
            "wdl_first_pass_order": list(WDL_CLASSES),
            "starts_per_phase": 4,
            "fourth_start": (
                "earliest remaining source from a not-yet-used trajectory when "
                "available; otherwise earliest remaining source"
            ),
        },
        "measurement_contract": {
            "purpose": "development-only common-anchor outcome measurement",
            "start_count": 12,
            "candidate_colors_per_start": ["W", "B"],
            "games_per_checkpoint": 24,
            "checkpoint_transition_boundaries": [4096, 8192],
            "optimizer_updates": 0,
            "training_games": 0,
            "writes_training_data": False,
        },
        "claim_boundaries": [
            "candidate-blind-source-selection",
            "development-corpus-not-held-out",
            "not-strength-or-promotion-evidence",
            "not-training-input",
        ],
        "records": records,
    }
    payload["corpus_identity"] = canonical_sha256(payload)
    validate_phase_replay_development_corpus(payload, phase_corpus=phase_corpus)
    return payload


def validate_phase_replay_development_corpus(
    payload: Mapping[str, Any],
    *,
    phase_corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate identities, coverage, and every complete legal replay."""
    if payload.get("schema_version") != CORPUS_SCHEMA:
        raise PhaseReplayCorpusError("replay corpus schema differs")
    if payload.get("corpus_id") != CORPUS_ID:
        raise PhaseReplayCorpusError("replay corpus id differs")
    if payload.get("status") != CORPUS_STATUS:
        raise PhaseReplayCorpusError("replay corpus status differs")
    identity = payload.get("corpus_identity")
    body = {key: value for key, value in payload.items() if key != "corpus_identity"}
    if identity != canonical_sha256(body):
        raise PhaseReplayCorpusError("replay corpus identity differs")
    if payload.get("source_phase_corpus") != {
        "path": SOURCE_CORPUS_PATH,
        "file_sha256": SOURCE_CORPUS_SHA256,
        "corpus_identity": SOURCE_CORPUS_IDENTITY,
    }:
        raise PhaseReplayCorpusError("source phase corpus binding differs")
    measurement = payload.get("measurement_contract", {})
    if (
        measurement.get("start_count") != 12
        or measurement.get("candidate_colors_per_start") != ["W", "B"]
        or measurement.get("games_per_checkpoint") != 24
        or measurement.get("checkpoint_transition_boundaries") != [4096, 8192]
        or measurement.get("optimizer_updates") != 0
        or measurement.get("training_games") != 0
        or measurement.get("writes_training_data") is not False
    ):
        raise PhaseReplayCorpusError("development measurement contract differs")

    source_entries: dict[int, Mapping[str, Any]] = {}
    if phase_corpus is not None:
        validate_phase_corpus(phase_corpus)
        source_entries = {
            int(entry["index"]): entry for entry in phase_corpus["entries"]
        }

    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 12:
        raise PhaseReplayCorpusError("replay corpus must contain 12 records")
    phase_counts = {phase: 0 for phase in PHASES}
    phase_wdls = {phase: set() for phase in PHASES}
    source_indices: set[int] = set()
    record_ids: set[str] = set()
    for expected_index, record in enumerate(records, 1):
        if not isinstance(record, Mapping):
            raise PhaseReplayCorpusError("replay record is not an object")
        record_body = {
            key: value for key, value in record.items() if key != "record_identity"
        }
        record_identity = record.get("record_identity")
        if record_identity != canonical_sha256(record_body):
            raise PhaseReplayCorpusError("replay record identity differs")
        if record_identity in record_ids:
            raise PhaseReplayCorpusError("replay record identity is duplicated")
        record_ids.add(str(record_identity))
        if record.get("record_index") != expected_index:
            raise PhaseReplayCorpusError("replay record order differs")
        source_index = int(record.get("source_entry_index", 0))
        if source_index in source_indices:
            raise PhaseReplayCorpusError("source phase entry is duplicated")
        source_indices.add(source_index)
        source = record.get("source")
        if not isinstance(source, Mapping) or source.get("color_transform") != "identity":
            raise PhaseReplayCorpusError("replay source is not an identity history")
        tokens = record.get("action_history")
        turns = record.get("logical_turns")
        if not isinstance(tokens, list) or not isinstance(turns, list):
            raise PhaseReplayCorpusError("replay history is incomplete")
        if [token for turn in turns for token in turn] != tokens:
            raise PhaseReplayCorpusError("logical turns do not flatten to history")
        if record.get("action_token_count") != len(tokens):
            raise PhaseReplayCorpusError("action-token count differs")
        if record.get("logical_ply_count") != len(turns):
            raise PhaseReplayCorpusError("logical-ply count differs")
        board = replay_logical_turns(turns)
        if board.to_fen_string() != record.get("fen"):
            raise PhaseReplayCorpusError("legal replay final FEN differs")
        if is_terminal(board)[0] or not get_all_legal_moves(board):
            raise PhaseReplayCorpusError("development start is not playable")
        phase = str(record.get("phase"))
        if phase not in PHASES:
            raise PhaseReplayCorpusError("development phase differs")
        if record.get("turn") != board.turn:
            raise PhaseReplayCorpusError("development side to move differs")
        wdl = str(record.get("malom_wdl_for_side_to_move"))
        if wdl not in WDL_CLASSES:
            raise PhaseReplayCorpusError("development Malom class differs")
        phase_counts[phase] += 1
        phase_wdls[phase].add(wdl)
        if source_entries:
            expected = source_entries.get(source_index)
            if expected is None:
                raise PhaseReplayCorpusError("source phase entry is absent")
            compared = {
                "fen": expected["fen"],
                "phase": expected["phase"],
                "turn": expected["turn"],
                "malom_wdl_for_side_to_move": expected[
                    "malom_wdl_for_side_to_move"
                ],
                "ring16_canonical_fen": expected["ring16_canonical_fen"],
                "source": dict(_source(expected)),
            }
            observed = {key: record.get(key) for key in compared}
            if observed != compared:
                raise PhaseReplayCorpusError("source phase entry projection differs")
    if phase_counts != {phase: 4 for phase in PHASES}:
        raise PhaseReplayCorpusError("phase coverage differs")
    if any(phase_wdls[phase] != set(WDL_CLASSES) for phase in PHASES):
        raise PhaseReplayCorpusError("per-phase WDL coverage differs")
    return {
        "corpus_identity": identity,
        "record_count": len(records),
        "phase_counts": phase_counts,
        "source_entry_indices": sorted(source_indices),
    }


def write_phase_replay_development_corpus(
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """Write the validated generated corpus without overwriting evidence."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"development corpus target exists: {output}")
    validate_phase_replay_development_corpus(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_document(payload))


def audit_phase_replays_with_sanmill(
    payload: Mapping[str, Any],
    installation: Any,
    *,
    repeat_passes: int = 2,
) -> dict[str, Any]:
    """Replay every history in fresh strict-referee processes twice."""
    validate_phase_replay_development_corpus(payload)
    if repeat_passes != 2:
        raise PhaseReplayCorpusError("strict replay audit requires two passes")
    passes: list[list[dict[str, Any]]] = []
    for _pass_index in range(repeat_passes):
        observations: list[dict[str, Any]] = []
        for record in payload["records"]:
            with SanmillTrainingGame(installation, seed=42) as game:
                board = replay_record_into_sanmill_game(record, game)
                observations.append(
                    {
                        "record_index": record["record_index"],
                        "record_identity": record["record_identity"],
                        "fen": board.to_fen_string(),
                        "history_sha256": game.state.history_sha256,
                        "logical_ply_count": game.state.logical_ply_count,
                        "action_token_count": game.state.action_token_count,
                        "terminal": False,
                    }
                )
        passes.append(observations)
    if passes[0] != passes[1]:
        raise PhaseReplayCorpusError("fresh-process strict replay differs")
    report: dict[str, Any] = {
        "schema_version": SANMILL_AUDIT_SCHEMA,
        "corpus_id": payload["corpus_id"],
        "corpus_identity": payload["corpus_identity"],
        "sanmill_training_runtime": training_installation_record(
            installation,
            seed=42,
        ),
        "repeat_passes": repeat_passes,
        "fresh_process_count": repeat_passes * len(payload["records"]),
        "repeat_passes_byte_equal": True,
        "records": passes[0],
        "claim_boundary": (
            "strict history replay and nonterminal-state evidence only; no "
            "candidate, search, training, strength, or promotion claim"
        ),
    }
    report["audit_identity"] = canonical_sha256(report)
    validate_phase_replay_sanmill_audit(report, corpus=payload)
    return report


def validate_phase_replay_sanmill_audit(
    report: Mapping[str, Any],
    *,
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the portable strict-referee replay evidence."""
    validate_phase_replay_development_corpus(corpus)
    if report.get("schema_version") != SANMILL_AUDIT_SCHEMA:
        raise PhaseReplayCorpusError("strict replay audit schema differs")
    identity = report.get("audit_identity")
    body = {key: value for key, value in report.items() if key != "audit_identity"}
    if identity != canonical_sha256(body):
        raise PhaseReplayCorpusError("strict replay audit identity differs")
    if (
        report.get("corpus_id") != corpus["corpus_id"]
        or report.get("corpus_identity") != corpus["corpus_identity"]
        or report.get("repeat_passes") != 2
        or report.get("fresh_process_count") != 24
        or report.get("repeat_passes_byte_equal") is not True
    ):
        raise PhaseReplayCorpusError("strict replay audit contract differs")
    runtime = report.get("sanmill_training_runtime")
    if not isinstance(runtime, Mapping):
        raise PhaseReplayCorpusError("strict replay runtime is absent")
    runtime_identity = runtime.get("identity")
    runtime_body = {key: value for key, value in runtime.items() if key != "identity"}
    if runtime_identity != canonical_sha256(runtime_body):
        raise PhaseReplayCorpusError("strict replay runtime identity differs")
    observations = report.get("records")
    if not isinstance(observations, list) or len(observations) != 12:
        raise PhaseReplayCorpusError("strict replay observations differ")
    for record, observation in zip(corpus["records"], observations, strict=True):
        expected = {
            "record_index": record["record_index"],
            "record_identity": record["record_identity"],
            "fen": record["fen"],
            "logical_ply_count": record["logical_ply_count"],
            "action_token_count": record["action_token_count"],
            "terminal": False,
        }
        if any(observation.get(key) != value for key, value in expected.items()):
            raise PhaseReplayCorpusError("strict replay observation differs")
        history_identity = observation.get("history_sha256")
        if not isinstance(history_identity, str) or len(history_identity) != 64:
            raise PhaseReplayCorpusError("strict replay history identity differs")
    return {
        "audit_identity": identity,
        "runtime_identity": runtime_identity,
        "record_count": len(observations),
        "fresh_process_count": report["fresh_process_count"],
    }


def write_phase_replay_sanmill_audit(
    report: Mapping[str, Any],
    output_path: str | Path,
    *,
    corpus: Mapping[str, Any],
) -> None:
    """Write validated portable strict-referee evidence without overwrite."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"strict replay audit target exists: {output}")
    validate_phase_replay_sanmill_audit(report, corpus=corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_document(report))


def source_corpus_sha256(path: str | Path) -> str:
    """Return the source file identity used by the generator."""
    return _sha256_file(Path(path))

"""Full-oracle alternatives for committed held-out WDL downgrades.

This diagnostic consumes the immutable held-out ledger and the already-bound
WDL transition audit.  It never loads the candidate policy.  At each observed
first downgrade it enumerates complete legal turns and compares their lossless
Malom values in the common parent context.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ai.malom_db import OracleMoveValue, compare_oracle_move_values
from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.data.data_contract import (
    load_dataset_manifest,
    verify_dataset_snapshot,
)
from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION
from learned_ai.evaluation.heldout_evaluation import (
    EXPECTED_MALOM_IDENTITY,
    HeldoutEvaluationError,
    load_game_ledger,
    load_runtime_spec,
    recompute_heldout_evaluation,
)
from learned_ai.evaluation.heldout_loss_audit import HELDOUT_WDL_AUDIT_SCHEMA
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


HELDOUT_ORACLE_ALTERNATIVE_SCHEMA = (
    "nmm.sanmill-heldout-oracle-alternative-audit.v1"
)
EXPECTED_WDL_AUDIT_SHA256 = (
    "871dd7935f7aa3231e6e364974e5207ef272501483e29338014cde16525b5692"
)
EXPECTED_WDL_AUDIT_IDENTITY = (
    "6bbb4a50aa7999d06679c802cfeb5b913f0f5abf0689aa0291ec55459304b504"
)
OUTCOME_TO_WDL = {"win": "W", "draw": "D", "loss": "L"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _oracle_record(value: OracleMoveValue) -> dict[str, Any]:
    return {
        "absolute_key1": int(value.absolute_key1),
        "entry_source": value.source,
        "key1": int(value.key1),
        "key2": int(value.key2),
        "ordering_key": [int(item) for item in value.ordering_key()],
        "outcome": value.outcome,
        "perspective": value.perspective,
        "sector": [int(item) for item in value.sector],
        "sector_value": int(value.sector_value),
        "terminal": bool(value.terminal),
    }


def _result_record(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("oracle_value")
    if not isinstance(value, OracleMoveValue):
        raise HeldoutEvaluationError("a legal alternative lacks full Malom value")
    move = dict(row["move"])
    return {
        "dtm_projection": row.get("dtm"),
        "move": move,
        "move_identity": canonical_sha256(move),
        "oracle_value": _oracle_record(value),
        "wdl": row["wdl"],
    }


def _same_primary(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("from") == right.get("from") and left.get("to") == right.get(
        "to"
    )


def analyze_oracle_alternatives(
    *,
    chosen_move: Mapping[str, Any],
    before_candidate_wdl: str,
    after_candidate_wdl: str,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify why one committed complete turn loses full oracle value."""

    if before_candidate_wdl not in {"W", "D"}:
        raise HeldoutEvaluationError("oracle alternative parent is already losing")
    if after_candidate_wdl not in {"D", "L"}:
        raise HeldoutEvaluationError("oracle alternative child is not downgraded")
    expected_before = {"W": "win", "D": "draw"}[before_candidate_wdl]
    expected_after = {"D": "draw", "L": "loss"}[after_candidate_wdl]

    normalized: list[dict[str, Any]] = []
    values: list[OracleMoveValue] = []
    move_identities: set[str] = set()
    for row in results:
        if row.get("wdl") not in OUTCOME_TO_WDL:
            raise HeldoutEvaluationError("a legal alternative has unknown Malom WDL")
        value = row.get("oracle_value")
        if not isinstance(value, OracleMoveValue):
            raise HeldoutEvaluationError("a legal alternative lacks full Malom value")
        if OUTCOME_TO_WDL[row["wdl"]] != value.outcome:
            raise HeldoutEvaluationError("coarse and full Malom outcomes differ")
        move_identity = canonical_sha256(row["move"])
        if move_identity in move_identities:
            raise HeldoutEvaluationError("oracle alternatives contain a duplicate turn")
        move_identities.add(move_identity)
        normalized.append(dict(row))
        values.append(value)
    if not normalized:
        raise HeldoutEvaluationError("oracle alternative state has no legal turns")

    contexts = {
        (value.sector, value.sector_value, value.perspective) for value in values
    }
    if len(contexts) != 1:
        raise HeldoutEvaluationError("oracle alternatives have mixed parent contexts")

    chosen_rows = [row for row in normalized if row["move"] == dict(chosen_move)]
    if len(chosen_rows) != 1:
        raise HeldoutEvaluationError("committed downgrade is not one legal oracle turn")
    chosen = chosen_rows[0]
    if chosen["wdl"] != expected_after:
        raise HeldoutEvaluationError("committed downgrade WDL differs from full oracle")

    preserving = [row for row in normalized if row["wdl"] == expected_before]
    if not preserving:
        raise HeldoutEvaluationError("downgrade state has no value-preserving alternative")
    same_primary_preserving = [
        row
        for row in preserving
        if _same_primary(row["move"], chosen["move"])
    ]

    best = normalized[0]
    for row in normalized[1:]:
        if compare_oracle_move_values(row["oracle_value"], best["oracle_value"]) > 0:
            best = row
    full_best = [
        row
        for row in normalized
        if compare_oracle_move_values(row["oracle_value"], best["oracle_value"]) == 0
    ]
    strictly_better = [
        row
        for row in normalized
        if compare_oracle_move_values(
            row["oracle_value"], chosen["oracle_value"]
        )
        > 0
    ]
    if not strictly_better:
        raise HeldoutEvaluationError("downgraded turn has no better oracle alternative")

    if same_primary_preserving:
        classification = "wrong_capture_target"
    elif chosen["move"].get("capture") is not None:
        classification = "primary_action_or_mill_timing"
    else:
        classification = "primary_action"

    def sort_key(row: Mapping[str, Any]) -> str:
        return canonical_sha256(row["move"])

    return {
        "classification": classification,
        "chosen": _result_record(chosen),
        "chosen_is_full_oracle_best": any(chosen is row for row in full_best),
        "full_oracle_best": [
            _result_record(row) for row in sorted(full_best, key=sort_key)
        ],
        "legal_complete_turns": len(normalized),
        "preserving_alternatives": [
            _result_record(row) for row in sorted(preserving, key=sort_key)
        ],
        "preserving_alternatives_count": len(preserving),
        "same_primary_preserving": [
            _result_record(row)
            for row in sorted(same_primary_preserving, key=sort_key)
        ],
        "same_primary_preserving_count": len(same_primary_preserving),
        "strictly_better_alternatives_count": len(strictly_better),
    }


def _replay_to_downgrade(
    record: Mapping[str, Any],
    downgrade: Mapping[str, Any],
) -> tuple[BoardState, Mapping[str, Any]]:
    board = BoardState.from_fen_string(str(record["prefix"]["final_nmm_fen"]))
    target_ply = int(downgrade["post_prefix_logical_ply"])
    for turn in record["turns"]:
        move = dict(turn["move"])
        if board.turn != turn["mover_color"] or move not in get_all_legal_moves(board):
            raise HeldoutEvaluationError("ledger differs during oracle replay")
        if int(turn["post_prefix_logical_ply"]) == target_ply:
            if (
                turn["actor"] != "candidate"
                or board.to_fen_string() != downgrade["before_fen"]
                or move != downgrade["move"]
                or canonical_sha256(turn) != downgrade["turn_identity"]
            ):
                raise HeldoutEvaluationError("oracle downgrade binding differs")
            return board, turn
        after = board.apply_move(move)
        if after.to_fen_string() != turn["local_fen_after"]:
            raise HeldoutEvaluationError("ledger FEN differs during oracle replay")
        board = after
    raise HeldoutEvaluationError("oracle downgrade turn is absent from the ledger")


def _load_source_audit(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_WDL_AUDIT_SHA256:
        raise HeldoutEvaluationError("WDL transition audit file hash differs")
    try:
        audit = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeldoutEvaluationError("WDL transition audit is invalid JSON") from exc
    if raw != canonical_json_bytes(audit):
        raise HeldoutEvaluationError("WDL transition audit is not canonical JSON")
    identity = audit.pop("audit_identity", None)
    if (
        identity != EXPECTED_WDL_AUDIT_IDENTITY
        or canonical_sha256(audit) != identity
        or audit.get("schema_version") != HELDOUT_WDL_AUDIT_SCHEMA
    ):
        raise HeldoutEvaluationError("WDL transition audit identity differs")
    return {**audit, "audit_identity": identity}


def _validate_auditor(auditor: Mapping[str, Any]) -> None:
    expected = {
        "implementation_commit",
        "implementation_tree",
        "module_sha256",
        "script_sha256",
        "tracked_worktree_clean",
    }
    if (
        set(auditor) != expected
        or auditor.get("tracked_worktree_clean") is not True
        or any(
            not isinstance(auditor.get(field), str) or not auditor[field]
            for field in expected - {"tracked_worktree_clean"}
        )
    ):
        raise HeldoutEvaluationError("oracle auditor implementation identity differs")


def build_heldout_oracle_alternative_audit(
    *,
    spec_path: Path,
    ledger_path: Path,
    report_path: Path,
    wdl_audit_path: Path,
    malom_path: Path,
    malom_manifest_path: Path,
    auditor: Mapping[str, Any],
    query_factory: Callable[[Path], Any] = lambda path: ExternalSolvedDB(
        str(path), strict=True
    ),
) -> dict[str, Any]:
    """Build the full-oracle counterfactual audit for 19 committed states."""

    _validate_auditor(auditor)
    spec = load_runtime_spec(spec_path)
    records, ledger_tail = load_game_ledger(spec, ledger_path)
    if len(records) != 128:
        raise HeldoutEvaluationError("oracle audit requires the complete ledger")
    record_by_ordinal = {int(record["ordinal"]): record for record in records}
    recomputed = recompute_heldout_evaluation(spec_path, ledger_path)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    if persisted != recomputed or persisted.get("status") != "completed":
        raise HeldoutEvaluationError("oracle audit report binding differs")

    source_audit = _load_source_audit(wdl_audit_path)
    if (
        source_audit.get("spec_identity") != spec["spec_identity"]
        or source_audit.get("ledger_sha256") != _sha256_file(ledger_path)
        or source_audit.get("ledger_tail_record_sha256") != ledger_tail
        or source_audit.get("report_identity") != persisted["result_identity"]
    ):
        raise HeldoutEvaluationError("oracle source audit evaluation binding differs")
    downgrade_rows = sorted(
        (
            row
            for row in source_audit["per_game"]
            if row["cohort"] == "candidate_loss"
            and row["first_candidate_wdl_downgrade"] is not None
        ),
        key=lambda row: int(row["ordinal"]),
    )
    if len(downgrade_rows) != 19:
        raise HeldoutEvaluationError("oracle source audit downgrade count differs")

    manifest = load_dataset_manifest(malom_manifest_path)
    if (
        manifest.manifest_sha256 != EXPECTED_MALOM_IDENTITY
        or manifest.logical_name != "malom_tablebase"
        or manifest.trust_level != CURRENT_MALOM_LABEL_VERSION
    ):
        raise HeldoutEvaluationError("oracle audit Malom manifest differs")
    snapshot = verify_dataset_snapshot(malom_path, manifest)

    database = query_factory(malom_path)
    per_state: list[dict[str, Any]] = []
    try:
        if not database.is_available():
            raise HeldoutEvaluationError("oracle audit Malom tablebase is unavailable")
        for row in downgrade_rows:
            record = record_by_ordinal[int(row["ordinal"])]
            if any(
                record[field] != row[field]
                for field in (
                    "candidate_color",
                    "game_id",
                    "pair_index",
                    "source_core_id",
                    "stratum",
                )
            ):
                raise HeldoutEvaluationError("oracle source row differs from ledger")
            downgrade = row["first_candidate_wdl_downgrade"]
            board, turn = _replay_to_downgrade(record, downgrade)
            if board.turn != row["candidate_color"]:
                raise HeldoutEvaluationError("oracle candidate colour differs")
            results = database.query_all_moves(board, board.turn)
            legal_identities = sorted(
                canonical_sha256(move) for move in get_all_legal_moves(board)
            )
            result_identities = sorted(
                canonical_sha256(result["move"]) for result in results
            )
            if result_identities != legal_identities:
                raise HeldoutEvaluationError("oracle result inventory differs")
            analysis = analyze_oracle_alternatives(
                chosen_move=turn["move"],
                before_candidate_wdl=downgrade["before_candidate_wdl"],
                after_candidate_wdl=downgrade["after_candidate_wdl"],
                results=results,
            )
            per_state.append(
                {
                    "candidate_color": row["candidate_color"],
                    "downgrade_transition": (
                        f'{downgrade["before_candidate_wdl"]}'
                        f'->{downgrade["after_candidate_wdl"]}'
                    ),
                    "game_id": row["game_id"],
                    "ordinal": int(row["ordinal"]),
                    "pair_index": int(row["pair_index"]),
                    "phase": downgrade["phase"],
                    "post_prefix_logical_ply": int(
                        downgrade["post_prefix_logical_ply"]
                    ),
                    "source_core_id": row["source_core_id"],
                    "stratum": row["stratum"],
                    "turn_identity": downgrade["turn_identity"],
                    **analysis,
                }
            )
    finally:
        database.close()

    body = {
        "schema_version": HELDOUT_ORACLE_ALTERNATIVE_SCHEMA,
        "evaluation_id": spec["evaluation_id"],
        "spec_identity": spec["spec_identity"],
        "ledger_sha256": _sha256_file(ledger_path),
        "ledger_tail_record_sha256": ledger_tail,
        "report_identity": persisted["result_identity"],
        "source_wdl_audit": {
            "audit_identity": source_audit["audit_identity"],
            "file_sha256": EXPECTED_WDL_AUDIT_SHA256,
        },
        "auditor": dict(auditor),
        "malom": {
            "component_count": snapshot["component_count"],
            "identity": manifest.manifest_sha256,
            "size_bytes": snapshot["size_bytes"],
            "trust_level": manifest.trust_level,
        },
        "summary": {
            "classification_counts": dict(
                sorted(Counter(row["classification"] for row in per_state).items())
            ),
            "chosen_capture_counts": dict(
                sorted(
                    Counter(
                        "capture"
                        if row["chosen"]["move"].get("capture") is not None
                        else "no_capture"
                        for row in per_state
                    ).items()
                )
            ),
            "chosen_full_oracle_best": sum(
                bool(row["chosen_is_full_oracle_best"]) for row in per_state
            ),
            "phase_counts": dict(
                sorted(Counter(row["phase"] for row in per_state).items())
            ),
            "states": len(per_state),
            "stratum_counts": dict(
                sorted(Counter(row["stratum"] for row in per_state).items())
            ),
        },
        "per_state": per_state,
        "claim_boundary": {
            "candidate_policy_loaded": False,
            "candidate_requeried": False,
            "complete_legal_turns_enumerated": True,
            "counterfactual_oracle_search": True,
            "causal_training_component_identified": False,
            "diagnostic_not_inferential": True,
            "history_dependent_draws_in_malom": False,
            "new_games": False,
        },
    }
    return {**body, "audit_identity": canonical_sha256(body)}


def write_new_audit(path: Path, audit: Mapping[str, Any]) -> None:
    """Write one canonical audit without overwriting prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(dict(audit)))
        handle.flush()
        os.fsync(handle.fileno())

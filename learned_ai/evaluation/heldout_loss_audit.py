"""Post-hoc WDL transition audit for the completed frozen held-out ledger.

The audit never loads or calls the candidate policy.  It replays only moves
already committed to the immutable held-out ledger and queries the corrected
Malom tablebase for the candidate's actual turns.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase
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
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


HELDOUT_WDL_AUDIT_SCHEMA = "nmm.sanmill-heldout-wdl-transition-audit.v1"
WDL_RANK = {"W": 1, "D": 0, "L": -1}
NEGATE_WDL = {"W": "L", "D": "D", "L": "W"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_losses_and_matched_draws(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Select every loss and one deterministic non-reused matched draw.

    Controls match source stratum, candidate colour, and strict-subset
    membership.  This is an outcome-aware diagnostic cohort, not an
    inferential sample.
    """

    losses = sorted(
        (record for record in records if float(record["candidate_score"]) == 0.0),
        key=lambda record: int(record["ordinal"]),
    )
    draws = sorted(
        (record for record in records if float(record["candidate_score"]) == 0.5),
        key=lambda record: int(record["ordinal"]),
    )
    used_controls: set[int] = set()
    controls: list[Mapping[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for loss in losses:
        candidates = [
            draw
            for draw in draws
            if int(draw["ordinal"]) not in used_controls
            and draw["stratum"] == loss["stratum"]
            and draw["candidate_color"] == loss["candidate_color"]
            and bool(draw["strict_independence_sensitivity"])
            == bool(loss["strict_independence_sensitivity"])
        ]
        if not candidates:
            raise HeldoutEvaluationError(
                "a held-out loss has no unused colour/stratum/strict matched draw"
            )
        control = candidates[0]
        used_controls.add(int(control["ordinal"]))
        controls.append(control)
        matches.append(
            {
                "candidate_color": loss["candidate_color"],
                "control_ordinal": int(control["ordinal"]),
                "loss_ordinal": int(loss["ordinal"]),
                "stratum": loss["stratum"],
                "strict_independence_sensitivity": bool(
                    loss["strict_independence_sensitivity"]
                ),
            }
        )
    return losses, controls, matches


def _candidate_perspective(wdl: str, *, mover: str, candidate_color: str) -> str:
    if wdl not in WDL_RANK:
        raise HeldoutEvaluationError("Malom returned an invalid WDL value")
    return wdl if mover == candidate_color else NEGATE_WDL[wdl]


def _primary_mobility(board: BoardState, color: str) -> int:
    phase = get_game_phase(board, color)
    if phase == "place":
        return len(board.legal_placements(color))
    return len(board.legal_moves(color))


def _position_features(board: BoardState, candidate_color: str) -> dict[str, Any]:
    opponent = "B" if candidate_color == "W" else "W"
    candidate_mobility = _primary_mobility(board, candidate_color)
    opponent_mobility = _primary_mobility(board, opponent)
    candidate_pieces = int(board.pieces_on_board[candidate_color])
    opponent_pieces = int(board.pieces_on_board[opponent])
    return {
        "candidate": {
            "color": candidate_color,
            "phase": get_game_phase(board, candidate_color),
            "pieces_on_board": candidate_pieces,
            "pieces_placed": int(board.pieces_placed[candidate_color]),
            "primary_mobility": candidate_mobility,
        },
        "material_difference": candidate_pieces - opponent_pieces,
        "opponent": {
            "color": opponent,
            "phase": get_game_phase(board, opponent),
            "pieces_on_board": opponent_pieces,
            "pieces_placed": int(board.pieces_placed[opponent]),
            "primary_mobility": opponent_mobility,
        },
        "primary_mobility_difference": candidate_mobility - opponent_mobility,
        "side_to_move": board.turn,
    }


def _numeric_summary(values: Sequence[int]) -> dict[str, Any]:
    return {
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "support": len(values),
    }


def _position_summary(features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_phase_counts": dict(
            sorted(
                Counter(str(row["candidate"]["phase"]) for row in features).items()
            )
        ),
        "candidate_pieces_on_board": _numeric_summary(
            [int(row["candidate"]["pieces_on_board"]) for row in features]
        ),
        "material_difference": _numeric_summary(
            [int(row["material_difference"]) for row in features]
        ),
        "primary_mobility_difference": _numeric_summary(
            [int(row["primary_mobility_difference"]) for row in features]
        ),
    }


def audit_game_wdl_transitions(
    record: Mapping[str, Any],
    query_state: Callable[[BoardState], str | None],
    *,
    cohort: str,
) -> dict[str, Any]:
    """Locate the first candidate WDL downgrade in one committed game."""

    candidate_color = str(record["candidate_color"])
    board = BoardState.from_fen_string(str(record["prefix"]["final_nmm_fen"]))
    prefix_side_wdl = query_state(board)
    prefix_candidate_wdl = (
        None
        if prefix_side_wdl is None
        else _candidate_perspective(
            prefix_side_wdl,
            mover=board.turn,
            candidate_color=candidate_color,
        )
    )
    prefix_position_features = _position_features(board, candidate_color)
    first_downgrade: dict[str, Any] | None = None
    unknown: dict[str, Any] | None = None
    candidate_turns = 0
    known_candidate_turns = 0
    transitions: Counter[str] = Counter()
    candidate_positions: list[dict[str, Any]] = []

    for turn in record["turns"]:
        move = dict(turn["move"])
        if board.turn != turn["mover_color"]:
            raise HeldoutEvaluationError("ledger mover differs during WDL replay")
        if move not in get_all_legal_moves(board):
            raise HeldoutEvaluationError("ledger move is illegal during WDL replay")
        before_fen = board.to_fen_string()
        after = board.apply_move(move)
        if after.to_fen_string() != turn["local_fen_after"]:
            raise HeldoutEvaluationError("ledger local FEN differs during WDL replay")

        if turn["actor"] == "candidate" and first_downgrade is None and unknown is None:
            candidate_turns += 1
            before_features = _position_features(board, candidate_color)
            after_features = _position_features(after, candidate_color)
            candidate_positions.append(before_features)
            before_wdl = query_state(board)
            after_opponent_wdl = query_state(after)
            if before_wdl is None or after_opponent_wdl is None:
                unknown = {
                    "before_fen": before_fen,
                    "post_prefix_logical_ply": int(
                        turn["post_prefix_logical_ply"]
                    ),
                    "turn_identity": canonical_sha256(turn),
                }
            else:
                if board.turn != candidate_color:
                    raise HeldoutEvaluationError(
                        "candidate actor does not match candidate colour"
                    )
                after_wdl = NEGATE_WDL[after_opponent_wdl]
                delta = WDL_RANK[after_wdl] - WDL_RANK[before_wdl]
                if delta > 0:
                    raise HeldoutEvaluationError(
                        "Malom candidate transition improves beyond the parent value"
                    )
                known_candidate_turns += 1
                transition = f"{before_wdl}->{after_wdl}"
                transitions[transition] += 1
                if delta < 0:
                    first_downgrade = {
                        "after_candidate_wdl": after_wdl,
                        "after_fen": after.to_fen_string(),
                        "before_candidate_wdl": before_wdl,
                        "before_fen": before_fen,
                        "before_position_features": before_features,
                        "delta": delta,
                        "move": move,
                        "after_position_features": after_features,
                        "phase": get_game_phase(board, candidate_color),
                        "post_prefix_logical_ply": int(
                            turn["post_prefix_logical_ply"]
                        ),
                        "turn_identity": canonical_sha256(turn),
                    }
        board = after

    if prefix_candidate_wdl is None or unknown is not None:
        classification = "insufficient_malom_coverage"
    elif prefix_candidate_wdl == "L":
        classification = "candidate_already_losing_at_prefix"
    elif first_downgrade is not None:
        classification = "candidate_wdl_downgrade_found"
    else:
        classification = "no_candidate_wdl_downgrade_observed"

    return {
        "candidate_color": candidate_color,
        "candidate_score": float(record["candidate_score"]),
        "candidate_turns_probed": candidate_turns,
        "classification": classification,
        "cohort": cohort,
        "first_candidate_wdl_downgrade": first_downgrade,
        "game_id": record["game_id"],
        "known_candidate_turns": known_candidate_turns,
        "ordinal": int(record["ordinal"]),
        "outcome_reason": record["outcome_reason"],
        "pair_index": int(record["pair_index"]),
        "prefix_candidate_wdl": prefix_candidate_wdl,
        "prefix_position_features": prefix_position_features,
        "source_core_id": record["source_core_id"],
        "stratum": record["stratum"],
        "strict_independence_sensitivity": bool(
            record["strict_independence_sensitivity"]
        ),
        "transition_counts_before_stop": dict(sorted(transitions.items())),
        "position_summary_before_stop": _position_summary(candidate_positions),
        "unknown_transition": unknown,
    }


def _cohort_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first_plies = [
        int(row["first_candidate_wdl_downgrade"]["post_prefix_logical_ply"])
        for row in rows
        if row["first_candidate_wdl_downgrade"] is not None
    ]
    transitions = Counter(
        (
            f'{row["first_candidate_wdl_downgrade"]["before_candidate_wdl"]}'
            f'->{row["first_candidate_wdl_downgrade"]["after_candidate_wdl"]}'
        )
        for row in rows
        if row["first_candidate_wdl_downgrade"] is not None
    )
    return {
        "classification_counts": dict(
            sorted(Counter(str(row["classification"]) for row in rows).items())
        ),
        "first_downgrade_phase_counts": dict(
            sorted(
                Counter(
                    str(row["first_candidate_wdl_downgrade"]["phase"])
                    for row in rows
                    if row["first_candidate_wdl_downgrade"] is not None
                ).items()
            )
        ),
        "first_downgrade_ply": {
            "max": max(first_plies) if first_plies else None,
            "mean": sum(first_plies) / len(first_plies) if first_plies else None,
            "min": min(first_plies) if first_plies else None,
            "support": len(first_plies),
        },
        "first_downgrade_transition_counts": dict(sorted(transitions.items())),
        "games": len(rows),
        "prefix_candidate_wdl_counts": dict(
            sorted(Counter(str(row["prefix_candidate_wdl"]) for row in rows).items())
        ),
    }


def build_heldout_wdl_transition_audit(
    *,
    spec_path: Path,
    ledger_path: Path,
    report_path: Path,
    malom_path: Path,
    malom_manifest_path: Path,
    auditor: Mapping[str, Any],
    query_state_factory: Callable[[Path], Any] = lambda path: ExternalSolvedDB(
        str(path), strict=True
    ),
) -> dict[str, Any]:
    """Build the deterministic loss/control audit from frozen inputs."""

    expected_auditor_fields = {
        "implementation_commit",
        "implementation_tree",
        "module_sha256",
        "script_sha256",
        "tracked_worktree_clean",
    }
    if (
        set(auditor) != expected_auditor_fields
        or auditor.get("tracked_worktree_clean") is not True
        or any(
            not isinstance(auditor.get(field), str) or not auditor[field]
            for field in expected_auditor_fields - {"tracked_worktree_clean"}
        )
    ):
        raise HeldoutEvaluationError("WDL auditor implementation identity differs")

    spec = load_runtime_spec(spec_path)
    records, ledger_tail = load_game_ledger(spec, ledger_path)
    if len(records) != 128:
        raise HeldoutEvaluationError("WDL audit requires the complete 128-game ledger")
    recomputed = recompute_heldout_evaluation(spec_path, ledger_path)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    if persisted != recomputed or persisted.get("status") != "completed":
        raise HeldoutEvaluationError("WDL audit report binding differs")

    manifest = load_dataset_manifest(malom_manifest_path)
    if (
        manifest.manifest_sha256 != EXPECTED_MALOM_IDENTITY
        or manifest.logical_name != "malom_tablebase"
        or manifest.trust_level != CURRENT_MALOM_LABEL_VERSION
    ):
        raise HeldoutEvaluationError("WDL audit Malom manifest differs")
    snapshot = verify_dataset_snapshot(malom_path, manifest)

    losses, controls, matches = select_losses_and_matched_draws(records)
    database = query_state_factory(malom_path)
    try:
        if not database.is_available():
            raise HeldoutEvaluationError("WDL audit Malom tablebase is unavailable")
        audited_losses = [
            audit_game_wdl_transitions(
                record,
                database.query_state,
                cohort="candidate_loss",
            )
            for record in losses
        ]
        audited_controls = [
            audit_game_wdl_transitions(
                record,
                database.query_state,
                cohort="matched_draw_control",
            )
            for record in controls
        ]
    finally:
        database.close()

    per_game = sorted(
        [*audited_losses, *audited_controls],
        key=lambda row: (str(row["cohort"]), int(row["ordinal"])),
    )
    body = {
        "schema_version": HELDOUT_WDL_AUDIT_SCHEMA,
        "evaluation_id": spec["evaluation_id"],
        "spec_identity": spec["spec_identity"],
        "ledger_sha256": _sha256_file(ledger_path),
        "ledger_tail_record_sha256": ledger_tail,
        "report_identity": persisted["result_identity"],
        "auditor": dict(auditor),
        "malom": {
            "component_count": snapshot["component_count"],
            "identity": manifest.manifest_sha256,
            "size_bytes": snapshot["size_bytes"],
            "trust_level": manifest.trust_level,
        },
        "selection": {
            "diagnostic_not_inferential": True,
            "loss_games": len(losses),
            "matched_draw_controls": len(controls),
            "matching_fields": [
                "stratum",
                "candidate_color",
                "strict_independence_sensitivity",
            ],
            "matches": matches,
            "selection_identity": canonical_sha256(matches),
        },
        "summary": {
            "candidate_losses": _cohort_summary(audited_losses),
            "matched_draw_controls": _cohort_summary(audited_controls),
        },
        "per_game": per_game,
        "claim_boundary": {
            "candidate_policy_loaded": False,
            "candidate_requeried": False,
            "complete_oracle_step_ordering": False,
            "counterfactual_move_search": False,
            "first_wdl_downgrade_only": True,
            "history_dependent_draws_in_malom": False,
            "statistical_effect_claim": False,
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

"""No-update integration evidence for the corrected mill-bonus contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from game.board import MILLS, BoardState
from game.rules import get_all_legal_moves
from learned_ai.data.data_contract import (
    load_dataset_manifest,
    verify_dataset_snapshot,
)
from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION
from learned_ai.evaluation.heldout_loss_audit import HELDOUT_WDL_AUDIT_SCHEMA
from learned_ai.evaluation.heldout_oracle_alternative_audit import (
    HELDOUT_ORACLE_ALTERNATIVE_SCHEMA,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from scripts.train_s_gen_v2 import _mill_formation_reward


MILL_BONUS_NO_UPDATE_PROBE_SCHEMA = "nmm.mill-bonus-no-update-probe.v1"
EXPECTED_WDL_AUDIT_SHA256 = (
    "871dd7935f7aa3231e6e364974e5207ef272501483e29338014cde16525b5692"
)
EXPECTED_WDL_AUDIT_IDENTITY = (
    "6bbb4a50aa7999d06679c802cfeb5b913f0f5abf0689aa0291ec55459304b504"
)
EXPECTED_ORACLE_AUDIT_SHA256 = (
    "29e3ed6d2af1389a90ef46869db5a2b8800e8c9c3993e13dd80af72ef07a7f28"
)
EXPECTED_ORACLE_AUDIT_IDENTITY = (
    "7cfa9ede873ae4fb34d7821472c62bba540f1b509476073062d52b487995cf65"
)
EXPECTED_MALOM_IDENTITY = (
    "f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747"
)


class MillBonusProbeError(RuntimeError):
    """A frozen input or no-update invariant differs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_canonical_audit(
    path: Path,
    *,
    expected_sha256: str,
    expected_identity: str,
    expected_schema: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise MillBonusProbeError(f"frozen audit hash differs: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MillBonusProbeError(f"frozen audit is invalid JSON: {path}") from exc
    if raw != canonical_json_bytes(value):
        raise MillBonusProbeError(f"frozen audit is not canonical JSON: {path}")
    identity = value.pop("audit_identity", None)
    if (
        identity != expected_identity
        or canonical_sha256(value) != identity
        or value.get("schema_version") != expected_schema
    ):
        raise MillBonusProbeError(f"frozen audit identity differs: {path}")
    return {**value, "audit_identity": identity}


def _mill_count(board: BoardState, color: str) -> int:
    return sum(
        1
        for mill in MILLS
        if all(board.positions.get(point) == color for point in mill)
    )


def probe_reward_transition(
    *,
    board: BoardState,
    move: Mapping[str, Any],
    malom_quality: float,
    expected_after_fen: str,
) -> dict[str, Any]:
    """Apply one frozen complete turn and compare only reward-mode output."""
    normalized_move = {
        "from": move.get("from"),
        "to": move.get("to"),
        "capture": move.get("capture"),
    }
    if normalized_move not in get_all_legal_moves(board):
        raise MillBonusProbeError("frozen probe move is not a complete legal turn")
    before_fen = board.to_fen_string()
    after = board.apply_move(normalized_move)
    if after.to_fen_string() != expected_after_fen:
        raise MillBonusProbeError("frozen probe successor FEN differs")
    mills_formed = max(
        0,
        _mill_count(after, board.turn) - _mill_count(board, board.turn),
    )
    rewards = {
        mode: _mill_formation_reward(
            mills_formed=mills_formed,
            malom_quality=malom_quality,
            mode=mode,
        )
        for mode in (
            "legacy-unconditional",
            "malom-preserving-only",
            "disabled",
        )
    }
    if board.to_fen_string() != before_fen:
        raise MillBonusProbeError("reward probe mutated its input board")
    return {
        "after_fen": after.to_fen_string(),
        "before_fen": before_fen,
        "malom_quality": float(malom_quality),
        "mills_formed": mills_formed,
        "move": normalized_move,
        "move_identity": canonical_sha256(normalized_move),
        "rewards": rewards,
    }


def _validate_auditor(auditor: Mapping[str, Any]) -> None:
    required = {
        "implementation_commit",
        "implementation_tree",
        "module_sha256",
        "script_sha256",
        "tracked_worktree_clean",
    }
    if set(auditor) != required or auditor.get("tracked_worktree_clean") is not True:
        raise MillBonusProbeError("probe implementation identity differs")
    if any(
        not isinstance(auditor.get(field), str) or not auditor[field]
        for field in required - {"tracked_worktree_clean"}
    ):
        raise MillBonusProbeError("probe implementation identity is incomplete")


def build_mill_bonus_no_update_probe(
    *,
    wdl_audit_path: Path,
    oracle_audit_path: Path,
    malom_path: Path,
    malom_manifest_path: Path,
    auditor: Mapping[str, Any],
    query_factory: Callable[[Path], Any] = lambda path: ExternalSolvedDB(
        str(path), strict=True
    ),
) -> dict[str, Any]:
    """Probe all 19 frozen downgrade turns without a model or update."""
    _validate_auditor(auditor)
    wdl_audit = _load_canonical_audit(
        wdl_audit_path,
        expected_sha256=EXPECTED_WDL_AUDIT_SHA256,
        expected_identity=EXPECTED_WDL_AUDIT_IDENTITY,
        expected_schema=HELDOUT_WDL_AUDIT_SCHEMA,
    )
    oracle_audit = _load_canonical_audit(
        oracle_audit_path,
        expected_sha256=EXPECTED_ORACLE_AUDIT_SHA256,
        expected_identity=EXPECTED_ORACLE_AUDIT_IDENTITY,
        expected_schema=HELDOUT_ORACLE_ALTERNATIVE_SCHEMA,
    )
    if any(
        wdl_audit.get(field) != oracle_audit.get(field)
        for field in (
            "evaluation_id",
            "ledger_sha256",
            "ledger_tail_record_sha256",
            "report_identity",
            "spec_identity",
        )
    ):
        raise MillBonusProbeError("frozen held-out audit bindings differ")

    wdl_rows = sorted(
        (
            row
            for row in wdl_audit["per_game"]
            if row["cohort"] == "candidate_loss"
            and row["first_candidate_wdl_downgrade"] is not None
        ),
        key=lambda row: int(row["ordinal"]),
    )
    oracle_rows = {
        int(row["ordinal"]): row for row in oracle_audit["per_state"]
    }
    if len(wdl_rows) != 19 or len(oracle_rows) != 19:
        raise MillBonusProbeError("frozen probe does not contain 19 states")

    manifest = load_dataset_manifest(malom_manifest_path)
    if (
        manifest.manifest_sha256 != EXPECTED_MALOM_IDENTITY
        or manifest.logical_name != "malom_tablebase"
        or manifest.trust_level != CURRENT_MALOM_LABEL_VERSION
    ):
        raise MillBonusProbeError("corrected Malom manifest differs")
    snapshot = verify_dataset_snapshot(malom_path, manifest)

    database = query_factory(malom_path)
    per_state: list[dict[str, Any]] = []
    try:
        if not database.is_available():
            raise MillBonusProbeError("corrected Malom is unavailable")
        for source_row in wdl_rows:
            ordinal = int(source_row["ordinal"])
            oracle_row = oracle_rows.get(ordinal)
            if oracle_row is None:
                raise MillBonusProbeError("oracle probe row is missing")
            downgrade = source_row["first_candidate_wdl_downgrade"]
            move = dict(downgrade["move"])
            if (
                oracle_row["game_id"] != source_row["game_id"]
                or oracle_row["chosen"]["move"] != move
                or oracle_row["turn_identity"] != downgrade["turn_identity"]
            ):
                raise MillBonusProbeError("frozen oracle and WDL rows differ")
            board = BoardState.from_fen_string(downgrade["before_fen"])
            quality = database.query_move_quality(board, move)
            if quality is None or float(quality) != float(downgrade["delta"]):
                raise MillBonusProbeError("live corrected Malom quality differs")
            transition = probe_reward_transition(
                board=board,
                move=move,
                malom_quality=float(quality),
                expected_after_fen=downgrade["after_fen"],
            )
            per_state.append(
                {
                    "candidate_color": source_row["candidate_color"],
                    "downgrade_transition": (
                        f'{downgrade["before_candidate_wdl"]}'
                        f'->{downgrade["after_candidate_wdl"]}'
                    ),
                    "game_id": source_row["game_id"],
                    "ordinal": ordinal,
                    "phase": downgrade["phase"],
                    "source_core_id": source_row["source_core_id"],
                    "stratum": source_row["stratum"],
                    "turn_identity": downgrade["turn_identity"],
                    **transition,
                }
            )
    finally:
        database.close()

    body = {
        "schema_version": MILL_BONUS_NO_UPDATE_PROBE_SCHEMA,
        "evaluation_id": wdl_audit["evaluation_id"],
        "spec_identity": wdl_audit["spec_identity"],
        "ledger_sha256": wdl_audit["ledger_sha256"],
        "ledger_tail_record_sha256": wdl_audit["ledger_tail_record_sha256"],
        "report_identity": wdl_audit["report_identity"],
        "sources": {
            "wdl_audit": {
                "audit_identity": wdl_audit["audit_identity"],
                "file_sha256": EXPECTED_WDL_AUDIT_SHA256,
            },
            "oracle_audit": {
                "audit_identity": oracle_audit["audit_identity"],
                "file_sha256": EXPECTED_ORACLE_AUDIT_SHA256,
            },
        },
        "auditor": dict(auditor),
        "malom": {
            "component_count": snapshot["component_count"],
            "identity": manifest.manifest_sha256,
            "size_bytes": snapshot["size_bytes"],
            "trust_level": manifest.trust_level,
        },
        "summary": {
            "states": len(per_state),
            "phase_counts": dict(
                sorted(Counter(row["phase"] for row in per_state).items())
            ),
            "stratum_counts": dict(
                sorted(Counter(row["stratum"] for row in per_state).items())
            ),
            "mill_forming_states": sum(
                row["mills_formed"] > 0 for row in per_state
            ),
            "legacy_reward_total": sum(
                row["rewards"]["legacy-unconditional"] for row in per_state
            ),
            "preserving_only_reward_total": sum(
                row["rewards"]["malom-preserving-only"] for row in per_state
            ),
            "disabled_reward_total": sum(
                row["rewards"]["disabled"] for row in per_state
            ),
        },
        "per_state": per_state,
        "claim_boundary": {
            "candidate_policy_loaded": False,
            "candidate_requeried": False,
            "new_games": False,
            "optimizer_created": False,
            "weights_updated": False,
            "actions_changed_between_modes": False,
            "states_changed_between_modes": False,
            "reward_component_only": True,
            "causal_training_effect_proven": False,
        },
    }
    return {**body, "probe_identity": canonical_sha256(body)}


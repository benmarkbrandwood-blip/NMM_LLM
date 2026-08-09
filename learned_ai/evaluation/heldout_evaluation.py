"""Frozen Sanmill-refereed held-out evaluation and evidence ledger.

This module is deliberately separate from :mod:`paired_protocol`.  The older
Stage-0 runner uses the local GameEngine, model bundles on both sides, and a
max-ply draw.  The retained-v2 protocol instead requires a training-aligned
candidate, fixed-node Sanmill search, complete-history Sanmill adjudication,
and fail-closed handling of every non-rules termination.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.heldout_exposure import validate_executable_corpus
from learned_ai.evaluation.sanmill_uci import EXPECTED_RULES_IDENTITY_SHA256
from learned_ai.evaluation.training_aligned_policy import (
    TrainingAlignedPolicy,
    load_training_aligned_policy,
)
from learned_ai.training.checkpoint_envelope import load_checkpoint
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.training.sanmill_referee import (
    TRAINING_REFEREE_SEMANTIC_DIGEST,
    TRAINING_SANMILL_BINARY_SHA256,
    TRAINING_SANMILL_COMMIT,
    TRAINING_SANMILL_TREE,
    SanmillAppliedTurn,
    SanmillTrainingGame,
    inspect_sanmill_training_installation,
    nmm_move_actions,
    probe_sanmill_training_runtime,
    training_installation_record,
)
from learned_ai.training.training_identity import (
    TRAINER_RULESET_SEMANTIC_DIGEST,
    load_trainer_ruleset,
    mif_release_identity,
)


HELDOUT_SPEC_SCHEMA = "nmm.sanmill-heldout-runtime-spec.v1"
HELDOUT_GAME_SCHEMA = "nmm.sanmill-heldout-game.v1"
HELDOUT_REPORT_SCHEMA = "nmm.sanmill-heldout-result.v1"
HELDOUT_READINESS_SCHEMA = "nmm.sanmill-heldout-readiness.v1"
HELDOUT_LAUNCH_SCHEMA = "nmm.sanmill-heldout-launch.v1"
HELDOUT_PROGRESS_SCHEMA = "nmm.sanmill-heldout-progress.v1"
HELDOUT_FAILURE_SCHEMA = "nmm.sanmill-heldout-failure.v1"

ATOMIC_REPLACE_PERMISSION_RETRIES = 80
ATOMIC_REPLACE_RETRY_SECONDS = 0.025

PLAN_RELATIVE = Path(
    "docs/experiments/sanmill-corrected-retained-v2-heldout-eval-v1.json"
)
AUTHORIZATION_RELATIVE = Path(
    "docs/experiments/sanmill-corrected-retained-v2-heldout-eval-v1-authorization.json"
)
SPECIALIST_DB_RELATIVE = Path("data/specialist_db.sanmill_corrected_retained_v2.sqlite")
MALOM_MANIFEST_RELATIVE = Path("data/manifests/malom-sector-corrected-v1.json")
RULESET_RELATIVE = Path("data/rulesets/nmm-training-core@2.json")

EXPECTED_PLAN_FILE_SHA256 = (
    "06f168d1687557a9146455fae0a8174c7714b7dd864cfd5a1e2c383c26009b21"
)
EXPECTED_PLAN_IDENTITY = (
    "212076e9423b671b83783efef411db3b4a56c8c67ae36a463d381d6939d4d982"
)
EXPECTED_PLAN_COMMIT = "106d015b23debee7d5c8d691195ff958da66f1fc"
EXPECTED_AUTHORIZATION_FILE_SHA256 = (
    "36d3eae20970143b8fe600402e5d7d915bc9d2bf091b69e0b31c85ebfe98afaf"
)
EXPECTED_AUTHORIZATION_IDENTITY = (
    "6426ffd109d28145a4148855d70a181d50fd4277068fe01b501934d212378fb1"
)
EXPECTED_AUDIT_FILE_SHA256 = (
    "6ca9d040e55ed2fdabf1b6bf079c2f2164615fd15e818c99888390dee4de1678"
)
EXPECTED_AUDIT_IDENTITY = (
    "df5b04128e4cf21f5325e0601596f0fe74b8f54fb6708c6dfd2c6b79fffdc21e"
)
EXPECTED_CORPUS_FILE_SHA256 = (
    "3bcf9db2d003d10769b88767763eb7dfb950eecbff578b7c7ff7d1c208e19771"
)
EXPECTED_CORPUS_IDENTITY = (
    "417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9"
)
EXPECTED_RECORDS_IDENTITY = (
    "e8a1828cb1d7e0e86c686d934e87934c6c12e6a8cf7610974ed8035937ab8cff"
)
EXPECTED_STRICT_SUBSET_IDENTITY = (
    "a01be0c72b395f2a624c2f5ae7538d9d08eaccde0a392dba566bebe2221806f8"
)
EXPECTED_EVALUATION_ID = "dev-v4-sanmill-corrected-retained-v2-heldout-v1"
EXPECTED_BUNDLE_IDENTITY = (
    "c2652119b64a2808ebcd5e7dc661873f3f897065b7d529bd9e261328f0981f23"
)
EXPECTED_CHECKPOINT_FILE_SHA256 = (
    "df00861a5ced53b6c9b16ed89f2762d41a82f1d74fce970b5d0bdf6adba4ac4d"
)
EXPECTED_CHECKPOINT_PAYLOAD_SHA256 = (
    "8b4017ce856012fa3c4d578c56c5f32a6d5ebae97b9f17c6cbd2c5228146de19"
)
EXPECTED_CHECKPOINT_ID = (
    "managed-sanmill-corrected-retained-v2-segment-0020:checkpoint:00000006"
)
EXPECTED_HUMAN_DB_IDENTITY = (
    "8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31"
)
EXPECTED_SPECIALIST_DB_IDENTITY = (
    "ea2df42d6df837588e1a2d87e37bd025c2b612f87695aa9ae16da064aebf62a8"
)
EXPECTED_MALOM_IDENTITY = (
    "f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GAME_FIELDS = {
    "schema_version",
    "spec_identity",
    "ordinal",
    "pair_index",
    "game_in_pair",
    "game_id",
    "source_core_id",
    "stratum",
    "strict_independence_sensitivity",
    "candidate_color",
    "candidate_score",
    "winner",
    "outcome_reason",
    "prefix",
    "post_prefix_logical_plies",
    "final_state",
    "turns",
    "search_summary",
    "game_elapsed_seconds",
    "cumulative_active_seconds",
    "complete",
    "previous_record_sha256",
}
_TURN_FIELDS = {
    "post_prefix_logical_ply",
    "mover_color",
    "actor",
    "move",
    "actions",
    "before_history_sha256",
    "after_history_sha256",
    "logical_ply_count",
    "local_fen_after",
    "sanmill_fen_after",
    "terminal",
    "outcome_reason",
    "search",
}
_PREFIX_FIELDS = {
    "prefix_identity",
    "expected_history_sha256",
    "observed_history_sha256",
    "action_token_count",
    "logical_ply_count",
    "logical_plies_by_side",
    "final_nmm_fen",
    "final_sanmill_fen",
}
_MOVE_FIELDS = {"from", "to", "capture"}
_SEARCH_FIELDS = {
    "status",
    "full_turn_actions",
    "logical_move_id",
    "model_action",
    "logical_ply_delta",
    "resulting_fen",
    "resulting_side_to_move",
    "terminal",
    "winner",
    "winner_code",
    "outcome_reason",
    "effective_depth",
    "completed_depth",
    "score_kind",
    "score",
    "score_perspective",
    "node_budget",
    "primary_nodes",
    "removal_nodes",
    "total_nodes",
    "search_calls",
}
_SEARCH_SUMMARY_FIELDS = {
    "turns",
    "node_ceiling",
    "total_nodes",
    "min_nodes",
    "max_nodes",
    "min_completed_depth",
    "max_completed_depth",
}
_FINAL_STATE_FIELDS = {
    "status",
    "ruleset_id",
    "rules_identity_sha256",
    "history_origin",
    "fen",
    "side_to_move",
    "phase",
    "action",
    "terminal",
    "removal_pending",
    "pending_removal_count",
    "pending_removals",
    "legal_actions",
    "action_token_count",
    "logical_ply_count",
    "logical_plies_by_side",
    "no_capture_count",
    "repetition_current_count",
    "repetition_history_length",
    "snapshot_history_length",
    "history_sha256",
    "outcome",
    "strict_referee_identity",
}
_OUTCOME_FIELDS = {
    "terminal",
    "winner",
    "winner_code",
    "reason",
    "reason_code",
}
_STRICT_REFEREE_FIELDS = {
    "format",
    "profile",
    "repetitionObservation",
    "originCounted",
    "semanticDigest",
}


class HeldoutEvaluationError(RuntimeError):
    """Raised when the held-out contract or evidence fails closed."""


class HeldoutEvaluationInvalid(HeldoutEvaluationError):
    """Raised when an authorized run becomes invalid, never a game loss."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HeldoutEvaluationError(
                    f"duplicate JSON key {key!r} in {path.name}"
                )
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8", newline="") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeldoutEvaluationError(f"cannot read {path.name}") from exc
    if not isinstance(value, dict):
        raise HeldoutEvaluationError(f"{path.name} must contain one object")
    return value


def _identity(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    observed = body.pop(field, None)
    if not isinstance(observed, str) or canonical_sha256(body) != observed:
        raise HeldoutEvaluationError(f"{field} is missing or inconsistent")
    return observed


def _require_file_hash(path: Path, expected: str, *, name: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        raise HeldoutEvaluationError(f"{name} file identity differs from the pin")


def _repo_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise HeldoutEvaluationError(f"{field} is not a path")
    relative = Path(value)
    if relative.is_absolute():
        raise HeldoutEvaluationError(f"{field} must be repository-relative")
    resolved = (_REPO_ROOT / relative).resolve(strict=False)
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise HeldoutEvaluationError(f"{field} leaves the repository") from exc
    return resolved


def _resolve_local_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HeldoutEvaluationError(f"local path {field} is not configured")
    path = Path(os.path.expandvars(os.path.expanduser(value.strip())))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve(strict=False)


@dataclass(frozen=True)
class FrozenHeldoutContract:
    plan_path: Path
    authorization_path: Path
    plan: Mapping[str, Any]
    authorization: Mapping[str, Any]
    corpus: Mapping[str, Any]
    audit: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    strict_source_ids: frozenset[str]

    @property
    def plan_identity(self) -> str:
        return str(self.plan["plan_identity"])

    @property
    def authorization_identity(self) -> str:
        return str(self.authorization["authorization_identity"])


@dataclass(frozen=True)
class HeldoutPaths:
    paths_config: Path
    candidate_bundle: Path
    checkpoint: Path
    corpus: Path
    exposure_audit: Path
    human_db: Path
    specialist_db: Path
    malom_db: Path
    malom_manifest: Path
    ruleset_manifest: Path
    sanmill_checkout: Path
    output_root: Path
    output_plan: Path
    output_authorization: Path
    output_spec: Path
    output_ledger: Path
    output_report: Path


def load_frozen_heldout_contract(
    plan_path: str | Path = _REPO_ROOT / PLAN_RELATIVE,
    authorization_path: str | Path = _REPO_ROOT / AUTHORIZATION_RELATIVE,
) -> FrozenHeldoutContract:
    """Load the exact preregistered plan, grant, corpus, and exposure ledger."""
    plan_file = Path(plan_path)
    authorization_file = Path(authorization_path)
    _require_file_hash(plan_file, EXPECTED_PLAN_FILE_SHA256, name="held-out plan")
    _require_file_hash(
        authorization_file,
        EXPECTED_AUTHORIZATION_FILE_SHA256,
        name="held-out authorization",
    )
    plan = _strict_json(plan_file)
    authorization = _strict_json(authorization_file)
    if _identity(plan, "plan_identity") != EXPECTED_PLAN_IDENTITY:
        raise HeldoutEvaluationError("held-out plan identity differs from the pin")
    if (
        _identity(authorization, "authorization_identity")
        != EXPECTED_AUTHORIZATION_IDENTITY
    ):
        raise HeldoutEvaluationError(
            "held-out authorization identity differs from the pin"
        )
    if plan.get("evaluation_id") != EXPECTED_EVALUATION_ID:
        raise HeldoutEvaluationError("held-out evaluation ID differs")
    if plan.get("status") != "frozen_awaiting_runner_and_final_preflight":
        raise HeldoutEvaluationError("held-out plan status differs")
    if plan.get("plan_identity") != authorization.get("plan", {}).get("identity"):
        raise HeldoutEvaluationError("authorization does not bind the plan")
    expected_plan_ref = {
        "commit": EXPECTED_PLAN_COMMIT,
        "file_sha256": EXPECTED_PLAN_FILE_SHA256,
        "identity": EXPECTED_PLAN_IDENTITY,
        "plan_id": "sanmill-corrected-retained-v2-heldout-eval-v1",
        "tracked_file": PLAN_RELATIVE.as_posix(),
    }
    if authorization.get("plan") != expected_plan_ref:
        raise HeldoutEvaluationError("authorization plan binding differs")
    if authorization.get("consumption", {}).get("grant_count") != 1:
        raise HeldoutEvaluationError("authorization is not a one-run grant")
    if (
        authorization.get("execution_scope", {}).get(
            "launch_without_further_product_confirmation_after_all_gates_pass"
        )
        is not True
    ):
        raise HeldoutEvaluationError("authorization does not permit gated launch")
    if any(
        authorization.get("claim_boundary", {}).get(field) is not False
        for field in ("new_training", "model_promotion", "model_publication")
    ):
        raise HeldoutEvaluationError("authorization claim boundary differs")

    workload = plan.get("workload")
    if workload != {
        "games": 128,
        "max_active_hours": 6.0,
        "pairs": 64,
        "safe_exact_resume_same_spec": True,
        "unique_starts": 64,
    }:
        raise HeldoutEvaluationError("held-out workload differs")
    if plan.get("baseline", {}).get("fixed_node_ceiling_per_logical_turn") != 500_000:
        raise HeldoutEvaluationError("held-out node ceiling differs")
    if plan.get("protocol", {}).get("max_post_prefix_logical_plies") != 1536:
        raise HeldoutEvaluationError("held-out safety cap differs")
    if plan.get("protocol", {}).get("max_ply_disposition") != (
        "incomplete-invalid-not-draw"
    ):
        raise HeldoutEvaluationError("held-out safety disposition differs")
    if plan.get("candidate", {}).get("bundle", {}).get("identity") != (
        EXPECTED_BUNDLE_IDENTITY
    ):
        raise HeldoutEvaluationError("candidate bundle identity differs")

    corpus_path = _repo_path(plan["corpus"]["path"], field="corpus.path")
    audit_path = _repo_path(
        plan["corpus"]["strict_independence_audit"]["path"],
        field="corpus.strict_independence_audit.path",
    )
    _require_file_hash(corpus_path, EXPECTED_CORPUS_FILE_SHA256, name="held-out corpus")
    _require_file_hash(audit_path, EXPECTED_AUDIT_FILE_SHA256, name="exposure audit")
    corpus = _strict_json(corpus_path)
    records = validate_executable_corpus(
        corpus,
        expected_corpus_identity=EXPECTED_CORPUS_IDENTITY,
        expected_records_identity=EXPECTED_RECORDS_IDENTITY,
    )
    audit = _strict_json(audit_path)
    if _identity(audit, "audit_identity") != EXPECTED_AUDIT_IDENTITY:
        raise HeldoutEvaluationError("exposure audit identity differs")
    strict_ids = audit.get("strict_independence_source_core_ids")
    if not isinstance(strict_ids, list) or any(
        not isinstance(item, str) for item in strict_ids
    ):
        raise HeldoutEvaluationError("strict sensitivity membership is invalid")
    if canonical_sha256(strict_ids) != EXPECTED_STRICT_SUBSET_IDENTITY:
        raise HeldoutEvaluationError("strict sensitivity membership differs")
    if len(strict_ids) != 34 or len(set(strict_ids)) != 34:
        raise HeldoutEvaluationError("strict sensitivity size differs")
    record_ids = [str(record.get("source_core_id")) for record in records]
    if not set(strict_ids).issubset(record_ids):
        raise HeldoutEvaluationError("strict sensitivity member is absent")
    if plan["corpus"]["strict_independence_audit"] != {
        "audit_file_sha256": EXPECTED_AUDIT_FILE_SHA256,
        "audit_identity": EXPECTED_AUDIT_IDENTITY,
        "count": 34,
        "path": audit_path.relative_to(_REPO_ROOT).as_posix(),
        "strata": {"book": 13, "perfect_db": 21},
        "subset_identity": EXPECTED_STRICT_SUBSET_IDENTITY,
    }:
        raise HeldoutEvaluationError("plan exposure-audit binding differs")

    return FrozenHeldoutContract(
        plan_path=plan_file,
        authorization_path=authorization_file,
        plan=plan,
        authorization=authorization,
        corpus=corpus,
        audit=audit,
        records=tuple(records),
        strict_source_ids=frozenset(strict_ids),
    )


def resolve_heldout_paths(
    contract: FrozenHeldoutContract,
    paths_config: str | Path = _REPO_ROOT / "data/training_paths.local.json",
) -> HeldoutPaths:
    """Resolve machine-local inputs while keeping them out of tracked records."""
    config_path = Path(paths_config)
    config = _strict_json(config_path)
    human_db = _resolve_local_path(config.get("human_db_path"), field="human_db_path")
    malom_db = _resolve_local_path(config.get("malom_db_path"), field="malom_db_path")
    sanmill_checkout = _resolve_local_path(
        config.get("sanmill_training_checkout"),
        field="sanmill_training_checkout",
    )
    bundle = _repo_path(
        contract.plan["candidate"]["bundle"]["path"],
        field="candidate.bundle.path",
    )
    checkpoint = _repo_path(
        contract.plan["candidate"]["checkpoint"]["path"],
        field="candidate.checkpoint.path",
    )
    corpus = _repo_path(contract.plan["corpus"]["path"], field="corpus.path")
    audit = _repo_path(
        contract.plan["corpus"]["strict_independence_audit"]["path"],
        field="corpus.strict_independence_audit.path",
    )
    outputs = contract.plan["outputs"]
    output_plan = _repo_path(outputs["plan"], field="outputs.plan")
    output_root = output_plan.parent
    resolved_outputs = {
        key: _repo_path(value, field=f"outputs.{key}") for key, value in outputs.items()
    }
    if any(path.parent != output_root for path in resolved_outputs.values()):
        raise HeldoutEvaluationError("held-out outputs do not share one root")
    if bundle.parent != output_root:
        raise HeldoutEvaluationError("candidate bundle is outside the output root")
    return HeldoutPaths(
        paths_config=config_path,
        candidate_bundle=bundle,
        checkpoint=checkpoint,
        corpus=corpus,
        exposure_audit=audit,
        human_db=human_db,
        specialist_db=(_REPO_ROOT / SPECIALIST_DB_RELATIVE).resolve(),
        malom_db=malom_db,
        malom_manifest=(_REPO_ROOT / MALOM_MANIFEST_RELATIVE).resolve(),
        ruleset_manifest=(_REPO_ROOT / RULESET_RELATIVE).resolve(),
        sanmill_checkout=sanmill_checkout,
        output_root=output_root,
        output_plan=resolved_outputs["plan"],
        output_authorization=resolved_outputs["authorization"],
        output_spec=resolved_outputs["specification"],
        output_ledger=resolved_outputs["ledger"],
        output_report=resolved_outputs["report"],
    )


def _matching_move(
    board: BoardState,
    actions: Sequence[str],
) -> dict[str, Any]:
    matches = [
        dict(move)
        for move in get_all_legal_moves(board)
        if nmm_move_actions(move) == tuple(actions)
    ]
    if len(matches) != 1:
        raise HeldoutEvaluationError(
            "prefix actions do not select one legal NMM logical turn"
        )
    return matches[0]


def replay_frozen_prefix(
    game: SanmillTrainingGame,
    record: Mapping[str, Any],
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[BoardState, dict[str, Any]]:
    """Replay one frozen twelve-ply history through the strict referee."""
    execution = record.get("execution_record")
    if not isinstance(execution, Mapping):
        raise HeldoutEvaluationError("corpus record lacks execution data")
    steps = execution.get("steps")
    if not isinstance(steps, list) or len(steps) != 12:
        raise HeldoutEvaluationError("corpus prefix is not twelve logical plies")
    board = BoardState.new_game()
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("logical_ply") != index:
            raise HeldoutEvaluationError("corpus prefix step order differs")
        actions = step.get("action_tokens")
        if not isinstance(actions, list) or any(
            not isinstance(item, str) for item in actions
        ):
            raise HeldoutEvaluationError("corpus prefix actions are invalid")
        move = _matching_move(board, actions)
        game.apply_nmm_move(board, move)
        board = board.apply_move(move)
        if progress is not None:
            progress(index + 1)

    final = execution.get("final")
    if not isinstance(final, Mapping):
        raise HeldoutEvaluationError("corpus prefix has no final state")
    if board.to_fen_string() != final.get("nmm_fen"):
        raise HeldoutEvaluationError("prefix local final FEN differs")
    if game.state.logical_ply_count != 12:
        raise HeldoutEvaluationError("prefix referee logical count differs")
    if game.state.logical_plies_by_side != (6, 6):
        raise HeldoutEvaluationError("prefix per-side logical count differs")
    if game.state.action_token_count != len(execution.get("action_tokens", [])):
        raise HeldoutEvaluationError("prefix action-token count differs")
    if game.state.history_sha256 != final.get("history_sha256"):
        raise HeldoutEvaluationError("prefix history SHA-256 differs")
    if game.state.terminal:
        raise HeldoutEvaluationError("held-out prefix is already terminal")
    game.assert_current_board(board)
    return board, {
        "prefix_identity": execution.get("prefix_identity"),
        "expected_history_sha256": final.get("history_sha256"),
        "observed_history_sha256": game.state.history_sha256,
        "action_token_count": game.state.action_token_count,
        "logical_ply_count": game.state.logical_ply_count,
        "logical_plies_by_side": list(game.state.logical_plies_by_side),
        "final_nmm_fen": board.to_fen_string(),
        "final_sanmill_fen": game.state.fen,
    }


def audit_frozen_prefixes(
    contract: FrozenHeldoutContract,
    installation: Any,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Replay every start once without loading or querying the candidate."""
    observations = []
    for record in contract.records:
        with SanmillTrainingGame(installation, seed=seed) as game:
            _board, prefix = replay_frozen_prefix(game, record)
        observations.append(
            {
                "source_core_id": record["source_core_id"],
                "prefix_identity": prefix["prefix_identity"],
                "history_sha256": prefix["observed_history_sha256"],
            }
        )
    return {
        "records": len(observations),
        "fresh_processes": len(observations),
        "candidate_loaded": False,
        "games_played": 0,
        "observations_identity": canonical_sha256(observations),
    }


def _game_schedule(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    schedule = spec.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 128:
        raise HeldoutEvaluationError("runtime schedule must contain 128 games")
    result: list[dict[str, Any]] = []
    for ordinal, item in enumerate(schedule):
        if not isinstance(item, dict):
            raise HeldoutEvaluationError("runtime schedule member is invalid")
        expected_keys = {
            "ordinal",
            "pair_index",
            "game_in_pair",
            "game_id",
            "source_core_id",
            "stratum",
            "strict_independence_sensitivity",
            "candidate_color",
            "prefix_identity",
            "expected_prefix_history_sha256",
        }
        if set(item) != expected_keys or item["ordinal"] != ordinal:
            raise HeldoutEvaluationError("runtime schedule order differs")
        if item["pair_index"] != ordinal // 2:
            raise HeldoutEvaluationError("runtime pair index differs")
        game_in_pair = ordinal % 2
        if item["game_in_pair"] != game_in_pair:
            raise HeldoutEvaluationError("runtime game-in-pair differs")
        if item["candidate_color"] != ("W" if game_in_pair == 0 else "B"):
            raise HeldoutEvaluationError("runtime color role differs")
        result.append(item)
    return result


def _runtime_spec_body(
    contract: FrozenHeldoutContract,
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    git_gate = next(
        (
            gate
            for gate in readiness.get("gates", [])
            if gate.get("gate") == "repository"
        ),
        None,
    )
    if not isinstance(git_gate, Mapping) or git_gate.get("result") != "pass":
        raise HeldoutEvaluationError("runtime spec requires a passing Git gate")
    observed_git = git_gate.get("observed")
    if not isinstance(observed_git, Mapping):
        raise HeldoutEvaluationError("runtime Git evidence is missing")
    schedule = []
    for pair_index, record in enumerate(contract.records):
        execution = record["execution_record"]
        final = execution["final"]
        for game_in_pair, color in enumerate(("W", "B")):
            ordinal = pair_index * 2 + game_in_pair
            identity_body = {
                "plan_identity": contract.plan_identity,
                "pair_index": pair_index,
                "game_in_pair": game_in_pair,
                "source_core_id": record["source_core_id"],
                "candidate_color": color,
            }
            schedule.append(
                {
                    "ordinal": ordinal,
                    "pair_index": pair_index,
                    "game_in_pair": game_in_pair,
                    "game_id": "heldout-game:" + canonical_sha256(identity_body),
                    "source_core_id": record["source_core_id"],
                    "stratum": record["stratum"],
                    "strict_independence_sensitivity": (
                        record["source_core_id"] in contract.strict_source_ids
                    ),
                    "candidate_color": color,
                    "prefix_identity": execution["prefix_identity"],
                    "expected_prefix_history_sha256": final["history_sha256"],
                }
            )
    return {
        "schema_version": HELDOUT_SPEC_SCHEMA,
        "evaluation_id": EXPECTED_EVALUATION_ID,
        "plan": {
            "commit": EXPECTED_PLAN_COMMIT,
            "identity": contract.plan_identity,
            "file_sha256": EXPECTED_PLAN_FILE_SHA256,
        },
        "authorization": {
            "identity": contract.authorization_identity,
            "file_sha256": EXPECTED_AUTHORIZATION_FILE_SHA256,
            "grant_count": 1,
        },
        "implementation": {
            "branch": observed_git["branch"],
            "commit": observed_git["head"],
            "tree": observed_git["tree"],
            "upstream_commit": observed_git["upstream_commit"],
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "precision": "float32",
            "seed": 42,
        },
        "candidate": dict(contract.plan["candidate"]),
        "baseline": dict(contract.plan["baseline"]),
        "corpus": dict(contract.plan["corpus"]),
        "rules": dict(contract.plan["rules"]),
        "analysis": dict(contract.plan["analysis"]),
        "workload": dict(contract.plan["workload"]),
        "protocol": dict(contract.plan["protocol"]),
        "readiness_identity": readiness["readiness_identity"],
        "schedule": schedule,
    }


def build_runtime_spec(
    contract: FrozenHeldoutContract,
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    body = _runtime_spec_body(contract, readiness)
    return {**body, "spec_identity": canonical_sha256(body)}


def load_runtime_spec(path: str | Path) -> dict[str, Any]:
    spec = _strict_json(Path(path))
    identity = _identity(spec, "spec_identity")
    if spec.get("schema_version") != HELDOUT_SPEC_SCHEMA:
        raise HeldoutEvaluationError("runtime spec schema differs")
    if spec.get("evaluation_id") != EXPECTED_EVALUATION_ID:
        raise HeldoutEvaluationError("runtime spec evaluation differs")
    _game_schedule(spec)
    if spec["spec_identity"] != identity:
        raise HeldoutEvaluationError("runtime spec identity differs")
    return spec


def _expected_game(spec: Mapping[str, Any], ordinal: int) -> Mapping[str, Any]:
    return _game_schedule(spec)[ordinal]


def _sha256_text(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HeldoutEvaluationError(f"{field} is not a lowercase SHA-256")
    return value


def _validate_search(
    search: Mapping[str, Any],
    *,
    turn: Mapping[str, Any],
) -> None:
    if set(search) != _SEARCH_FIELDS:
        raise HeldoutEvaluationError("Sanmill search fields differ")
    if search["status"] != "ok":
        raise HeldoutEvaluationError("Sanmill search status differs")
    if search["full_turn_actions"] != turn["actions"]:
        raise HeldoutEvaluationError("Sanmill search actions differ")
    if search["model_action"] != turn["move"]:
        raise HeldoutEvaluationError("Sanmill search move differs")
    if search["logical_ply_delta"] != 1:
        raise HeldoutEvaluationError("Sanmill search logical-ply delta differs")
    if search["resulting_fen"] != turn["sanmill_fen_after"]:
        raise HeldoutEvaluationError("Sanmill search resulting FEN differs")
    if search["terminal"] is not turn["terminal"]:
        raise HeldoutEvaluationError("Sanmill search terminal state differs")
    if search["outcome_reason"] != turn["outcome_reason"]:
        raise HeldoutEvaluationError("Sanmill search outcome reason differs")
    integers = (
        "node_budget",
        "primary_nodes",
        "removal_nodes",
        "total_nodes",
        "search_calls",
    )
    if any(
        not isinstance(search[field], int)
        or isinstance(search[field], bool)
        or search[field] < 0
        for field in integers
    ):
        raise HeldoutEvaluationError("Sanmill search counters are invalid")
    if search["node_budget"] != 500_000:
        raise HeldoutEvaluationError("Sanmill search node ceiling differs")
    if search["primary_nodes"] + search["removal_nodes"] != search["total_nodes"]:
        raise HeldoutEvaluationError("Sanmill search node accounting differs")
    if search["total_nodes"] > search["node_budget"]:
        raise HeldoutEvaluationError("Sanmill search exceeded its node ceiling")
    if search["search_calls"] <= 0:
        raise HeldoutEvaluationError("Sanmill search has no search call")
    for field in ("effective_depth", "completed_depth"):
        value = search[field]
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise HeldoutEvaluationError("Sanmill search depth is invalid")


def _validate_turns(record: Mapping[str, Any]) -> None:
    turns = record.get("turns")
    if not isinstance(turns, list) or len(turns) != record.get(
        "post_prefix_logical_plies"
    ):
        raise HeldoutEvaluationError("game turn ledger length differs")
    previous = _sha256_text(
        record["prefix"]["observed_history_sha256"],
        field="prefix history",
    )
    candidate_color = record["candidate_color"]
    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, dict) or set(turn) != _TURN_FIELDS:
            raise HeldoutEvaluationError("game turn fields differ")
        if turn["post_prefix_logical_ply"] != index:
            raise HeldoutEvaluationError("game turn order differs")
        mover = "W" if index % 2 == 1 else "B"
        if turn["mover_color"] != mover:
            raise HeldoutEvaluationError("game mover order differs")
        expected_actor = "candidate" if mover == candidate_color else "sanmill"
        if turn["actor"] != expected_actor:
            raise HeldoutEvaluationError("game actor and color role differ")
        move = turn["move"]
        if not isinstance(move, dict) or set(move) != _MOVE_FIELDS:
            raise HeldoutEvaluationError("game move fields differ")
        actions = turn["actions"]
        if (
            not isinstance(actions, list)
            or not actions
            or any(not isinstance(action, str) for action in actions)
            or list(nmm_move_actions(move)) != actions
        ):
            raise HeldoutEvaluationError("game logical-turn actions differ")
        if turn["before_history_sha256"] != previous:
            raise HeldoutEvaluationError("game history chain differs")
        previous = _sha256_text(turn["after_history_sha256"], field="turn history")
        if turn["logical_ply_count"] != 12 + index:
            raise HeldoutEvaluationError("game logical-ply count differs")
        terminal = turn["terminal"]
        if not isinstance(terminal, bool):
            raise HeldoutEvaluationError("game terminal flag is invalid")
        is_last = index == len(turns)
        if terminal is not is_last:
            raise HeldoutEvaluationError("only the final game turn may terminate")
        if terminal:
            if turn["outcome_reason"] != record["outcome_reason"]:
                raise HeldoutEvaluationError("terminal turn reason differs")
        elif turn["outcome_reason"] != "ongoing":
            raise HeldoutEvaluationError("ongoing turn has a terminal reason")
        if not isinstance(turn["local_fen_after"], str) or not isinstance(
            turn["sanmill_fen_after"], str
        ):
            raise HeldoutEvaluationError("game turn FEN is invalid")
        if expected_actor == "candidate" and turn["search"] is not None:
            raise HeldoutEvaluationError("candidate turn contains baseline search")
        if expected_actor == "sanmill":
            search = turn["search"]
            if not isinstance(search, dict):
                raise HeldoutEvaluationError("Sanmill turn lacks search evidence")
            _validate_search(search, turn=turn)
            expected_search_winner = (
                {None: None, "W": "white", "B": "black"}[record["winner"]]
                if terminal
                else None
            )
            expected_winner_code = (
                {None: None, "W": 0, "B": 1}[record["winner"]] if terminal else None
            )
            if (
                search["winner"] != expected_search_winner
                or search["winner_code"] != expected_winner_code
            ):
                raise HeldoutEvaluationError("Sanmill search winner differs")
    if previous != record["final_state"]["history_sha256"]:
        raise HeldoutEvaluationError("game final history differs from turns")


def _validate_game_record(
    spec: Mapping[str, Any],
    record: Mapping[str, Any],
    ordinal: int,
    previous_hash: str | None,
) -> None:
    if set(record) != _GAME_FIELDS:
        raise HeldoutEvaluationError("game record fields are unknown or incomplete")
    if record["schema_version"] != HELDOUT_GAME_SCHEMA:
        raise HeldoutEvaluationError("game record schema differs")
    if record["spec_identity"] != spec["spec_identity"]:
        raise HeldoutEvaluationError("game record spec differs")
    expected = _expected_game(spec, ordinal)
    for key in (
        "ordinal",
        "pair_index",
        "game_in_pair",
        "game_id",
        "source_core_id",
        "stratum",
        "strict_independence_sensitivity",
        "candidate_color",
    ):
        if record[key] != expected[key]:
            raise HeldoutEvaluationError(f"game record {key} differs")
    if record["previous_record_sha256"] != previous_hash:
        raise HeldoutEvaluationError("game record chain differs")
    if record["complete"] is not True:
        raise HeldoutEvaluationError("game record is incomplete")
    if record["candidate_score"] not in (0.0, 0.5, 1.0):
        raise HeldoutEvaluationError("game score is invalid")
    winner = record["winner"]
    if winner not in {None, "W", "B"}:
        raise HeldoutEvaluationError("game winner is invalid")
    color = record["candidate_color"]
    expected_score = 0.5 if winner is None else (1.0 if winner == color else 0.0)
    if record["candidate_score"] != expected_score:
        raise HeldoutEvaluationError("game score and winner disagree")
    if record["outcome_reason"] in {None, "ongoing", "max_ply"}:
        raise HeldoutEvaluationError("game has no rules terminal reason")
    if record["post_prefix_logical_plies"] <= 0:
        raise HeldoutEvaluationError("game has no played logical turn")
    if record["post_prefix_logical_plies"] > int(
        spec["protocol"]["max_post_prefix_logical_plies"]
    ):
        raise HeldoutEvaluationError("game exceeds the safety ceiling")
    prefix = record["prefix"]
    if not isinstance(prefix, dict) or set(prefix) != _PREFIX_FIELDS:
        raise HeldoutEvaluationError("game prefix evidence is invalid")
    if prefix.get("prefix_identity") != expected["prefix_identity"]:
        raise HeldoutEvaluationError("game prefix identity differs")
    if (
        prefix.get("observed_history_sha256")
        != expected["expected_prefix_history_sha256"]
    ):
        raise HeldoutEvaluationError("game prefix history differs")
    if prefix.get("expected_history_sha256") != prefix.get("observed_history_sha256"):
        raise HeldoutEvaluationError("game expected prefix history differs")
    if prefix.get("logical_ply_count") != 12 or prefix.get("logical_plies_by_side") != [
        6,
        6,
    ]:
        raise HeldoutEvaluationError("game prefix logical counts differ")
    prefix_actions = prefix.get("action_token_count")
    if (
        not isinstance(prefix_actions, int)
        or isinstance(prefix_actions, bool)
        or prefix_actions < 12
    ):
        raise HeldoutEvaluationError("game prefix action count is invalid")
    if not isinstance(prefix.get("final_nmm_fen"), str) or not isinstance(
        prefix.get("final_sanmill_fen"), str
    ):
        raise HeldoutEvaluationError("game prefix FEN is invalid")
    final_state = record["final_state"]
    if not isinstance(final_state, dict) or set(final_state) != _FINAL_STATE_FIELDS:
        raise HeldoutEvaluationError("game final state is invalid")
    outcome = final_state.get("outcome")
    if (
        not isinstance(outcome, dict)
        or set(outcome) != _OUTCOME_FIELDS
        or outcome.get("terminal") is not True
    ):
        raise HeldoutEvaluationError("game final state is not terminal")
    strict_referee = final_state.get("strict_referee_identity")
    if (
        not isinstance(strict_referee, dict)
        or set(strict_referee) != _STRICT_REFEREE_FIELDS
        or strict_referee.get("semanticDigest") != TRAINING_REFEREE_SEMANTIC_DIGEST
    ):
        raise HeldoutEvaluationError("game strict-referee identity differs")
    expected_winner = {None: None, "W": "white", "B": "black"}[winner]
    if outcome.get("winner") != expected_winner:
        raise HeldoutEvaluationError("game final winner differs")
    if outcome.get("reason") != record["outcome_reason"]:
        raise HeldoutEvaluationError("game final reason differs")
    if final_state.get("terminal") is not True:
        raise HeldoutEvaluationError("game final terminal flag differs")
    if final_state.get("rules_identity_sha256") != (EXPECTED_RULES_IDENTITY_SHA256):
        raise HeldoutEvaluationError("game final Sanmill rules identity differs")
    if final_state.get("removal_pending") is not False:
        raise HeldoutEvaluationError("terminal game retains a removal obligation")
    if (
        final_state.get("pending_removal_count") != 0
        or final_state.get("legal_actions") != []
    ):
        raise HeldoutEvaluationError("terminal game advertises an action")
    if final_state.get("logical_ply_count") != (
        12 + record["post_prefix_logical_plies"]
    ):
        raise HeldoutEvaluationError("game final logical-ply count differs")
    white_turns = (record["post_prefix_logical_plies"] + 1) // 2
    black_turns = record["post_prefix_logical_plies"] // 2
    if final_state.get("logical_plies_by_side") != [
        6 + white_turns,
        6 + black_turns,
    ]:
        raise HeldoutEvaluationError("game final per-side counts differ")
    expected_action_tokens = prefix["action_token_count"] + sum(
        len(turn["actions"]) for turn in record["turns"]
    )
    if final_state.get("action_token_count") != expected_action_tokens:
        raise HeldoutEvaluationError("game final action-token count differs")
    if final_state.get("fen") != record["turns"][-1]["sanmill_fen_after"]:
        raise HeldoutEvaluationError("game final Sanmill FEN differs")
    if set(record["search_summary"]) != _SEARCH_SUMMARY_FIELDS or record[
        "search_summary"
    ] != _search_summary(record["turns"]):
        raise HeldoutEvaluationError("game search summary differs")
    for field in ("game_elapsed_seconds", "cumulative_active_seconds"):
        value = record[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise HeldoutEvaluationError("game timing evidence is invalid")
    if record["cumulative_active_seconds"] < record["game_elapsed_seconds"]:
        raise HeldoutEvaluationError("game cumulative time is invalid")
    _validate_turns(record)


def load_game_ledger(
    spec: Mapping[str, Any],
    path: str | Path,
) -> tuple[list[dict[str, Any]], str | None]:
    ledger = Path(path)
    if not ledger.exists():
        return [], None
    records: list[dict[str, Any]] = []
    previous_hash: str | None = None
    with ledger.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith(b"\n") or b"\r" in line:
                raise HeldoutEvaluationError(
                    f"ledger line {line_number} is not LF-framed"
                )
            try:
                wrapper = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HeldoutEvaluationError(
                    f"ledger line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(wrapper, dict) or set(wrapper) != {
                "record",
                "record_sha256",
            }:
                raise HeldoutEvaluationError(
                    f"ledger line {line_number} wrapper differs"
                )
            if line != canonical_json_bytes(wrapper) + b"\n":
                raise HeldoutEvaluationError(
                    f"ledger line {line_number} is not canonical JSON"
                )
            record = wrapper["record"]
            record_hash = wrapper["record_sha256"]
            if not isinstance(record, dict) or not isinstance(record_hash, str):
                raise HeldoutEvaluationError(
                    f"ledger line {line_number} has invalid types"
                )
            if canonical_sha256(record) != record_hash:
                raise HeldoutEvaluationError(f"ledger line {line_number} hash differs")
            _validate_game_record(spec, record, len(records), previous_hash)
            if (
                records
                and record["cumulative_active_seconds"]
                < records[-1]["cumulative_active_seconds"]
            ):
                raise HeldoutEvaluationError(
                    f"ledger line {line_number} cumulative time regressed"
                )
            records.append(record)
            previous_hash = record_hash
    if len(records) > 128:
        raise HeldoutEvaluationError("ledger contains too many games")
    return records, previous_hash


def append_game_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    must_create: bool,
) -> str:
    """Append one complete fsynced record without rewriting prior evidence."""
    record_hash = canonical_sha256(record)
    encoded = (
        canonical_json_bytes({"record": dict(record), "record_sha256": record_hash})
        + b"\n"
    )
    mode = "xb" if must_create else "ab"
    with Path(path).open(mode) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return record_hash


def _interval(values: Sequence[float], z: float) -> dict[str, Any]:
    support = len(values)
    if not values:
        return {
            "support_pairs": 0,
            "mean_pair_score_difference": None,
            "sample_standard_deviation": None,
            "standard_error": None,
            "interval": [None, None],
            "decision": "inconclusive",
        }
    mean = sum(values) / support
    if support == 1:
        return {
            "support_pairs": 1,
            "mean_pair_score_difference": mean,
            "sample_standard_deviation": None,
            "standard_error": None,
            "interval": [None, None],
            "decision": "inconclusive",
        }
    variance = sum((value - mean) ** 2 for value in values) / (support - 1)
    deviation = math.sqrt(variance)
    standard_error = deviation / math.sqrt(support)
    lower = mean - z * standard_error
    upper = mean + z * standard_error
    if lower > 0:
        decision = "candidate_ahead"
    elif upper < 0:
        decision = "candidate_behind"
    else:
        decision = "inconclusive"
    return {
        "support_pairs": support,
        "mean_pair_score_difference": mean,
        "sample_standard_deviation": deviation,
        "standard_error": standard_error,
        "interval": [lower, upper],
        "decision": decision,
    }


def _wdl(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [float(record["candidate_score"]) for record in records]
    wins = scores.count(1.0)
    draws = scores.count(0.5)
    losses = scores.count(0.0)
    return {
        "games": len(scores),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": (sum(scores) / len(scores) if scores else None),
    }


def _pair_summary(
    records: Sequence[Mapping[str, Any]],
    pair_indices: Iterable[int],
    *,
    z: float,
) -> dict[str, Any]:
    by_pair = {int(record["pair_index"]): [] for record in records}
    for record in records:
        by_pair[int(record["pair_index"])].append(record)
    selected: list[Mapping[str, Any]] = []
    differences = []
    for pair_index in pair_indices:
        pair = sorted(
            by_pair.get(pair_index, []), key=lambda item: item["game_in_pair"]
        )
        if len(pair) != 2:
            raise HeldoutEvaluationError("result pair is incomplete")
        selected.extend(pair)
        differences.append(sum(float(item["candidate_score"]) for item in pair) - 1.0)
    return {**_wdl(selected), **_interval(differences, z)}


def recompute_heldout_evaluation(
    spec_path: str | Path,
    ledger_path: str | Path,
) -> dict[str, Any]:
    """Recompute every result from the complete immutable game ledger."""
    spec = load_runtime_spec(spec_path)
    records, tail_hash = load_game_ledger(spec, ledger_path)
    if len(records) != 128:
        raise HeldoutEvaluationError(
            f"held-out ledger is incomplete: {len(records)}/128"
        )
    z = float(spec["analysis"]["confidence"]["z"])
    complete = _pair_summary(records, range(64), z=z)
    if complete["decision"] not in {
        "candidate_ahead",
        "candidate_behind",
        "inconclusive",
    }:
        raise HeldoutEvaluationError("held-out decision is invalid")

    strata: dict[str, Any] = {}
    for stratum in ("book", "human_db", "perfect_db"):
        indices = sorted(
            {
                int(record["pair_index"])
                for record in records
                if record["stratum"] == stratum
            }
        )
        strata[stratum] = _pair_summary(records, indices, z=z)
    strict_indices = sorted(
        {
            int(record["pair_index"])
            for record in records
            if record["strict_independence_sensitivity"]
        }
    )
    by_color = {
        color: _wdl(
            [record for record in records if record["candidate_color"] == color]
        )
        for color in ("W", "B")
    }
    termination = dict(
        sorted(Counter(str(record["outcome_reason"]) for record in records).items())
    )
    lengths = [int(record["post_prefix_logical_plies"]) for record in records]
    search_turns = [
        turn["search"]
        for record in records
        for turn in record["turns"]
        if turn["actor"] == "sanmill"
    ]
    nodes = [int(turn["total_nodes"]) for turn in search_turns]
    depths = [
        int(turn["completed_depth"])
        for turn in search_turns
        if turn["completed_depth"] is not None
    ]
    body = {
        "schema_version": HELDOUT_REPORT_SCHEMA,
        "evaluation_id": spec["evaluation_id"],
        "spec_identity": spec["spec_identity"],
        "ledger_sha256": _sha256_file(Path(ledger_path)),
        "ledger_tail_record_sha256": tail_hash,
        "status": "completed",
        "primary": complete,
        "by_source_stratum": strata,
        "by_candidate_color": by_color,
        "strict_independence_sensitivity": _pair_summary(records, strict_indices, z=z),
        "termination_reasons": termination,
        "game_lengths": {
            "support_games": len(lengths),
            "min_post_prefix_logical_plies": min(lengths),
            "max_post_prefix_logical_plies": max(lengths),
            "mean_post_prefix_logical_plies": sum(lengths) / len(lengths),
        },
        "baseline_search": {
            "support_turns": len(search_turns),
            "fixed_node_ceiling": 500_000,
            "total_observed_nodes": sum(nodes),
            "min_observed_nodes": min(nodes) if nodes else None,
            "max_observed_nodes": max(nodes) if nodes else None,
            "mean_observed_nodes": sum(nodes) / len(nodes) if nodes else None,
            "min_completed_depth": min(depths) if depths else None,
            "max_completed_depth": max(depths) if depths else None,
            "mean_completed_depth": (sum(depths) / len(depths) if depths else None),
        },
        "claim_boundary": {
            "fixed_corpus_relation_only": True,
            "full_corpus_data_disjoint": False,
            "multiple_training_seeds": False,
            "automatic_promotion": False,
            "automatic_publication": False,
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}


def write_new_canonical(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_json_bytes(dict(value)))
        handle.flush()
        os.fsync(handle.fileno())


def replace_canonical(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise HeldoutEvaluationError("atomic temporary path already exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(ATOMIC_REPLACE_PERMISSION_RETRIES):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt + 1 == ATOMIC_REPLACE_PERMISSION_RETRIES:
                    raise
                time.sleep(ATOMIC_REPLACE_RETRY_SECONDS)
    finally:
        if temporary.exists():
            temporary.unlink()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=not binary,
            encoding=None if binary else "utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HeldoutEvaluationError("cannot inspect repository state") from exc
    if binary:
        return result.stdout
    return result.stdout.strip()


def _repository_record() -> dict[str, Any]:
    branch = str(_git("branch", "--show-current"))
    head = str(_git("rev-parse", "HEAD"))
    tree = str(_git("rev-parse", "HEAD^{tree}"))
    upstream = str(_git("rev-parse", "@{upstream}"))
    status = str(_git("status", "--porcelain=v1", "--untracked-files=all"))
    if branch != "dev":
        raise HeldoutEvaluationError("held-out evaluation must run from dev")
    if status:
        raise HeldoutEvaluationError("held-out evaluation requires a clean tree")
    if head != upstream:
        raise HeldoutEvaluationError("held-out evaluation requires dev == origin/dev")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_PLAN_COMMIT, head],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HeldoutEvaluationError(
            "held-out plan commit is not an ancestor of runtime HEAD"
        ) from exc
    plan_blob = _git(
        "show",
        f"{EXPECTED_PLAN_COMMIT}:{PLAN_RELATIVE.as_posix()}",
        binary=True,
    )
    if not isinstance(plan_blob, bytes) or hashlib.sha256(plan_blob).hexdigest() != (
        EXPECTED_PLAN_FILE_SHA256
    ):
        raise HeldoutEvaluationError("tracked plan blob differs from the pin")
    subprocess.run(
        ["git", "diff", "--check"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return {
        "branch": branch,
        "head": head,
        "tree": tree,
        "upstream_commit": upstream,
        "tracked_worktree": "clean",
        "published": True,
        "plan_commit_is_ancestor": True,
    }


def _output_targets(paths: HeldoutPaths) -> dict[str, Path]:
    return {
        "plan": paths.output_plan,
        "authorization": paths.output_authorization,
        "readiness": paths.output_root / "readiness.json",
        "specification": paths.output_spec,
        "launch": paths.output_root / "launch.json",
        "ledger": paths.output_ledger,
        "progress": paths.output_root / "progress.json",
        "report": paths.output_report,
        "failure": paths.output_root / "failure.json",
        "lock": paths.output_root / "evaluator.lock",
    }


def _assert_ignored(path: Path) -> None:
    try:
        relative = path.resolve(strict=False).relative_to(_REPO_ROOT).as_posix()
    except ValueError as exc:
        raise HeldoutEvaluationError("evaluation output leaves the repository") from exc
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise HeldoutEvaluationError("evaluation output is not ignored by Git")


def _output_record(paths: HeldoutPaths, *, resume: bool) -> dict[str, Any]:
    targets = _output_targets(paths)
    for target in targets.values():
        _assert_ignored(target)
    if not paths.candidate_bundle.is_dir():
        raise HeldoutEvaluationError("candidate route bundle is missing")
    if resume:
        required = {
            "plan",
            "authorization",
            "readiness",
            "specification",
            "progress",
        }
        missing = sorted(name for name in required if not targets[name].is_file())
        if missing:
            raise HeldoutEvaluationError(
                "resume output is incomplete: " + ", ".join(missing)
            )
        if targets["report"].exists() or targets["failure"].exists():
            raise HeldoutEvaluationError("completed or failed output cannot resume")
    else:
        existing = sorted(name for name, path in targets.items() if path.exists())
        if existing:
            raise HeldoutEvaluationError(
                "new evaluation output already contains: " + ", ".join(existing)
            )
    return {
        "root": "learned_ai/checkpoints/evaluation/"
        "sanmill-corrected-retained-v2-heldout-v1",
        "candidate_bundle_present": True,
        "mode": "resume" if resume else "initial",
        "targets": "valid-existing-run" if resume else "absent",
        "git_ignored": True,
    }


def _checkpoint_and_policy_record(
    contract: FrozenHeldoutContract,
    paths: HeldoutPaths,
) -> dict[str, Any]:
    _require_file_hash(
        paths.checkpoint,
        EXPECTED_CHECKPOINT_FILE_SHA256,
        name="candidate checkpoint",
    )
    envelope = load_checkpoint(paths.checkpoint, map_location="cpu")
    if envelope.payload_sha256 != EXPECTED_CHECKPOINT_PAYLOAD_SHA256:
        raise HeldoutEvaluationError("checkpoint payload identity differs")
    if envelope.descriptor.checkpoint_id != EXPECTED_CHECKPOINT_ID:
        raise HeldoutEvaluationError("checkpoint ID differs")
    checkpoint_assets = dict(envelope.descriptor.asset_identities)
    expected_checkpoint_assets = {
        "human_db": EXPECTED_HUMAN_DB_IDENTITY,
        "malom_tablebase": EXPECTED_MALOM_IDENTITY,
        "specialist_db": EXPECTED_SPECIALIST_DB_IDENTITY,
        "sanmill_training_runtime": contract.plan["baseline"]["identity"],
        "training_ruleset": contract.plan["rules"]["ruleset_semantic_digest"],
        "mif_suite_1_0": "sha256:"
        + mif_release_identity()["releaseManifestSha256"].removeprefix("sha256:"),
    }
    if any(
        checkpoint_assets.get(name) != identity
        for name, identity in expected_checkpoint_assets.items()
    ):
        raise HeldoutEvaluationError("checkpoint resource identities differ")

    policy = load_training_aligned_policy(
        paths.candidate_bundle,
        human_db_path=paths.human_db,
        specialist_db_path=paths.specialist_db,
        malom_path=paths.malom_db,
        malom_manifest_path=paths.malom_manifest,
        device="cpu",
    )
    try:
        if policy.bundle_identity != EXPECTED_BUNDLE_IDENTITY:
            raise HeldoutEvaluationError("loaded candidate bundle differs")
        manifest = policy.manifest
        if manifest["producer"]["checkpoint_id"] != EXPECTED_CHECKPOINT_ID:
            raise HeldoutEvaluationError("bundle producer checkpoint differs")
        if manifest["producer"]["checkpoint_payload_sha256"] != (
            EXPECTED_CHECKPOINT_PAYLOAD_SHA256
        ):
            raise HeldoutEvaluationError("bundle producer payload differs")
        observed_resources = {
            "human_db": policy.resource_reports["human_db"]["identity"],
            "specialist_db": policy.resource_reports["specialist_db"]["content_sha256"],
            "malom_tablebase": policy.resource_reports["malom_tablebase"]["identity"],
        }
        if observed_resources != {
            "human_db": EXPECTED_HUMAN_DB_IDENTITY,
            "specialist_db": EXPECTED_SPECIALIST_DB_IDENTITY,
            "malom_tablebase": EXPECTED_MALOM_IDENTITY,
        }:
            raise HeldoutEvaluationError("loaded route resource identities differ")
        synthetic_board = BoardState.new_game()
        first_move = policy.choose_move(synthetic_board)
        second_move = policy.choose_move(synthetic_board)
        if not first_move or first_move != second_move:
            raise HeldoutEvaluationError(
                "candidate synthetic CPU canary is not deterministic"
            )
        if first_move not in get_all_legal_moves(synthetic_board):
            raise HeldoutEvaluationError(
                "candidate synthetic CPU canary returned an illegal move"
            )
        return {
            "checkpoint_id": envelope.descriptor.checkpoint_id,
            "checkpoint_file_sha256": EXPECTED_CHECKPOINT_FILE_SHA256,
            "checkpoint_payload_sha256": envelope.payload_sha256,
            "bundle_identity": policy.bundle_identity,
            "route": manifest["route"]["name"],
            "device": "cpu",
            "precision": "float32",
            "synthetic_candidate_moves_requested": 2,
            "corpus_candidate_moves_requested": 0,
            "synthetic_canary_move_identity": canonical_sha256(first_move),
            "resource_identities": observed_resources,
        }
    finally:
        policy.close()


def _rules_record(
    contract: FrozenHeldoutContract, paths: HeldoutPaths
) -> dict[str, Any]:
    release = mif_release_identity()
    expected_release = {
        "tag": contract.plan["rules"]["mif_tag"],
        "releaseCommit": contract.plan["rules"]["mif_release_commit"],
        "suiteJcsSha256": contract.plan["rules"]["mif_suite_jcs_sha256"],
    }
    if any(release[key] != value for key, value in expected_release.items()):
        raise HeldoutEvaluationError("MIF release identity differs")
    ruleset = load_trainer_ruleset(paths.ruleset_manifest)
    if ruleset.semantic_digest != TRAINER_RULESET_SEMANTIC_DIGEST:
        raise HeldoutEvaluationError("trainer ruleset identity differs")
    if ruleset.semantic_digest != contract.plan["rules"]["ruleset_semantic_digest"]:
        raise HeldoutEvaluationError("plan ruleset identity differs")
    return {
        "mif": release,
        "ruleset": ruleset.to_dict(),
    }


def _sanmill_record(
    contract: FrozenHeldoutContract,
    paths: HeldoutPaths,
) -> tuple[dict[str, Any], Any]:
    installation = inspect_sanmill_training_installation(paths.sanmill_checkout)
    installed = training_installation_record(installation, seed=42)
    if installed["identity"] != contract.plan["baseline"]["identity"]:
        raise HeldoutEvaluationError("Sanmill runtime identity differs")
    if installed["commit"] != TRAINING_SANMILL_COMMIT:
        raise HeldoutEvaluationError("Sanmill commit differs")
    if installed["tree"] != TRAINING_SANMILL_TREE:
        raise HeldoutEvaluationError("Sanmill tree differs")
    if installed["binary_sha256"] != TRAINING_SANMILL_BINARY_SHA256:
        raise HeldoutEvaluationError("Sanmill binary differs")
    if installed["strict_referee"]["semanticDigest"] != (
        TRAINING_REFEREE_SEMANTIC_DIGEST
    ):
        raise HeldoutEvaluationError("Sanmill referee identity differs")
    probe = probe_sanmill_training_runtime(
        paths.sanmill_checkout,
        node_budget=500_000,
        depth=None,
        seed=42,
    )
    first = probe["probe"]["first_turn"]
    return (
        {
            "identity": installed["identity"],
            "commit": installed["commit"],
            "tree": installed["tree"],
            "binary_sha256": installed["binary_sha256"],
            "strict_referee_semantic_digest": installed["strict_referee"][
                "semanticDigest"
            ],
            "strict_options": installed["strict_options"],
            "deterministic_fresh_processes": probe["probe"]["fresh_processes"],
            "node_ceiling": probe["probe"]["node_budget"],
            "probe_observation_sha256": probe["probe"]["observation_sha256"],
            "probe_total_nodes": first["total_nodes"],
            "probe_completed_depth": first["completed_depth"],
        },
        installation,
    )


def _synthetic_route_interop_record(
    paths: HeldoutPaths,
    installation: Any,
) -> dict[str, Any]:
    """Exercise both actors on a fresh non-corpus board without scoring a game."""
    policy = _load_policy(paths)
    try:
        with SanmillTrainingGame(installation, seed=42) as game:
            board = BoardState.new_game()
            candidate_move = policy.choose_move(board)
            if not candidate_move:
                raise HeldoutEvaluationError(
                    "candidate returned no synthetic interoperability move"
                )
            candidate_turn = game.apply_nmm_move(board, candidate_move)
            board = board.apply_move(candidate_turn.move)
            baseline_turn = game.search_and_apply(
                board,
                node_budget=500_000,
                depth=None,
            )
            board = board.apply_move(baseline_turn.move)
            game.assert_current_board(board)
            if game.state.logical_ply_count != 2:
                raise HeldoutEvaluationError(
                    "synthetic interoperability logical count differs"
                )
            if baseline_turn.search is None:
                raise HeldoutEvaluationError(
                    "synthetic interoperability search evidence is absent"
                )
            return {
                "source": "fresh-empty-board-not-heldout-prefix",
                "candidate_bundle_identity": policy.bundle_identity,
                "candidate_actions": list(candidate_turn.actions),
                "sanmill_actions": list(baseline_turn.actions),
                "sanmill_node_ceiling": baseline_turn.search.node_budget,
                "sanmill_observed_nodes": baseline_turn.search.total_nodes,
                "sanmill_completed_depth": baseline_turn.search.completed_depth,
                "resulting_history_sha256": game.state.history_sha256,
                "logical_plies": game.state.logical_ply_count,
                "corpus_prefix_loaded": False,
                "game_scored": False,
            }
    finally:
        policy.close()


def _resume_continuity_record(
    contract: FrozenHeldoutContract,
    paths: HeldoutPaths,
) -> dict[str, Any]:
    if paths.output_plan.read_bytes() != contract.plan_path.read_bytes():
        raise HeldoutEvaluationError("runtime plan copy differs")
    if paths.output_authorization.read_bytes() != (
        contract.authorization_path.read_bytes()
    ):
        raise HeldoutEvaluationError("runtime authorization copy differs")
    spec = load_runtime_spec(paths.output_spec)
    if spec["plan"]["identity"] != contract.plan_identity:
        raise HeldoutEvaluationError("resume plan identity differs")
    if spec["authorization"]["identity"] != contract.authorization_identity:
        raise HeldoutEvaluationError("resume authorization identity differs")
    if spec["implementation"]["commit"] != str(_git("rev-parse", "HEAD")):
        raise HeldoutEvaluationError("resume implementation commit differs")
    current_runtime = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "device": "cpu",
        "precision": "float32",
        "seed": 42,
    }
    if spec.get("runtime") != current_runtime:
        raise HeldoutEvaluationError("resume host runtime differs")
    persisted_readiness = _strict_json(paths.output_root / "readiness.json")
    readiness_identity = _identity(persisted_readiness, "readiness_identity")
    if readiness_identity != spec["readiness_identity"]:
        raise HeldoutEvaluationError("persisted readiness identity differs")
    records, tail = load_game_ledger(spec, paths.output_ledger)
    progress = _load_progress(
        paths.output_root / "progress.json", spec["spec_identity"]
    )
    completed = int(progress["completed_games"])
    if completed < 0 or completed > len(records):
        raise HeldoutEvaluationError("resume progress is ahead of the ledger")
    expected_progress_tail = (
        canonical_sha256(records[completed - 1]) if completed else None
    )
    if progress["ledger_tail_sha256"] != expected_progress_tail:
        raise HeldoutEvaluationError("resume progress ledger tail differs")
    launch_path = paths.output_root / "launch.json"
    if records and not launch_path.is_file():
        raise HeldoutEvaluationError("resume ledger exists without launch evidence")
    if launch_path.exists():
        launch = _strict_json(launch_path)
        _identity(launch, "launch_identity")
        if launch.get("spec_identity") != spec["spec_identity"]:
            raise HeldoutEvaluationError("resume launch spec differs")
    return {
        "spec_identity": spec["spec_identity"],
        "completed_games": len(records),
        "progress_completed_games": completed,
        "ledger_tail_record_sha256": tail,
        "authorization_consumed": launch_path.is_file(),
        "missing_suffix_games": 128 - len(records),
    }


def _competing_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        raise HeldoutEvaluationError(
            "competing-process audit is implemented only for the Windows host"
        )
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        raise HeldoutEvaluationError("PowerShell is unavailable for process audit")
    pattern = (
        "train_s_gen_v2\\.py|manage_generalist_run\\.py|run_heldout_evaluation\\.py"
    )
    script = (
        f"$self={os.getpid()}; $parent={os.getppid()}; $scanner=$PID; "
        "@(Get-CimInstance Win32_Process | "
        "Where-Object { $_.ProcessId -ne $self -and "
        "$_.ProcessId -ne $parent -and $_.ProcessId -ne $scanner -and "
        f"$_.CommandLine -match "
        f"'{pattern}' }} | Select-Object ProcessId,Name,CommandLine) | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [shell, "-NoProfile", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        payload = json.loads(result.stdout or "[]")
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise HeldoutEvaluationError("cannot audit competing processes") from exc
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise HeldoutEvaluationError("process audit returned an invalid shape")
    observed = []
    for item in payload:
        if not isinstance(item, dict):
            raise HeldoutEvaluationError("process audit member is invalid")
        command = str(item.get("CommandLine") or "")
        kind = (
            "heldout-evaluator"
            if "run_heldout_evaluation.py" in command
            else "managed-generalist"
            if "manage_generalist_run.py" in command
            else "generalist-trainer"
        )
        observed.append(
            {
                "pid": int(item["ProcessId"]),
                "name": str(item.get("Name") or "unknown"),
                "kind": kind,
            }
        )
    return observed


def _run_check(command: Sequence[str], *, label: str) -> dict[str, Any]:
    result = subprocess.run(
        list(command),
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise HeldoutEvaluationError(f"{label} failed with exit {result.returncode}")
    meaningful = [line.strip() for line in output.splitlines() if line.strip()]
    return {
        "label": label,
        "exit_code": result.returncode,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "last_output_line": meaningful[-1] if meaningful else "",
    }


def _test_record() -> dict[str, Any]:
    focused = _run_check(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_heldout_exposure.py",
            "tests/test_heldout_evaluation_plan.py",
            "tests/test_heldout_evaluation_runner.py",
            "tests/test_training_aligned_policy.py",
            "tests/test_sanmill_training_referee.py",
            "tests/test_sanmill_node_calibration.py",
            "tests/test_training_route_bundle.py",
            "tests/test_checkpoint_envelope.py",
            "tests/test_run_contract.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            ".tmp/pytest-heldout-final-preflight",
        ],
        label="held-out focused tests",
    )
    mandatory = _run_check(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_malom_db.py",
            "tests/test_sentinel_db_teacher.py",
            "tests/test_malom_label_provenance.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            ".tmp/pytest-heldout-final-provenance",
        ],
        label="mandatory Malom and provenance tests",
    )
    ruff = shutil.which("ruff")
    if ruff is None:
        raise HeldoutEvaluationError("Ruff is unavailable")
    lint = _run_check(
        [
            ruff,
            "check",
            "learned_ai/evaluation/heldout_evaluation.py",
            "learned_ai/evaluation/heldout_exposure.py",
            "scripts/run_heldout_evaluation.py",
            "scripts/audit_heldout_evaluation_corpus.py",
            "tests/test_heldout_evaluation_runner.py",
            "tests/test_heldout_evaluation_plan.py",
            "tests/test_heldout_exposure.py",
        ],
        label="held-out Ruff checks",
    )
    return {"focused": focused, "mandatory": mandatory, "ruff": lint}


def _gate(
    gates: list[dict[str, Any]],
    name: str,
    expected: str,
    operation: Callable[[], Any],
) -> Any | None:
    try:
        observed = operation()
    except Exception as exc:
        gates.append(
            {
                "gate": name,
                "expected": expected,
                "observed": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                "result": "fail",
            }
        )
        return None
    gates.append(
        {
            "gate": name,
            "expected": expected,
            "observed": observed,
            "result": "pass",
        }
    )
    return observed


def build_readiness_report(
    contract: FrozenHeldoutContract,
    paths: HeldoutPaths,
    *,
    resume: bool = False,
    run_tests: bool = True,
    audit_prefixes: bool = True,
) -> dict[str, Any]:
    """Run every read-only launch gate without consuming a corpus game."""
    gates: list[dict[str, Any]] = []
    _gate(
        gates,
        "repository",
        "clean published dev with the frozen plan commit as an ancestor",
        _repository_record,
    )
    _gate(
        gates,
        "contract",
        "exact plan, authorization, corpus, and exposure identities",
        lambda: {
            "plan_identity": contract.plan_identity,
            "authorization_identity": contract.authorization_identity,
            "corpus_identity": EXPECTED_CORPUS_IDENTITY,
            "exposure_audit_identity": EXPECTED_AUDIT_IDENTITY,
            "games_authorized": 128,
        },
    )
    _gate(
        gates,
        "outputs",
        "ignored absent targets or one valid same-spec partial run",
        lambda: _output_record(paths, resume=resume),
    )
    if resume:
        _gate(
            gates,
            "resume_continuity",
            "same plan, grant, implementation, spec, ledger and progress chain",
            lambda: _resume_continuity_record(contract, paths),
        )
    _gate(
        gates,
        "rules",
        "published MIF Suite and exact trainer rules semantic digest",
        lambda: _rules_record(contract, paths),
    )
    candidate_result = _gate(
        gates,
        "candidate",
        "verified checkpoint, route bundle, CPU canaries, and read-only data",
        lambda: _checkpoint_and_policy_record(contract, paths),
    )
    sanmill_result: tuple[dict[str, Any], Any] | None = _gate(
        gates,
        "sanmill",
        "pinned strict runtime and deterministic 500,000-node canary",
        lambda: _sanmill_record(contract, paths),
    )
    if sanmill_result is not None:
        # The installation object is an internal convenience, never evidence.
        gates[-1]["observed"] = sanmill_result[0]
    if candidate_result is None or sanmill_result is None:
        gates.append(
            {
                "gate": "synthetic_route_interop",
                "expected": "candidate and Sanmill complete two non-corpus turns",
                "observed": {"error": "candidate or Sanmill gate did not pass"},
                "result": "fail",
            }
        )
    else:
        _gate(
            gates,
            "synthetic_route_interop",
            "candidate and Sanmill complete two non-corpus turns",
            lambda: _synthetic_route_interop_record(paths, sanmill_result[1]),
        )
    if audit_prefixes:
        if sanmill_result is None:
            gates.append(
                {
                    "gate": "prefix_replay",
                    "expected": "all 64 histories replay in fresh strict processes",
                    "observed": {"error": "Sanmill gate did not pass"},
                    "result": "fail",
                }
            )
        else:
            _gate(
                gates,
                "prefix_replay",
                "all 64 histories replay in fresh strict processes",
                lambda: audit_frozen_prefixes(contract, sanmill_result[1], seed=42),
            )
    _gate(
        gates,
        "process_ownership",
        "no competing trainer or held-out evaluator",
        lambda: (
            {"competing_processes": []}
            if not (processes := _competing_processes())
            else (_raise_competing(processes))
        ),
    )
    if run_tests:
        _gate(
            gates,
            "tests",
            "focused, mandatory provenance, and Ruff checks pass",
            _test_record,
        )
    passed = all(gate["result"] == "pass" for gate in gates)
    body = {
        "schema_version": HELDOUT_READINESS_SCHEMA,
        "evaluation_id": EXPECTED_EVALUATION_ID,
        "plan_identity": contract.plan_identity,
        "authorization_identity": contract.authorization_identity,
        "mode": "resume" if resume else "initial",
        "corpus_candidate_move_requested": False,
        "corpus_games_played": 0,
        "gates": gates,
        "ready": passed,
        "status": "ready_for_heldout_evaluation" if passed else "not_ready",
    }
    return {**body, "readiness_identity": canonical_sha256(body)}


def _raise_competing(processes: list[dict[str, Any]]) -> None:
    summary = ", ".join(f"{item['kind']}:{item['pid']}" for item in processes)
    raise HeldoutEvaluationError("competing processes are running: " + summary)


def require_ready(report: Mapping[str, Any]) -> None:
    if report.get("ready") is not True or report.get("status") != (
        "ready_for_heldout_evaluation"
    ):
        failed = [
            str(gate.get("gate"))
            for gate in report.get("gates", [])
            if gate.get("result") != "pass"
        ]
        raise HeldoutEvaluationError(
            "held-out preflight is not ready: " + ", ".join(failed)
        )


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class EvaluatorLock:
    """Exclusive machine-local evaluator ownership with stale-lock retention."""

    def __init__(self, root: Path, spec_identity: str, *, resume: bool) -> None:
        self.root = root
        self.path = root / "evaluator.lock"
        self.spec_identity = spec_identity
        self.resume = resume
        self.record: dict[str, Any] | None = None

    def __enter__(self) -> "EvaluatorLock":
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            existing = _strict_json(self.path)
            pid = existing.get("pid")
            if not isinstance(pid, int):
                raise HeldoutEvaluationError("evaluator lock is malformed")
            if _pid_is_running(pid):
                raise HeldoutEvaluationError("another evaluator owns the lock")
            if not self.resume or existing.get("spec_identity") != self.spec_identity:
                raise HeldoutEvaluationError("stale evaluator lock is not resumable")
            archive = self.root / "lock-history"
            archive.mkdir(exist_ok=True)
            identity = canonical_sha256(existing)
            destination = archive / f"stale-{identity}.json"
            if destination.exists():
                raise HeldoutEvaluationError("stale lock archive already exists")
            os.replace(self.path, destination)
        body = {
            "schema_version": "nmm.sanmill-heldout-lock.v1",
            "spec_identity": self.spec_identity,
            "pid": os.getpid(),
            "acquired_at_utc": utc_now(),
        }
        self.record = {**body, "lock_identity": canonical_sha256(body)}
        write_new_canonical(self.path, self.record)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.record is None or not self.path.exists():
            return
        observed = _strict_json(self.path)
        if observed != self.record:
            raise HeldoutEvaluationError("evaluator lock changed while owned")
        self.path.unlink()


class ActiveClock:
    """Persistable active-time ceiling, excluding time between processes."""

    def __init__(
        self,
        *,
        base_seconds: float,
        max_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(base_seconds) or base_seconds < 0:
            raise HeldoutEvaluationError("active-time base is invalid")
        self.base_seconds = base_seconds
        self.max_seconds = max_seconds
        self.monotonic = monotonic
        self.started = monotonic()

    def elapsed(self) -> float:
        value = self.base_seconds + (self.monotonic() - self.started)
        if not math.isfinite(value) or value < self.base_seconds:
            raise HeldoutEvaluationError("active evaluator time is invalid")
        return value

    def require_within_budget(self) -> float:
        value = self.elapsed()
        if value > self.max_seconds:
            raise HeldoutEvaluationInvalid("active evaluator time ceiling reached")
        return value


def _progress_body(
    spec_identity: str,
    *,
    completed_games: int,
    current_game_ordinal: int | None,
    current_stage: str | None,
    current_stage_ply: int,
    active_seconds: float,
    ledger_tail_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": HELDOUT_PROGRESS_SCHEMA,
        "spec_identity": spec_identity,
        "completed_games": completed_games,
        "current_game_ordinal": current_game_ordinal,
        "current_stage": current_stage,
        "current_stage_ply": current_stage_ply,
        "active_seconds": round(active_seconds, 6),
        "ledger_tail_sha256": ledger_tail_sha256,
    }


def _write_progress(path: Path, body: Mapping[str, Any]) -> None:
    value = {**dict(body), "progress_identity": canonical_sha256(body)}
    replace_canonical(path, value)


def _load_progress(path: Path, spec_identity: str) -> dict[str, Any]:
    value = _strict_json(path)
    if _identity(value, "progress_identity") != value["progress_identity"]:
        raise HeldoutEvaluationError("progress identity differs")
    if value.get("schema_version") != HELDOUT_PROGRESS_SCHEMA:
        raise HeldoutEvaluationError("progress schema differs")
    if value.get("spec_identity") != spec_identity:
        raise HeldoutEvaluationError("progress spec differs")
    active = value.get("active_seconds")
    if not isinstance(active, (int, float)) or not math.isfinite(active) or active < 0:
        raise HeldoutEvaluationError("progress active time is invalid")
    return value


def _portable_failure_message(message: str) -> str:
    result = message.replace(str(_REPO_ROOT), "<repo>")
    return result.replace(str(_REPO_ROOT).replace("\\", "/"), "<repo>")


def _write_failure(
    paths: HeldoutPaths,
    spec_identity: str,
    error: BaseException,
    *,
    completed_games: int,
    ledger_tail_sha256: str | None,
) -> None:
    target = paths.output_root / "failure.json"
    if target.exists():
        return
    body = {
        "schema_version": HELDOUT_FAILURE_SCHEMA,
        "spec_identity": spec_identity,
        "failed_at_utc": utc_now(),
        "error_type": type(error).__name__,
        "message": _portable_failure_message(str(error)),
        "completed_games": completed_games,
        "ledger_tail_sha256": ledger_tail_sha256,
        "candidate_loss_manufactured": False,
        "game_result_manufactured": False,
    }
    write_new_canonical(target, {**body, "failure_identity": canonical_sha256(body)})


def _initialise_output(
    contract: FrozenHeldoutContract,
    paths: HeldoutPaths,
    readiness: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    with paths.output_plan.open("xb") as handle:
        handle.write(contract.plan_path.read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    with paths.output_authorization.open("xb") as handle:
        handle.write(contract.authorization_path.read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    write_new_canonical(paths.output_root / "readiness.json", readiness)
    write_new_canonical(paths.output_spec, spec)
    _write_progress(
        paths.output_root / "progress.json",
        _progress_body(
            spec["spec_identity"],
            completed_games=0,
            current_game_ordinal=None,
            current_stage=None,
            current_stage_ply=0,
            active_seconds=0.0,
            ledger_tail_sha256=None,
        ),
    )


def _validate_resume_output(
    contract: FrozenHeldoutContract,
    paths: HeldoutPaths,
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    if paths.output_plan.read_bytes() != contract.plan_path.read_bytes():
        raise HeldoutEvaluationError("runtime plan copy differs")
    if paths.output_authorization.read_bytes() != (
        contract.authorization_path.read_bytes()
    ):
        raise HeldoutEvaluationError("runtime authorization copy differs")
    spec = load_runtime_spec(paths.output_spec)
    if spec["plan"]["identity"] != contract.plan_identity:
        raise HeldoutEvaluationError("resume plan identity differs")
    if spec["authorization"]["identity"] != contract.authorization_identity:
        raise HeldoutEvaluationError("resume authorization identity differs")
    repository = next(
        gate["observed"] for gate in readiness["gates"] if gate["gate"] == "repository"
    )
    if spec["implementation"]["commit"] != repository["head"]:
        raise HeldoutEvaluationError("resume implementation commit differs")
    persisted_readiness = _strict_json(paths.output_root / "readiness.json")
    if (
        _identity(persisted_readiness, "readiness_identity")
        != spec["readiness_identity"]
    ):
        raise HeldoutEvaluationError("persisted readiness identity differs")
    return spec


def _launch_record(spec: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": HELDOUT_LAUNCH_SCHEMA,
        "evaluation_id": spec["evaluation_id"],
        "spec_identity": spec["spec_identity"],
        "authorization_identity": spec["authorization"]["identity"],
        "authorization_consumed": True,
        "consumption_reason": "first corpus game opened for play",
        "first_game_id": spec["schedule"][0]["game_id"],
        "launched_at_utc": utc_now(),
    }
    return {**body, "launch_identity": canonical_sha256(body)}


def _search_summary(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    searches = [turn["search"] for turn in turns if turn["actor"] == "sanmill"]
    nodes = [int(search["total_nodes"]) for search in searches]
    depths = [
        int(search["completed_depth"])
        for search in searches
        if search["completed_depth"] is not None
    ]
    return {
        "turns": len(searches),
        "node_ceiling": 500_000,
        "total_nodes": sum(nodes),
        "min_nodes": min(nodes) if nodes else None,
        "max_nodes": max(nodes) if nodes else None,
        "min_completed_depth": min(depths) if depths else None,
        "max_completed_depth": max(depths) if depths else None,
    }


def _turn_record(
    *,
    post_prefix_ply: int,
    mover_color: str,
    actor: str,
    board_after: BoardState,
    before_history: str,
    applied: SanmillAppliedTurn,
) -> dict[str, Any]:
    search = applied.search.semantic_record() if applied.search is not None else None
    if search is not None:
        if search["node_budget"] != 500_000:
            raise HeldoutEvaluationError("Sanmill search node ceiling differs")
        if search["total_nodes"] > 500_000:
            raise HeldoutEvaluationError("Sanmill exceeded the node ceiling")
        if search["search_calls"] <= 0:
            raise HeldoutEvaluationError("Sanmill did not attempt search")
    return {
        "post_prefix_logical_ply": post_prefix_ply,
        "mover_color": mover_color,
        "actor": actor,
        "move": dict(applied.move),
        "actions": list(applied.actions),
        "before_history_sha256": before_history,
        "after_history_sha256": applied.state.history_sha256,
        "logical_ply_count": applied.state.logical_ply_count,
        "local_fen_after": board_after.to_fen_string(),
        "sanmill_fen_after": applied.state.fen,
        "terminal": applied.state.terminal,
        "outcome_reason": applied.state.outcome_reason,
        "search": search,
    }


def play_heldout_game(
    *,
    spec: Mapping[str, Any],
    schedule_item: Mapping[str, Any],
    corpus_record: Mapping[str, Any],
    policy: TrainingAlignedPolicy,
    installation: Any,
    previous_record_sha256: str | None,
    clock: ActiveClock,
    progress_callback: Callable[[str, int], None],
    game_factory: Callable[..., Any] = SanmillTrainingGame,
) -> dict[str, Any]:
    """Play one rules-terminal game; any other ending raises invalid."""
    game_started = clock.elapsed()
    turns: list[dict[str, Any]] = []
    with game_factory(installation, seed=42) as game:

        def prefix_progress(ply: int) -> None:
            clock.require_within_budget()
            progress_callback("prefix", ply)

        board, prefix = replay_frozen_prefix(
            game, corpus_record, progress=prefix_progress
        )
        if prefix["prefix_identity"] != schedule_item["prefix_identity"]:
            raise HeldoutEvaluationError("runtime prefix identity differs")
        if (
            prefix["observed_history_sha256"]
            != schedule_item["expected_prefix_history_sha256"]
        ):
            raise HeldoutEvaluationError("runtime prefix history differs")

        candidate_color = schedule_item["candidate_color"]
        max_plies = int(spec["protocol"]["max_post_prefix_logical_plies"])
        for post_prefix_ply in range(1, max_plies + 1):
            clock.require_within_budget()
            before_history = game.state.history_sha256
            mover = board.turn
            if mover == candidate_color:
                actor = "candidate"
                move = policy.choose_move(board)
                if not move:
                    raise HeldoutEvaluationInvalid(
                        "candidate returned no move in an ongoing state"
                    )
                applied = game.apply_nmm_move(board, move)
            else:
                actor = "sanmill"
                applied = game.search_and_apply(board, node_budget=500_000, depth=None)
            board = board.apply_move(applied.move)
            turn = _turn_record(
                post_prefix_ply=post_prefix_ply,
                mover_color=mover,
                actor=actor,
                board_after=board,
                before_history=before_history,
                applied=applied,
            )
            turns.append(turn)
            clock.require_within_budget()
            progress_callback("game", post_prefix_ply)
            if applied.state.terminal:
                break
        else:
            raise HeldoutEvaluationInvalid("post-prefix logical-ply safety cap reached")

        final_state = game.state.portable_record()
        winner_name = game.state.winner
        winner = {None: None, "white": "W", "black": "B"}.get(winner_name)
        if winner_name not in {None, "white", "black"}:
            raise HeldoutEvaluationError("Sanmill winner value is unknown")
        if not game.state.terminal or game.state.outcome_reason == "ongoing":
            raise HeldoutEvaluationInvalid("game did not reach a rules terminal")
        score = 0.5 if winner is None else (1.0 if winner == candidate_color else 0.0)
        active = clock.require_within_budget()
        return {
            "schema_version": HELDOUT_GAME_SCHEMA,
            "spec_identity": spec["spec_identity"],
            "ordinal": schedule_item["ordinal"],
            "pair_index": schedule_item["pair_index"],
            "game_in_pair": schedule_item["game_in_pair"],
            "game_id": schedule_item["game_id"],
            "source_core_id": schedule_item["source_core_id"],
            "stratum": schedule_item["stratum"],
            "strict_independence_sensitivity": schedule_item[
                "strict_independence_sensitivity"
            ],
            "candidate_color": candidate_color,
            "candidate_score": score,
            "winner": winner,
            "outcome_reason": game.state.outcome_reason,
            "prefix": prefix,
            "post_prefix_logical_plies": len(turns),
            "final_state": final_state,
            "turns": turns,
            "search_summary": _search_summary(turns),
            "game_elapsed_seconds": round(active - game_started, 6),
            "cumulative_active_seconds": round(active, 6),
            "complete": True,
            "previous_record_sha256": previous_record_sha256,
        }


def _load_policy(paths: HeldoutPaths) -> TrainingAlignedPolicy:
    return load_training_aligned_policy(
        paths.candidate_bundle,
        human_db_path=paths.human_db,
        specialist_db_path=paths.specialist_db,
        malom_path=paths.malom_db,
        malom_manifest_path=paths.malom_manifest,
        device="cpu",
    )


def run_frozen_heldout_evaluation(
    contract: FrozenHeldoutContract,
    paths: HeldoutPaths,
    readiness: Mapping[str, Any],
    *,
    resume: bool = False,
    policy_loader: Callable[[HeldoutPaths], TrainingAlignedPolicy] = _load_policy,
    game_factory: Callable[..., Any] = SanmillTrainingGame,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Consume the one-run grant and execute exactly the frozen schedule."""
    require_ready(readiness)
    if resume:
        spec = _validate_resume_output(contract, paths, readiness)
    else:
        spec = build_runtime_spec(contract, readiness)
    with EvaluatorLock(paths.output_root, spec["spec_identity"], resume=resume):
        if not resume:
            _initialise_output(contract, paths, readiness, spec)
        records, previous_hash = load_game_ledger(spec, paths.output_ledger)
        progress = _load_progress(
            paths.output_root / "progress.json", spec["spec_identity"]
        )
        completed = len(records)
        if progress["completed_games"] > completed:
            raise HeldoutEvaluationError("progress is ahead of the game ledger")
        ledger_active = (
            float(records[-1]["cumulative_active_seconds"]) if records else 0.0
        )
        base_active = max(float(progress["active_seconds"]), ledger_active)
        clock = ActiveClock(
            base_seconds=base_active,
            max_seconds=float(spec["workload"]["max_active_hours"]) * 3600.0,
            monotonic=monotonic,
        )
        launch_path = paths.output_root / "launch.json"
        if completed and not launch_path.is_file():
            raise HeldoutEvaluationError("game ledger exists without launch evidence")
        if launch_path.exists():
            launch = _strict_json(launch_path)
            _identity(launch, "launch_identity")
            if launch.get("spec_identity") != spec["spec_identity"]:
                raise HeldoutEvaluationError("launch evidence spec differs")

        records_by_id = {
            str(record["source_core_id"]): record for record in contract.records
        }
        policy: TrainingAlignedPolicy | None = None
        try:
            policy = policy_loader(paths)
            if policy.bundle_identity != EXPECTED_BUNDLE_IDENTITY:
                raise HeldoutEvaluationError("execution candidate bundle differs")
            installation = inspect_sanmill_training_installation(paths.sanmill_checkout)
            for ordinal in range(completed, 128):
                if not launch_path.exists():
                    write_new_canonical(launch_path, _launch_record(spec))
                item = spec["schedule"][ordinal]

                def persist(stage: str, stage_ply: int) -> None:
                    _write_progress(
                        paths.output_root / "progress.json",
                        _progress_body(
                            spec["spec_identity"],
                            completed_games=ordinal,
                            current_game_ordinal=ordinal,
                            current_stage=stage,
                            current_stage_ply=stage_ply,
                            active_seconds=clock.require_within_budget(),
                            ledger_tail_sha256=previous_hash,
                        ),
                    )

                persist("opening", 0)
                record = play_heldout_game(
                    spec=spec,
                    schedule_item=item,
                    corpus_record=records_by_id[item["source_core_id"]],
                    policy=policy,
                    installation=installation,
                    previous_record_sha256=previous_hash,
                    clock=clock,
                    progress_callback=persist,
                    game_factory=game_factory,
                )
                _validate_game_record(spec, record, ordinal, previous_hash)
                previous_hash = append_game_record(
                    paths.output_ledger,
                    record,
                    must_create=(ordinal == 0),
                )
                records.append(record)
                _write_progress(
                    paths.output_root / "progress.json",
                    _progress_body(
                        spec["spec_identity"],
                        completed_games=ordinal + 1,
                        current_game_ordinal=None,
                        current_stage=None,
                        current_stage_ply=0,
                        active_seconds=clock.require_within_budget(),
                        ledger_tail_sha256=previous_hash,
                    ),
                )
            report = recompute_heldout_evaluation(
                paths.output_spec, paths.output_ledger
            )
            write_new_canonical(paths.output_report, report)
            return report
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            _write_failure(
                paths,
                spec["spec_identity"],
                exc,
                completed_games=len(records),
                ledger_tail_sha256=previous_hash,
            )
            raise
        finally:
            if policy is not None:
                policy.close()

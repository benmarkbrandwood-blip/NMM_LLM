"""Strict trained-policy comparison against a frozen safe-random baseline.

The harness is deliberately evaluation-only.  It reconstructs the retained-v4
training route and the active three-specialist product route read-only, plays
complete strict-referee games against one pinned Sanmill runtime, and labels
candidate self-downgrades with corrected positional Malom values.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ai.human_db import HumanDB
from ai.malom_db import MalomDB
from ai.value_net import PhaseValueNet, ValueNet
from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.agents.specialist_router import (
    SpecialistRouter,
    load_specialist_router,
)
from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ResourceLedger,
    SafeGuidanceGameplayError,
    _checked_oracle_inventory,
    _checked_position_state,
    _checked_search_result,
    _final_positional_tier,
    _matching_move,
    _move_key,
    _normal_move,
    _phase,
    _strict_terminal_outcome,
    replay_start,
    sha256_file,
)
from learned_ai.evaluation.sanmill_safe_inducement import WDL_RANK
from learned_ai.evaluation.training_aligned_policy import (
    TrainingAlignedPolicy,
    load_training_aligned_policy,
)
from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead
from learned_ai.sentinel.config import load_config as load_sentinel_config
from learned_ai.sentinel.infer import load_advisor
from learned_ai.training.run_contract import canonical_json_bytes
from learned_ai.training.sanmill_referee import SanmillTrainingGame


PLAN_SCHEMA = "nmm.sanmill-trained-model-baseline-plan.v1"
AUTHORIZATION_SCHEMA = "nmm.sanmill-trained-model-baseline-authorization.v1"
ATTEMPT_SCHEMA = "nmm.sanmill-trained-model-baseline-attempt.v2"
ATTEMPT_AUTHORIZATION_SCHEMA = (
    "nmm.sanmill-trained-model-baseline-attempt-authorization.v2"
)
REHEARSAL_SCHEMA = "nmm.sanmill-trained-model-baseline-rehearsal.v1"
PREFLIGHT_SCHEMA = "nmm.sanmill-trained-model-baseline-preflight.v1"
GAME_SCHEMA = "nmm.sanmill-trained-model-baseline-game.v1"
RESULT_SCHEMA = "nmm.sanmill-trained-model-baseline-result.v1"

ARMS = (
    "retained-v4-free",
    "retained-v4-a-pos",
    "active-specialists-free",
    "active-specialists-a-pos",
)
CANDIDATES = ("retained-v4", "active-specialists")
PHASES = ("placement", "movement", "flying")
PRIMARY_NODE_BUDGET = 100_000
MAX_POST_START_LOGICAL_PLIES = 1536

INSTRUMENTATION_SOURCE_PATHS = (
    "ai/malom_db.py",
    "learned_ai/evaluation/human_feature_deviation_estimator_readiness.py",
    "learned_ai/evaluation/human_f0h0_feasibility.py",
    "learned_ai/evaluation/sanmill_safe_guidance_gameplay.py",
    "learned_ai/evaluation/sanmill_safe_inducement.py",
    "learned_ai/evaluation/sanmill_trained_model_baseline.py",
    "learned_ai/evaluation/sanmill_uci.py",
    "learned_ai/evaluation/training_aligned_policy.py",
    "learned_ai/sentinel/db_teacher.py",
    "learned_ai/training/sanmill_referee.py",
    "scripts/rehearse_sanmill_trained_model_baseline.py",
    "scripts/rehearse_sanmill_trained_model_baseline_attempt_002.py",
    "scripts/preflight_sanmill_trained_model_baseline.py",
    "scripts/run_sanmill_trained_model_baseline.py",
)


class TrainedModelBaselineError(SafeGuidanceGameplayError):
    """Raised when a frozen identity or runtime contract differs."""


def load_sealed(
    path: str | Path,
    *,
    schema: str,
    identity_field: str,
) -> tuple[dict[str, Any], str]:
    target = Path(path)
    raw = target.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise TrainedModelBaselineError(f"sealed schema differs: {target}")
    identity = value.get(identity_field)
    body = dict(value)
    body.pop(identity_field, None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise TrainedModelBaselineError(f"sealed identity differs: {target}")
    return value, sha256_file(target)


def load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, digest = load_sealed(
        path,
        schema=PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    if tuple(plan["experiment"]["arms"]) != ARMS:
        raise TrainedModelBaselineError("frozen arm order differs")
    return plan, digest


def load_authorization(path: str | Path) -> tuple[dict[str, Any], str]:
    return load_sealed(
        path,
        schema=AUTHORIZATION_SCHEMA,
        identity_field="authorization_identity",
    )


def load_attempt_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    return load_sealed(
        path,
        schema=ATTEMPT_SCHEMA,
        identity_field="attempt_identity",
    )


def load_attempt_authorization(path: str | Path) -> tuple[dict[str, Any], str]:
    return load_sealed(
        path,
        schema=ATTEMPT_AUTHORIZATION_SCHEMA,
        identity_field="authorization_identity",
    )


def load_rehearsal(path: str | Path) -> tuple[dict[str, Any], str]:
    return load_sealed(
        path,
        schema=REHEARSAL_SCHEMA,
        identity_field="rehearsal_identity",
    )


def load_preflight(path: str | Path) -> tuple[dict[str, Any], str]:
    return load_sealed(
        path,
        schema=PREFLIGHT_SCHEMA,
        identity_field="preflight_identity",
    )


def formal_states(
    pool: Mapping[str, Any],
    *,
    excluded_start_ids: Sequence[str],
) -> list[dict[str, Any]]:
    excluded = {str(value) for value in excluded_start_ids}
    rows = [dict(row) for row in pool["states"] if str(row["state_id"]) not in excluded]
    if len(rows) != 254 or len({str(row["state_id"]) for row in rows}) != 254:
        raise TrainedModelBaselineError("formal 254-start membership differs")
    return rows


def build_schedule(
    states: Sequence[Mapping[str, Any]],
    *,
    namespace: str,
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for start_index, state in enumerate(states):
        for color_index, candidate_color in enumerate(("W", "B")):
            unit_index = start_index * 2 + color_index
            for arm in ARMS:
                body = {
                    "namespace": namespace,
                    "start_id": str(state["state_id"]),
                    "candidate_color": candidate_color,
                    "arm": arm,
                }
                schedule.append(
                    {
                        "ordinal": len(schedule),
                        "unit_index": unit_index,
                        "start_index": start_index,
                        "start_id": str(state["state_id"]),
                        "phase": str(state["phase"]),
                        "candidate_color": candidate_color,
                        "arm": arm,
                        "game_id": canonical_sha256(body),
                    }
                )
    if len(schedule) != len(states) * 2 * len(ARMS):
        raise TrainedModelBaselineError("formal schedule cardinality differs")
    return schedule


def _candidate_from_arm(arm: str) -> str:
    if arm.startswith("retained-v4-"):
        return "retained-v4"
    if arm.startswith("active-specialists-"):
        return "active-specialists"
    raise TrainedModelBaselineError("unknown trained-model arm")


def _is_constrained(arm: str) -> bool:
    if arm not in ARMS:
        raise TrainedModelBaselineError("unknown trained-model arm")
    return arm.endswith("-a-pos")


class _RetainedV4Scorer:
    candidate_id = "retained-v4"

    def __init__(self, policy: TrainingAlignedPolicy) -> None:
        self._policy = policy

    @property
    def identity(self) -> str:
        return self._policy.bundle_identity

    def score(self, board: BoardState) -> tuple[list[dict[str, Any]], np.ndarray, str]:
        moves, logits = self._policy.score_moves(board)
        return moves, np.asarray(logits, dtype=np.float64), "training-aligned-route"

    def close(self) -> None:
        self._policy.close()


class _ProductSpecialistScorer:
    candidate_id = "active-specialists"

    def __init__(
        self,
        *,
        router: SpecialistRouter,
        human_db: HumanDB,
        identity: str,
    ) -> None:
        self._router = router
        self._human_db = human_db
        self._identity = identity
        for advisor in (router._la_open, router._la_mid, router._la_end):
            if advisor is None:
                raise TrainedModelBaselineError("specialist lookahead route is absent")
            advisor._strict = True

    @property
    def identity(self) -> str:
        return self._identity

    def score(self, board: BoardState) -> tuple[list[dict[str, Any]], np.ndarray, str]:
        model, lookahead, phase_label = self._router._pick_specialist(
            board, board.turn
        )
        if model is None or lookahead is None:
            raise TrainedModelBaselineError("preferred specialist route is absent")
        encoded = encode_position_with_lookahead(
            board,
            board.turn,
            sentinel_advisor=self._router._sentinel,
            db=None,
            value_net=self._router._value_net,
            lookahead_advisor=lookahead,
            specialist_db=None,
            strict=True,
        )
        if encoded is None or not encoded.legal_moves:
            raise TrainedModelBaselineError("specialist encoding is empty")
        legal = get_all_legal_moves(board)
        if Counter(_move_key(move) for move in encoded.legal_moves) != Counter(
            _move_key(move) for move in legal
        ):
            raise TrainedModelBaselineError("specialist legal-move inventory differs")
        features = np.asarray(encoded.feat_matrix, dtype=np.float32)
        if (
            features.ndim != 2
            or features.shape[0] != len(encoded.legal_moves)
            or features.shape[1] != 134
            or not np.isfinite(features).all()
        ):
            raise TrainedModelBaselineError("specialist feature contract differs")
        tensor = torch.as_tensor(features, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            probabilities = model.policy_probs(tensor)
        values = probabilities.detach().to(device="cpu", dtype=torch.float32).numpy()
        if (
            values.ndim != 1
            or values.shape[0] != len(encoded.legal_moves)
            or not np.isfinite(values).all()
            or float(np.sum(values)) <= 0.0
        ):
            raise TrainedModelBaselineError("specialist score contract differs")
        return (
            [dict(move) for move in encoded.legal_moves],
            values.astype(np.float64),
            phase_label,
        )

    def close(self) -> None:
        self._human_db.close()


class ModelPolicySet:
    """Own the two read-only candidate routes used by four experiment arms."""

    def __init__(
        self,
        *,
        retained_v4: _RetainedV4Scorer,
        specialists: _ProductSpecialistScorer,
    ) -> None:
        self._scorers = {
            retained_v4.candidate_id: retained_v4,
            specialists.candidate_id: specialists,
        }

    def scorer(self, arm: str) -> Any:
        return self._scorers[_candidate_from_arm(arm)]

    def close(self) -> None:
        for scorer in self._scorers.values():
            scorer.close()

    def __enter__(self) -> "ModelPolicySet":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def _require_file_identity(path: Path, expected: Mapping[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(expected["bytes"])
        or sha256_file(path) != str(expected["sha256"])
    ):
        raise TrainedModelBaselineError(f"candidate resource differs: {path}")


def load_model_policies(
    *,
    plan: Mapping[str, Any],
    root: Path,
    malom_path: Path,
    malom_manifest_path: Path,
    ledger: ResourceLedger,
) -> ModelPolicySet:
    """Load both frozen candidates without product fallback or mutable state."""
    runtime = plan["candidate_runtime"]
    v4 = runtime["retained_v4"]
    policy = load_training_aligned_policy(
        root / str(v4["bundle"]["path"]),
        human_db_path=root / str(v4["human_db"]["path"]),
        specialist_db_path=root / str(v4["specialist_db"]["path"]),
        malom_path=malom_path,
        malom_manifest_path=malom_manifest_path,
        device="cpu",
        malom_query_observer=ledger.add_malom,
    )
    if policy.bundle_identity != v4["bundle"]["identity"]:
        policy.close()
        raise TrainedModelBaselineError("retained-v4 bundle identity differs")

    specialist = runtime["active_specialists"]
    paths = {
        name: root / str(value["path"])
        for name, value in specialist["resource_files"].items()
    }
    for name, path in paths.items():
        _require_file_identity(path, specialist["resource_files"][name])
    absent = root / str(specialist["product_specialist_db"]["path"])
    if absent.exists() or specialist["product_specialist_db"]["expected"] != "absent":
        policy.close()
        raise TrainedModelBaselineError("product SpecialistDB presence changed")

    human = HumanDB(paths["human_db"], read_only=True, immutable=True)
    if not human.is_available():
        policy.close()
        raise TrainedModelBaselineError("product-route HumanDB open failed")
    sentinel = load_advisor(
        str(paths["sentinel_checkpoint"]),
        load_sentinel_config(),
        device="cpu",
    )
    value_net = PhaseValueNet.load_if_exists(
        root / str(specialist["phase_value_net_base"])
    )
    gap_net = ValueNet.load_if_exists(paths["gap_net"])
    if sentinel is None or value_net is None or gap_net is None:
        human.close()
        policy.close()
        raise TrainedModelBaselineError("product specialist component failed to load")
    router = load_specialist_router(
        ckpt_dir=root / str(specialist["checkpoint_root"]),
        sentinel_advisor=sentinel,
        db=None,
        human_db=human,
        value_net=value_net,
        gap_net=gap_net,
        specialist_db=None,
        runtime_quarantine=None,
        ply_depth=int(specialist["ply_depth"]),
    )
    if (
        router is None
        or router._spec_open is None
        or router._spec_mid is None
        or router._spec_end is None
    ):
        human.close()
        policy.close()
        raise TrainedModelBaselineError("all three active specialists are required")
    specialist_identity = canonical_sha256(
        {
            "resource_files": specialist["resource_files"],
            "product_specialist_db": specialist["product_specialist_db"],
            "ply_depth": specialist["ply_depth"],
            "presearch": specialist["presearch"],
        }
    )
    if specialist_identity != specialist["runtime_identity"]:
        human.close()
        policy.close()
        raise TrainedModelBaselineError("specialist runtime identity differs")
    return ModelPolicySet(
        retained_v4=_RetainedV4Scorer(policy),
        specialists=_ProductSpecialistScorer(
            router=router,
            human_db=human,
            identity=specialist_identity,
        ),
    )


def audit_specialist_gameai_dependency() -> dict[str, Any]:
    """Prove that warmed ``GameAI`` state is not read by specialist scoring."""
    source = inspect.getsource(SpecialistRouter)
    tree = ast.parse(source)
    reads: list[str] = []
    writes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "_gameai":
            function = next(
                (
                    parent.name
                    for parent in ast.walk(tree)
                    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node in ast.walk(parent)
                ),
                "<class>",
            )
            if isinstance(node.ctx, ast.Load):
                reads.append(function)
            elif isinstance(node.ctx, ast.Store):
                writes.append(function)
    return {
        "source_sha256": canonical_sha256(source),
        "read_methods": sorted(set(reads)),
        "write_methods": sorted(set(writes)),
        "score_path_reads_gameai": any(name == "score_moves" for name in reads),
        "presearch_effect_on_successful_argmax": False if not reads else None,
    }


def audit_instrumentation_surface(root: Path | None = None) -> dict[str, Any]:
    """Inventory every attempt-owned instrumentation boundary and signature.

    Transparent delegation is intentionally forbidden here.  Malom accounting
    is attached to the real :class:`MalomDB.query_value` throat, while the
    remaining seams are explicit validators, adapters, durable journals, or
    read-forbidden canaries.  The audit fails if a new transparent proxy or
    runtime rebinding of a Malom delegate appears in the attempt-owned files.
    """
    from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
        EstimatorAccess,
    )
    from learned_ai.evaluation.human_f0h0_feasibility import write_sealed_json
    from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
        append_resource_checkpoint,
        load_resource_checkpoints,
        write_json_atomic,
    )
    from learned_ai.evaluation.sanmill_uci import SanmillUciSession
    from learned_ai.sentinel.db_teacher import ExternalSolvedDB

    repository = root or Path(__file__).resolve().parents[2]

    def public_callables(value: type[Any]) -> list[str]:
        return sorted(
            name
            for name in dir(value)
            if not name.startswith("_") and callable(getattr(value, name))
        )

    def properties(value: type[Any]) -> list[str]:
        return sorted(
            name
            for name in dir(value)
            if not name.startswith("_")
            and isinstance(getattr(value, name), property)
        )

    expected_surfaces = {
        "MalomDB": {
            "methods": [
                "close",
                "is_available",
                "move_value",
                "query",
                "query_value",
                "terminal_move_value",
            ],
            "properties": [],
        },
        "ExternalSolvedDB": {
            "methods": [
                "best_move_result",
                "close",
                "is_available",
                "query",
                "query_all_moves",
                "query_move_quality",
                "query_state",
                "query_trajectory",
            ],
            "properties": [],
        },
        "ResourceLedger": {
            "methods": ["add_engine", "add_malom", "record", "require_within"],
            "properties": ["active_seconds"],
        },
        "TrainingAlignedPolicy": {
            "methods": ["choose_move", "close", "score_moves"],
            "properties": ["bundle_identity"],
        },
        "_RetainedV4Scorer": {
            "methods": ["close", "score"],
            "properties": ["identity"],
        },
        "_ProductSpecialistScorer": {
            "methods": ["close", "score"],
            "properties": ["identity"],
        },
        "ModelPolicySet": {
            "methods": ["close", "scorer"],
            "properties": [],
        },
        "SanmillUciSession": {
            "methods": [
                "close",
                "configure_strict_referee_profile",
                "debug_outcome",
                "export_fen",
                "history_summary",
                "legal_moves",
                "new_game",
                "position_fen",
                "position_startpos",
                "position_state",
                "probe_terminal_draw",
                "search_fixed_nodes",
                "search_logical_turn",
                "state_json",
            ],
            "properties": ["stderr_text"],
        },
        "SanmillTrainingGame": {
            "methods": [
                "apply_nmm_move",
                "assert_current_board",
                "close",
                "search_and_apply",
            ],
            "properties": ["history", "session", "state"],
        },
        "EstimatorAccess": {
            "methods": [
                "assert_allowed",
                "derive",
                "from_memberships",
                "load_decisions",
            ],
            "properties": [],
        },
    }
    types = {
        "MalomDB": MalomDB,
        "ExternalSolvedDB": ExternalSolvedDB,
        "ResourceLedger": ResourceLedger,
        "TrainingAlignedPolicy": TrainingAlignedPolicy,
        "_RetainedV4Scorer": _RetainedV4Scorer,
        "_ProductSpecialistScorer": _ProductSpecialistScorer,
        "ModelPolicySet": ModelPolicySet,
        "SanmillUciSession": SanmillUciSession,
        "SanmillTrainingGame": SanmillTrainingGame,
        "EstimatorAccess": EstimatorAccess,
    }
    observed_surfaces = {
        name: {
            "methods": public_callables(value),
            "properties": properties(value),
        }
        for name, value in types.items()
    }

    signature_checks: list[dict[str, Any]] = []

    def bind(
        name: str,
        value: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        signature = inspect.signature(value)
        try:
            signature.bind(*args, **kwargs)
        except TypeError as exc:
            signature_checks.append(
                {
                    "name": name,
                    "signature": str(signature),
                    "passed": False,
                    "error": str(exc),
                }
            )
        else:
            signature_checks.append(
                {"name": name, "signature": str(signature), "passed": True}
            )

    sentinel = object()
    bind("MalomDB.__init__.observer", MalomDB, sentinel, query_observer=sentinel)
    bind("MalomDB.query_value", MalomDB.query_value, sentinel, sentinel)
    bind("MalomDB.query", MalomDB.query, sentinel, sentinel)
    bind("MalomDB.is_available", MalomDB.is_available, sentinel)
    bind(
        "ExternalSolvedDB.__init__.observer",
        ExternalSolvedDB,
        query_observer=sentinel,
    )
    bind("ExternalSolvedDB.query_state", ExternalSolvedDB.query_state, sentinel, sentinel)
    bind("ExternalSolvedDB.query", ExternalSolvedDB.query, sentinel, sentinel)
    bind("ExternalSolvedDB.is_available", ExternalSolvedDB.is_available, sentinel)
    bind(
        "ExternalSolvedDB.query_move_quality",
        ExternalSolvedDB.query_move_quality,
        sentinel,
        sentinel,
        sentinel,
    )
    bind(
        "ExternalSolvedDB.query_all_moves",
        ExternalSolvedDB.query_all_moves,
        sentinel,
        sentinel,
        "W",
    )
    bind(
        "ExternalSolvedDB.query_trajectory",
        ExternalSolvedDB.query_trajectory,
        sentinel,
        [sentinel],
    )
    bind("ResourceLedger.add_malom", ResourceLedger.add_malom, sentinel, 1)
    bind("ResourceLedger.add_engine.default", ResourceLedger.add_engine, sentinel)
    bind("ResourceLedger.record", ResourceLedger.record, sentinel)
    bind("ResourceLedger.require_within", ResourceLedger.require_within, sentinel)
    bind(
        "TrainingAlignedPolicy.score_moves",
        TrainingAlignedPolicy.score_moves,
        sentinel,
        sentinel,
    )
    bind("_RetainedV4Scorer.score", _RetainedV4Scorer.score, sentinel, sentinel)
    bind(
        "_ProductSpecialistScorer.score",
        _ProductSpecialistScorer.score,
        sentinel,
        sentinel,
    )
    bind("ModelPolicySet.scorer", ModelPolicySet.scorer, sentinel, ARMS[0])
    bind(
        "SanmillUciSession.search_logical_turn",
        SanmillUciSession.search_logical_turn,
        sentinel,
        100_000,
        depth=None,
    )
    bind(
        "SanmillTrainingGame.apply_nmm_move",
        SanmillTrainingGame.apply_nmm_move,
        sentinel,
        sentinel,
        sentinel,
        search_result=None,
    )
    bind(
        "SanmillTrainingGame.search_and_apply",
        SanmillTrainingGame.search_and_apply,
        sentinel,
        sentinel,
        node_budget=100_000,
        depth=None,
    )
    bind(
        "SanmillTrainingGame.assert_current_board",
        SanmillTrainingGame.assert_current_board,
        sentinel,
        sentinel,
        transition=None,
    )
    bind(
        "_select_scored_move",
        _select_scored_move,
        legal_moves=[sentinel],
        scores=sentinel,
        allowed_keys={sentinel},
    )
    bind(
        "_candidate_choice",
        _candidate_choice,
        board=sentinel,
        arm=ARMS[0],
        policies=sentinel,
        database=sentinel,
        ledger=sentinel,
    )
    bind(
        "EstimatorAccess.derive",
        EstimatorAccess.derive,
        sentinel,
        "session",
        access_kind="audit",
        producer=lambda: None,
    )
    bind(
        "EstimatorAccess.assert_allowed",
        EstimatorAccess.assert_allowed,
        sentinel,
        "session",
        access_kind="audit",
    )
    bind(
        "EstimatorAccess.load_decisions",
        EstimatorAccess.load_decisions,
        sentinel,
        sentinel,
        sentinel,
        sentinel,
    )
    bind("validate_game_record", validate_game_record, sentinel)
    bind(
        "append_game_record",
        append_game_record,
        sentinel,
        sentinel,
        previous_record_sha256=None,
    )
    bind("load_game_records", load_game_records, sentinel)
    bind("compact_game", compact_game, sentinel)
    bind(
        "_finalize_game",
        _finalize_game,
        schedule_item=sentinel,
        start_state=sentinel,
        strict_start=sentinel,
        turns=sentinel,
        board=sentinel,
        terminal_state=sentinel,
        safety_cap=False,
        database=sentinel,
        ledger=sentinel,
        started=0.0,
        rehearsal_only=True,
    )
    bind(
        "append_resource_checkpoint",
        append_resource_checkpoint,
        sentinel,
        completion_index=0,
        complete_games_before=0,
        game_record=sentinel,
        resources_before=sentinel,
        resources_after=sentinel,
        previous_checkpoint_sha256=None,
    )
    bind(
        "load_resource_checkpoints",
        load_resource_checkpoints,
        sentinel,
        expected_baseline=sentinel,
        complete_games_before=0,
    )
    bind(
        "verify_resource_game_alignment",
        verify_resource_game_alignment,
        sentinel,
        sentinel,
    )
    bind("write_json_atomic", write_json_atomic, sentinel, sentinel)
    bind(
        "write_sealed_json",
        write_sealed_json,
        sentinel,
        sentinel,
        identity_field="identity",
    )
    bind(
        "_checked_search_result",
        _checked_search_result,
        sentinel,
        expected_node_budget=100_000,
    )
    bind("_checked_position_state", _checked_position_state, sentinel)
    bind("_strict_terminal_outcome", _strict_terminal_outcome, sentinel)
    bind(
        "audit_specialist_gameai_dependency",
        audit_specialist_gameai_dependency,
    )
    bind("audit_instrumentation_surface", audit_instrumentation_surface)

    transparent_proxies: list[str] = []
    attribute_interceptors: list[str] = []
    malom_delegate_rebindings: list[str] = []
    for relative in INSTRUMENTATION_SOURCE_PATHS:
        path = repository / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        parents: dict[ast.AST, ast.AST] = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

        def enclosing_function(node: ast.AST) -> str | None:
            current = parents.get(node)
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return current.name
                current = parents.get(current)
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if "__getattr__" in methods:
                    transparent_proxies.append(f"{relative}:{node.name}")
                if "__getattribute__" in methods:
                    attribute_interceptors.append(
                        f"{relative}:{node.name}.__getattribute__"
                    )
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr in {"malom", "_endgame_db"}
                        and enclosing_function(node) != "__init__"
                    ):
                        malom_delegate_rebindings.append(
                            f"{relative}:{target.lineno}:{target.attr}"
                        )

    expected_interceptors = [
        "scripts/preflight_sanmill_trained_model_baseline.py:"
        "_PoisonGameAI.__getattribute__"
    ]
    mismatches = []
    if observed_surfaces != expected_surfaces:
        mismatches.append("public method or property surface differs")
    if transparent_proxies:
        mismatches.append("transparent proxy reintroduced")
    if malom_delegate_rebindings:
        mismatches.append("Malom delegate rebound after construction")
    if sorted(attribute_interceptors) != expected_interceptors:
        mismatches.append("read-forbidden canary surface differs")
    if not all(row["passed"] for row in signature_checks):
        mismatches.append("one or more callable signatures reject the real call shape")
    return {
        "passed": not mismatches,
        "design": "single-throat MalomDB.query_value observer; no transparent proxy",
        "interception_inventory": {
            "malom_access": {
                "intercepted_methods": ["MalomDB.query_value"],
                "observer": "ResourceLedger.add_malom(count)",
                "delegated_entry_points": [
                    "MalomDB.query",
                    "ExternalSolvedDB.query_state",
                    "ExternalSolvedDB.query",
                    "ExternalSolvedDB.query_move_quality",
                    "ExternalSolvedDB.query_all_moves",
                    "ExternalSolvedDB.query_trajectory",
                ],
                "intercepted_attributes": ["MalomDB._query_observer"],
            },
            "sanmill_engine_and_referee": {
                "intercepted_methods": [
                    "SanmillUciSession.search_logical_turn",
                    "SanmillTrainingGame.apply_nmm_move",
                    "SanmillTrainingGame.assert_current_board",
                    "SanmillTrainingGame.search_and_apply",
                    "_checked_search_result",
                    "_checked_position_state",
                    "_strict_terminal_outcome",
                ],
                "intercepted_attributes": [
                    "SanmillTrainingGame.session",
                    "SanmillTrainingGame.state",
                    "SanmillTrainingGame.history",
                ],
            },
            "candidate_routes": {
                "intercepted_methods": [
                    "TrainingAlignedPolicy.score_moves",
                    "_RetainedV4Scorer.score",
                    "_ProductSpecialistScorer.score",
                    "ModelPolicySet.scorer",
                    "_select_scored_move",
                    "_candidate_choice",
                ],
                "intercepted_attributes": [
                    "TrainingAlignedPolicy.malom",
                    "TrainingAlignedPolicy.lookahead_advisor",
                    "SpecialistRouter._gameai",
                ],
            },
            "game_result_packaging": {
                "intercepted_methods": [
                    "validate_game_record",
                    "_finalize_game",
                    "compact_game",
                    "append_game_record",
                    "load_game_records",
                ],
                "intercepted_attributes": [
                    "winner",
                    "outcome_reason",
                    "candidate_score",
                    "termination_class",
                ],
            },
            "durable_resource_ledger": {
                "intercepted_methods": [
                    "ResourceLedger.add_engine",
                    "ResourceLedger.add_malom",
                    "ResourceLedger.record",
                    "append_resource_checkpoint",
                    "load_resource_checkpoints",
                    "verify_resource_game_alignment",
                    "write_json_atomic",
                    "write_sealed_json",
                ],
                "intercepted_attributes": [
                    "engine_searches",
                    "malom_queries",
                    "active_seconds",
                ],
            },
            "protected_access_guard": {
                "intercepted_methods": [
                    "EstimatorAccess.assert_allowed",
                    "EstimatorAccess.derive",
                    "EstimatorAccess.load_decisions",
                ],
                "intercepted_attributes": [
                    "official_partition_by_session",
                    "research_partition_by_session",
                    "allowed_sessions",
                    "successful",
                    "denied",
                ],
            },
            "canaries": {
                "intercepted_methods": [
                    "_PoisonGameAI.__getattribute__",
                    "audit_specialist_gameai_dependency",
                    "audit_instrumentation_surface",
                ],
                "intercepted_attributes": ["SpecialistRouter._gameai"],
            },
        },
        "source_paths": list(INSTRUMENTATION_SOURCE_PATHS),
        "expected_surfaces": expected_surfaces,
        "observed_surfaces": observed_surfaces,
        "signature_checks": signature_checks,
        "transparent_proxy_classes": sorted(transparent_proxies),
        "malom_delegate_rebindings": sorted(malom_delegate_rebindings),
        "attribute_interceptors": sorted(attribute_interceptors),
        "mismatches": mismatches,
    }


def _select_scored_move(
    *,
    legal_moves: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    allowed_keys: set[tuple[str, str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        scores.ndim != 1
        or len(legal_moves) != scores.shape[0]
        or not legal_moves
        or not np.isfinite(scores).all()
    ):
        raise TrainedModelBaselineError("candidate score vector differs")
    legal_keys = [_move_key(move) for move in legal_moves]
    if len(set(legal_keys)) != len(legal_keys):
        raise TrainedModelBaselineError("candidate legal moves are not unique")
    allowed_indices = [
        index for index, key in enumerate(legal_keys) if key in allowed_keys
    ]
    if not allowed_indices or {
        legal_keys[index] for index in allowed_indices
    } != allowed_keys:
        raise TrainedModelBaselineError("candidate allowed subset differs")
    subset = scores[np.asarray(allowed_indices, dtype=np.int64)]
    selected_relative = int(np.argmax(subset))
    selected_index = allowed_indices[selected_relative]
    selected_score = float(scores[selected_index])
    return dict(legal_moves[selected_index]), {
        "legal_move_count": len(legal_moves),
        "allowed_move_count": len(allowed_indices),
        "selected_legal_index": selected_index,
        "selected_allowed_index": selected_relative,
        "selected_score": selected_score,
        "exact_score_tie_count_within_allowed": int(np.sum(subset == selected_score)),
        "score_vector_identity": canonical_sha256(scores.tolist()),
    }


def _candidate_choice(
    *,
    board: BoardState,
    arm: str,
    policies: ModelPolicySet,
    database: MalomDB,
    ledger: ResourceLedger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    queries_before = ledger.malom_queries
    parent, inventory, queries = _checked_oracle_inventory(board, database)
    if ledger.malom_queries - queries_before != queries:
        raise TrainedModelBaselineError("Malom inventory accounting differs")
    inventory_by_key = {_move_key(move): value for move, value in inventory}
    best_rank = max(WDL_RANK[value.outcome] for value in inventory_by_key.values())
    safe_keys = {
        key
        for key, value in inventory_by_key.items()
        if WDL_RANK[value.outcome] == best_rank
    }
    if not safe_keys or any(inventory_by_key[key].outcome != parent for key in safe_keys):
        raise TrainedModelBaselineError("runtime A_pos construction differs")
    scorer = policies.scorer(arm)
    legal_moves, scores, route_phase = scorer.score(board)
    all_keys = {_move_key(move) for move in legal_moves}
    if all_keys != set(inventory_by_key):
        raise TrainedModelBaselineError("model and Malom legal inventories differ")
    allowed = safe_keys if _is_constrained(arm) else all_keys
    move, selection = _select_scored_move(
        legal_moves=legal_moves,
        scores=scores,
        allowed_keys=allowed,
    )
    after = inventory_by_key[_move_key(move)].outcome
    transition = (
        f"{parent}->{after}" if WDL_RANK[after] < WDL_RANK[parent] else None
    )
    if transition not in {None, "W->D", "W->L", "D->L"}:
        raise TrainedModelBaselineError("candidate downgrade transition differs")
    if _is_constrained(arm) and transition is not None:
        raise TrainedModelBaselineError("A_pos-constrained model self-downgraded")
    return move, {
        **selection,
        "candidate_id": scorer.candidate_id,
        "candidate_runtime_identity": scorer.identity,
        "route_phase": route_phase,
        "safety_mode": "A_pos-constrained" if _is_constrained(arm) else "free",
        "safe_set": "A_pos" if _is_constrained(arm) else None,
        "positional_only": True,
        "parent_tier": parent,
        "selected_after_tier": after,
        "a_pos_cardinality": len(safe_keys),
        "self_downgrade_transition": transition,
        "selected_move": _normal_move(move),
    }


def validate_game_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "ordinal",
        "game_id",
        "unit_index",
        "start_id",
        "phase",
        "arm",
        "candidate_color",
        "strict_start",
        "post_start_logical_plies",
        "termination_class",
        "outcome_reason",
        "winner",
        "candidate_score",
        "final_state",
        "final_positional",
        "turns",
        "self_downgrade_events",
        "game_elapsed_seconds",
    }
    if not required <= record.keys() or record.get("schema_version") != GAME_SCHEMA:
        raise TrainedModelBaselineError("trained-model game fields differ")
    rehearsal_only = record.get("rehearsal_only") is True
    if not rehearsal_only and record.get("arm") not in ARMS:
        raise TrainedModelBaselineError("trained-model game arm differs")
    if record.get("candidate_color") not in {"W", "B"}:
        raise TrainedModelBaselineError("trained-model candidate color differs")
    if record["termination_class"] == "rules_terminal":
        winner = record["winner"]
        expected_name = {None: None, "W": "white", "B": "black"}.get(
            winner, object()
        )
        expected_score = 0.5 if winner is None else float(
            winner == record["candidate_color"]
        )
        outcome = record["final_state"].get("outcome")
        if (
            not isinstance(outcome, Mapping)
            or outcome.get("terminal") is not True
            or outcome.get("reason") != record["outcome_reason"]
            or outcome.get("winner") != expected_name
            or record["candidate_score"] != expected_score
        ):
            raise TrainedModelBaselineError("trained-model terminal contract differs")
    elif record["termination_class"] == "safety_cap_incomplete":
        outcome = record["final_state"].get("outcome")
        if (
            record["candidate_score"] is not None
            or record["winner"] is not None
            or not isinstance(outcome, Mapping)
            or outcome.get("terminal") is not False
        ):
            raise TrainedModelBaselineError("safety cap was converted to a result")
    else:
        raise TrainedModelBaselineError("trained-model termination class differs")
    turns = record["turns"]
    if (
        not isinstance(turns, list)
        or not turns
        or len(turns) != record["post_start_logical_plies"]
    ):
        raise TrainedModelBaselineError("trained-model turn collection differs")
    expected_events = []
    for expected_ply, turn in enumerate(turns, start=1):
        if (
            turn.get("post_start_ply") != expected_ply
            or turn.get("mover_color") not in {"W", "B"}
            or turn.get("actor")
            not in ({"candidate", "sanmill", "scripted"} if rehearsal_only else {"candidate", "sanmill"})
            or not isinstance(turn.get("move"), Mapping)
            or not isinstance(turn.get("actions"), list)
            or "candidate_choice" not in turn
            or "engine_search" not in turn
        ):
            raise TrainedModelBaselineError("trained-model turn payload differs")
        choice = turn["candidate_choice"]
        if turn["actor"] == "candidate":
            if not isinstance(choice, Mapping) or turn["engine_search"] is not None:
                raise TrainedModelBaselineError("candidate turn payload differs")
            transition = choice.get("self_downgrade_transition")
            if transition is not None:
                expected_events.append(
                    {
                        "post_start_ply": expected_ply,
                        "phase": turn["phase"],
                        "transition": transition,
                        "move": turn["move"],
                    }
                )
        elif turn["actor"] == "sanmill" and (
            choice is not None or not isinstance(turn["engine_search"], Mapping)
        ):
            raise TrainedModelBaselineError("Sanmill turn payload differs")
        elif turn["actor"] == "scripted" and (
            choice is not None or turn["engine_search"] is not None
        ):
            raise TrainedModelBaselineError("scripted turn payload differs")
    if record["self_downgrade_events"] != expected_events:
        raise TrainedModelBaselineError("self-downgrade event collection differs")


def _finalize_game(
    *,
    schedule_item: Mapping[str, Any],
    start_state: Mapping[str, Any],
    strict_start: Mapping[str, Any],
    turns: list[dict[str, Any]],
    board: BoardState,
    terminal_state: Any,
    safety_cap: bool,
    database: MalomDB,
    ledger: ResourceLedger,
    started: float,
    rehearsal_only: bool,
) -> dict[str, Any]:
    candidate_color = str(schedule_item["candidate_color"])
    final_state = _checked_position_state(terminal_state)
    if safety_cap:
        if terminal_state.terminal:
            raise TrainedModelBaselineError("terminal state reached at safety cap")
        winner = None
        score = None
        termination = "safety_cap_incomplete"
        reason = "safety_cap_incomplete"
    else:
        winner, reason = _strict_terminal_outcome(terminal_state)
        score = 0.5 if winner is None else float(winner == candidate_color)
        termination = "rules_terminal"
    queries_before = ledger.malom_queries
    final_tier, queries = _final_positional_tier(board, database)
    if ledger.malom_queries - queries_before != queries:
        raise TrainedModelBaselineError("final Malom accounting differs")
    events = [
        {
            "post_start_ply": turn["post_start_ply"],
            "phase": turn["phase"],
            "transition": turn["candidate_choice"]["self_downgrade_transition"],
            "move": turn["move"],
        }
        for turn in turns
        if turn["actor"] == "candidate"
        and turn["candidate_choice"]["self_downgrade_transition"] is not None
    ]
    record = {
        "schema_version": GAME_SCHEMA,
        "ordinal": int(schedule_item["ordinal"]),
        "game_id": str(schedule_item["game_id"]),
        "unit_index": int(schedule_item["unit_index"]),
        "start_id": str(schedule_item["start_id"]),
        "phase": str(schedule_item["phase"]),
        "arm": str(schedule_item["arm"]),
        "candidate_color": candidate_color,
        "strict_start": dict(strict_start),
        "post_start_logical_plies": len(turns),
        "termination_class": termination,
        "outcome_reason": reason,
        "winner": winner,
        "candidate_score": score,
        "final_state": dict(final_state),
        "final_positional": {
            "side_to_move": board.turn,
            "side_to_move_wdl": final_tier,
            "history_aware": False,
        },
        "turns": turns,
        "self_downgrade_events": events,
        "game_elapsed_seconds": time.perf_counter() - started,
        "rehearsal_only": rehearsal_only,
    }
    validate_game_record(record)
    return record


def play_game(
    *,
    schedule_item: Mapping[str, Any],
    start_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    policies: ModelPolicySet,
    database: MalomDB,
    installation: Any,
    ledger: ResourceLedger,
    rehearsal_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    candidate_color = str(schedule_item["candidate_color"])
    turns: list[dict[str, Any]] = []
    safety_cap = False
    with SanmillTrainingGame(
        installation,
        seed=int(plan["sanmill_contract"]["seed"]),
    ) as game:
        board, strict_start = replay_start(game, start_state, ledger)
        for post_start_ply in range(1, MAX_POST_START_LOGICAL_PLIES + 1):
            ledger.require_within()
            mover = board.turn
            phase = _phase(board, mover)
            before_history = game.state.history_sha256
            if mover == candidate_color:
                move, choice = _candidate_choice(
                    board=board,
                    arm=str(schedule_item["arm"]),
                    policies=policies,
                    database=database,
                    ledger=ledger,
                )
                applied = game.apply_nmm_move(board, move)
                actor = "candidate"
                engine = None
            else:
                ledger.add_engine()
                result = game.session.search_logical_turn(PRIMARY_NODE_BUDGET)
                engine = _checked_search_result(
                    result,
                    expected_node_budget=PRIMARY_NODE_BUDGET,
                )
                if result.model_action is None:
                    raise TrainedModelBaselineError("Sanmill returned no model action")
                move = result.model_action
                applied = game.apply_nmm_move(board, move, search_result=result)
                actor = "sanmill"
                choice = None
            board = board.apply_move(applied.move)
            _checked_position_state(game.state)
            turns.append(
                {
                    "post_start_ply": post_start_ply,
                    "absolute_logical_ply": game.state.logical_ply_count,
                    "mover_color": mover,
                    "actor": actor,
                    "phase": phase,
                    "move": _normal_move(applied.move),
                    "actions": list(applied.actions),
                    "history_sha256_before": before_history,
                    "history_sha256_after": game.state.history_sha256,
                    "no_capture_count": game.state.no_capture_count,
                    "repetition_current_count": game.state.repetition_current_count,
                    "repetition_history_length": game.state.repetition_history_length,
                    "terminal": game.state.terminal,
                    "outcome_reason": game.state.outcome_reason,
                    "candidate_choice": choice,
                    "engine_search": engine,
                }
            )
            if game.state.terminal:
                break
        else:
            safety_cap = True
        terminal_state = game.state
    return _finalize_game(
        schedule_item=schedule_item,
        start_state=start_state,
        strict_start=strict_start,
        turns=turns,
        board=board,
        terminal_state=terminal_state,
        safety_cap=safety_cap,
        database=database,
        ledger=ledger,
        started=started,
        rehearsal_only=rehearsal_only,
    )


def replay_scripted_terminal_game(
    *,
    schedule_item: Mapping[str, Any],
    start_state: Mapping[str, Any],
    continuation_turns: Sequence[Sequence[str]],
    plan: Mapping[str, Any],
    database: MalomDB,
    installation: Any,
    ledger: ResourceLedger,
) -> dict[str, Any]:
    """Replay a real strict-terminal path without model or engine substitution."""
    if not continuation_turns:
        raise TrainedModelBaselineError("scripted continuation is empty")
    started = time.perf_counter()
    turns: list[dict[str, Any]] = []
    with SanmillTrainingGame(
        installation,
        seed=int(plan["sanmill_contract"]["seed"]),
    ) as game:
        board, strict_start = replay_start(game, start_state, ledger)
        for post_start_ply, actions in enumerate(continuation_turns, start=1):
            ledger.require_within()
            if game.state.terminal:
                raise TrainedModelBaselineError(
                    "scripted path continued after strict terminal"
                )
            mover = board.turn
            phase = _phase(board, mover)
            before_history = game.state.history_sha256
            move = _matching_move(board, actions)
            applied = game.apply_nmm_move(board, move)
            board = board.apply_move(applied.move)
            _checked_position_state(game.state)
            turns.append(
                {
                    "post_start_ply": post_start_ply,
                    "absolute_logical_ply": game.state.logical_ply_count,
                    "mover_color": mover,
                    "actor": "scripted",
                    "phase": phase,
                    "move": _normal_move(applied.move),
                    "actions": list(applied.actions),
                    "history_sha256_before": before_history,
                    "history_sha256_after": game.state.history_sha256,
                    "no_capture_count": game.state.no_capture_count,
                    "repetition_current_count": game.state.repetition_current_count,
                    "repetition_history_length": game.state.repetition_history_length,
                    "terminal": game.state.terminal,
                    "outcome_reason": game.state.outcome_reason,
                    "candidate_choice": None,
                    "engine_search": None,
                }
            )
        if not game.state.terminal:
            raise TrainedModelBaselineError(
                "scripted continuation did not reach a strict terminal"
            )
        terminal_state = game.state
    return _finalize_game(
        schedule_item=schedule_item,
        start_state=start_state,
        strict_start=strict_start,
        turns=turns,
        board=board,
        terminal_state=terminal_state,
        safety_cap=False,
        database=database,
        ledger=ledger,
        started=started,
        rehearsal_only=True,
    )


def append_game_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    previous_record_sha256: str | None,
) -> str:
    validate_game_record(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {**record, "previous_record_sha256": previous_record_sha256}
    identity = canonical_sha256(body)
    wrapper = {"record": body, "record_sha256": identity}
    with target.open("xb" if previous_record_sha256 is None else "ab") as handle:
        handle.write(canonical_json_bytes(wrapper) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return identity


def load_game_records(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    raw = target.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise TrainedModelBaselineError("trained-model ledger is partial")
    previous: str | None = None
    records: list[dict[str, Any]] = []
    for encoded in raw.splitlines():
        wrapper = json.loads(encoded)
        if not isinstance(wrapper, dict) or set(wrapper) != {
            "record",
            "record_sha256",
        }:
            raise TrainedModelBaselineError("trained-model ledger wrapper differs")
        body = wrapper["record"]
        identity = wrapper["record_sha256"]
        if (
            not isinstance(body, dict)
            or body.get("previous_record_sha256") != previous
            or canonical_sha256(body) != identity
        ):
            raise TrainedModelBaselineError("trained-model ledger chain differs")
        record = dict(body)
        record.pop("previous_record_sha256")
        validate_game_record(record)
        records.append(record)
        previous = identity
    return {
        "records": records,
        "record_count": len(records),
        "tail_record_sha256": previous,
        "file_sha256": sha256_file(target),
    }


def verify_resource_game_alignment(
    resource_recovery: Mapping[str, Any],
    game_recovery: Mapping[str, Any],
) -> None:
    checkpoints = resource_recovery.get("checkpoints")
    records = game_recovery.get("records")
    if (
        not isinstance(checkpoints, list)
        or not isinstance(records, list)
        or len(checkpoints) != len(records)
    ):
        raise TrainedModelBaselineError("resource/game ledger lengths differ")
    for checkpoint, record in zip(checkpoints, records, strict=True):
        if (
            checkpoint.get("schedule_ordinal") != record.get("ordinal")
            or checkpoint.get("game_id") != record.get("game_id")
            or checkpoint.get("game_record_identity") != canonical_sha256(record)
        ):
            raise TrainedModelBaselineError("resource/game alignment differs")


def _mean_interval(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise TrainedModelBaselineError("start-clustered values are empty")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values))
    return {
        "support": len(values),
        "mean": mean,
        "sample_standard_deviation": sd,
        "standard_error": sd / math.sqrt(len(values)),
        "half_width": half,
        "interval": [mean - half, mean + half],
        "method": "start-clustered normal 1.96 standard-error interval",
        "population_inference": False,
    }


def _score_by_start(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row["arm"] == arm and row["termination_class"] == "rules_terminal":
            grouped[str(row["start_id"])].append(float(row["candidate_score"]))
    if any(len(values) != 2 for values in grouped.values()):
        raise TrainedModelBaselineError("paired candidate colors are incomplete")
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def baseline_scores_by_start(
    baseline_manifest: Mapping[str, Any],
    *,
    arm: str,
) -> dict[str, float]:
    records = baseline_manifest.get("games")
    if not isinstance(records, list):
        raise TrainedModelBaselineError("baseline compact games are absent")
    return _score_by_start(records, arm=arm)


def _arm_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rules = [row for row in records if row["termination_class"] == "rules_terminal"]
    scores = [float(row["candidate_score"]) for row in rules]
    candidate_turns = [
        turn
        for row in records
        for turn in row["turns"]
        if turn["actor"] == "candidate"
    ]
    transitions = Counter(
        event["transition"]
        for row in records
        for event in row["self_downgrade_events"]
    )
    phase_transitions = {
        phase: dict(
            sorted(
                Counter(
                    event["transition"]
                    for row in records
                    for event in row["self_downgrade_events"]
                    if event["phase"] == phase
                ).items()
            )
        )
        for phase in PHASES
    }
    return {
        "games": len(records),
        "rules_terminal_games": len(rules),
        "safety_cap_incomplete_games": len(records) - len(rules),
        "strict_wdl": {
            "wins": sum(score == 1.0 for score in scores),
            "draws": sum(score == 0.5 for score in scores),
            "losses": sum(score == 0.0 for score in scores),
            "score_rate": statistics.fmean(scores) if scores else None,
        },
        "termination_reasons": dict(
            sorted(Counter(str(row["outcome_reason"]) for row in records).items())
        ),
        "rule_draws": {
            "games": sum(score == 0.5 for score in scores),
            "share": (
                sum(score == 0.5 for score in scores) / len(scores)
                if scores
                else None
            ),
        },
        "self_downgrade": {
            "candidate_turns": len(candidate_turns),
            "events": sum(transitions.values()),
            "event_rate_per_candidate_turn": (
                sum(transitions.values()) / len(candidate_turns)
                if candidate_turns
                else None
            ),
            "transitions": dict(sorted(transitions.items())),
            "by_action_phase": phase_transitions,
        },
    }


def _decision(interval: Mapping[str, Any], maximum_half_width: float) -> str:
    if float(interval["half_width"]) > maximum_half_width:
        return "inconclusive_precision"
    lower, upper = (float(value) for value in interval["interval"])
    if lower > 0.0:
        return "trained_arm_higher_than_safe_random"
    if upper < 0.0:
        return "safe_random_higher_than_trained_arm"
    return "no_directional_decision_at_frozen_precision"


def analyze_games(
    records: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any],
    expected_start_ids: Sequence[str],
) -> dict[str, Any]:
    if len(records) != len(expected_start_ids) * 2 * len(ARMS):
        raise TrainedModelBaselineError("formal result game count differs")
    if any(row["termination_class"] != "rules_terminal" for row in records):
        raise TrainedModelBaselineError("formal execution contains incomplete games")
    expected = set(expected_start_ids)
    if {str(row["start_id"]) for row in records} != expected:
        raise TrainedModelBaselineError("formal result start membership differs")
    baseline_random = baseline_scores_by_start(baseline_manifest, arm="random-safe")
    baseline_full = baseline_scores_by_start(baseline_manifest, arm="full-guided")
    if set(baseline_random) != expected or set(baseline_full) != expected:
        raise TrainedModelBaselineError("baseline paired start membership differs")
    maximum_half = float(plan["primary_decision"]["maximum_95_half_width"])
    primary: dict[str, Any] = {}
    arm_scores: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        arm_scores[arm] = _score_by_start(records, arm=arm)
        values = [
            arm_scores[arm][start_id] - baseline_random[start_id]
            for start_id in sorted(expected)
        ]
        interval = _mean_interval(values)
        primary[arm] = {
            **interval,
            "baseline_arm": "attempt-002:random-safe",
            "baseline_score_rate": float(
                plan["baseline"]["random_safe_score_rate"]
            ),
            "trained_arm_score_rate": statistics.fmean(
                arm_scores[arm].values()
            ),
            "decision": _decision(interval, maximum_half),
            "maximum_95_half_width": maximum_half,
        }
    free_vs_constrained = {}
    for candidate, free, constrained in (
        ("retained-v4", "retained-v4-free", "retained-v4-a-pos"),
        (
            "active-specialists",
            "active-specialists-free",
            "active-specialists-a-pos",
        ),
    ):
        free_vs_constrained[candidate] = _mean_interval(
            [
                arm_scores[constrained][start_id] - arm_scores[free][start_id]
                for start_id in sorted(expected)
            ]
        )
    constrained_vs_prior_full = {
        arm: _mean_interval(
            [
                arm_scores[arm][start_id] - baseline_full[start_id]
                for start_id in sorted(expected)
            ]
        )
        for arm in ("retained-v4-a-pos", "active-specialists-a-pos")
    }
    by_arm = {
        arm: _arm_summary([row for row in records if row["arm"] == arm])
        for arm in ARMS
    }
    by_start_phase = {
        phase: {
            arm: _arm_summary(
                [
                    row
                    for row in records
                    if row["phase"] == phase and row["arm"] == arm
                ]
            )
            for arm in ARMS
        }
        for phase in PHASES
    }
    decisions = [primary[arm]["decision"] for arm in ARMS]
    if all(value == "safe_random_higher_than_trained_arm" for value in decisions):
        overall = "all_four_trained_arms_below_safe_random"
    elif all(
        primary[arm]["trained_arm_score_rate"]
        < primary[arm]["baseline_score_rate"]
        for arm in ARMS
    ):
        overall = "all_four_point_estimates_below_safe_random"
    else:
        overall = "mixed_arm_results"
    return {
        "status": "complete_strict_terminal_analysis",
        "primary": primary,
        "overall_decision": overall,
        "by_arm": by_arm,
        "by_source_phase": by_start_phase,
        "secondary": {
            "a_pos_constraint_minus_free": free_vs_constrained,
            "a_pos_constrained_minus_prior_full_guidance": constrained_vs_prior_full,
        },
        "information_asymmetry": {
            "safe_random_has_A_pos_guarantee": True,
            "free_arms_have_A_pos_guarantee": False,
            "constrained_arms_remove_action_safety_asymmetry": True,
            "retained_v4_free_uses_malom_terminal_early_exit_in_lookahead": True,
            "active_specialists_free_uses_malom_in_inference_route": False,
        },
    }


def compact_game(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_game_record(record)
    turns = []
    for turn in record["turns"]:
        value = dict(turn)
        engine = value.get("engine_search")
        if engine is not None:
            value["engine_search_sha256"] = canonical_sha256(engine)
            value["engine_search"] = None
        turns.append(value)
    return {**record, "turns": turns}


def specialist_runtime_record(plan: Mapping[str, Any]) -> dict[str, Any]:
    runtime = plan["candidate_runtime"]["active_specialists"]
    return {
        "runtime_identity": runtime["runtime_identity"],
        "presearch": runtime["presearch"],
        "ply_depth": runtime["ply_depth"],
        "product_specialist_db": runtime["product_specialist_db"],
        "sentinel_config": asdict(load_sentinel_config()),
        "gameai_dependency_audit": audit_specialist_gameai_dependency(),
    }


__all__ = [
    "ARMS",
    "ATTEMPT_AUTHORIZATION_SCHEMA",
    "ATTEMPT_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "CANDIDATES",
    "GAME_SCHEMA",
    "MAX_POST_START_LOGICAL_PLIES",
    "PLAN_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "PRIMARY_NODE_BUDGET",
    "REHEARSAL_SCHEMA",
    "RESULT_SCHEMA",
    "ModelPolicySet",
    "TrainedModelBaselineError",
    "analyze_games",
    "append_game_record",
    "audit_instrumentation_surface",
    "audit_specialist_gameai_dependency",
    "baseline_scores_by_start",
    "build_schedule",
    "compact_game",
    "formal_states",
    "load_authorization",
    "load_attempt_authorization",
    "load_attempt_spec",
    "load_game_records",
    "load_model_policies",
    "load_plan",
    "load_preflight",
    "load_rehearsal",
    "load_sealed",
    "play_game",
    "replay_scripted_terminal_game",
    "specialist_runtime_record",
    "validate_game_record",
    "verify_resource_game_alignment",
]

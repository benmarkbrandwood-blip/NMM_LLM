"""Fixed-position node-throughput calibration for the pinned Sanmill runtime.

This module deliberately does not import a candidate model or any trainer.  It
measures the strict, single-threaded Sanmill logical-turn search used by the
fresh Sanmill-refereed lineage.  A calibration launch is an explicit,
separately authorised operation; loading and validating a plan is read-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_RULES_IDENTITY_SHA256,
    SanmillBridgeError,
    SanmillInstallation,
    SanmillUciSession,
    UciLogicalTurnResult,
    UciPositionState,
    assert_stable_legal_parity,
    project_stable_sanmill_fen,
)
from learned_ai.training.run_contract import canonical_sha256
from learned_ai.training.sanmill_referee import (
    TRAINING_REFEREE_FORMAT,
    TRAINING_REFEREE_PROFILE,
    TRAINING_REFEREE_SEMANTIC_DIGEST,
    TRAINING_REPETITION_OBSERVATION,
    TRAINING_SANMILL_BINARY_SHA256,
    TRAINING_SANMILL_BINARY_SIZE,
    TRAINING_SANMILL_COMMIT,
    TRAINING_SANMILL_TREE,
    inspect_sanmill_training_installation,
    training_installation_record,
)


PLAN_SCHEMA = "nmm.sanmill-node-throughput-calibration-plan.v1"
RESULT_SCHEMA = "nmm.sanmill-node-throughput-calibration-result.v1"
PREFLIGHT_SCHEMA = "nmm.sanmill-node-throughput-calibration-preflight.v1"
DEFAULT_PLAN_RELATIVE = Path(
    "docs/experiments/sanmill-node-throughput-calibration-v1.json"
)
DEFAULT_PATHS_RELATIVE = Path("data/training_paths.local.json")

_ROOT = Path(__file__).resolve().parents[2]
_MODES = ("cold_process", "warm_sequence")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_ROOT_KEYS = {
    "id",
    "stratum",
    "purpose",
    "history_token_count",
    "expected_state",
    "source",
}
_EXPECTED_STATE_KEYS = {
    "fen",
    "side_to_move",
    "phase",
    "action",
    "action_token_count",
    "logical_ply_count",
    "logical_plies_by_side",
    "no_capture_count",
    "repetition_current_count",
    "repetition_history_length",
    "history_sha256",
    "legal_action_count",
    "legal_actions_sha256",
    "state_identity",
}
_PLAN_KEYS = {
    "schema_version",
    "status",
    "experiment_id",
    "claim_boundary",
    "sanmill_runtime",
    "scope",
    "measurement",
    "source_game",
    "positions",
    "decision_rules",
    "plan_identity",
}


class SanmillCalibrationError(ValueError):
    """Raised when a calibration plan or result violates its contract."""


@dataclass(frozen=True)
class CalibrationRoot:
    root_id: str
    stratum: str
    purpose: str
    history_actions: tuple[str, ...]
    expected_state: Mapping[str, Any]
    source: Mapping[str, Any]


@dataclass(frozen=True)
class CalibrationPlan:
    path: Path
    raw_sha256: str
    identity: str
    experiment_id: str
    claim_boundary: str
    seed: int
    budgets: tuple[int, ...]
    repetitions: int
    protocol_timeout_seconds: float
    search_timeout_seconds: float
    positions: tuple[CalibrationRoot, ...]
    requested_node_ceiling_total: int
    process_launch_ceiling: int
    payload: Mapping[str, Any]


SessionFactory = Callable[..., SanmillUciSession]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise SanmillCalibrationError(
            f"{context} has the wrong members: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SanmillCalibrationError(f"{context} must be non-empty text")
    return value


def _require_positive_int(value: Any, *, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SanmillCalibrationError(f"{context} must be a positive integer")
    return value


def _require_positive_number(value: Any, *, context: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SanmillCalibrationError(f"{context} must be a finite positive number")
    return float(value)


def _parse_root(
    value: Any, *, index: int, source_actions: tuple[str, ...]
) -> CalibrationRoot:
    context = f"positions[{index}]"
    if not isinstance(value, Mapping):
        raise SanmillCalibrationError(f"{context} must be an object")
    _require_exact_keys(value, _ROOT_KEYS, context=context)
    history_token_count = value["history_token_count"]
    if (
        not isinstance(history_token_count, int)
        or isinstance(history_token_count, bool)
        or history_token_count < 0
        or history_token_count > len(source_actions)
    ):
        raise SanmillCalibrationError(f"{context}.history_token_count is invalid")
    expected_state = value["expected_state"]
    if not isinstance(expected_state, Mapping):
        raise SanmillCalibrationError(f"{context}.expected_state must be an object")
    _require_exact_keys(
        expected_state, _EXPECTED_STATE_KEYS, context=f"{context}.expected_state"
    )
    source = value["source"]
    if not isinstance(source, Mapping):
        raise SanmillCalibrationError(f"{context}.source must be an object")
    return CalibrationRoot(
        root_id=_require_text(value["id"], context=f"{context}.id"),
        stratum=_require_text(value["stratum"], context=f"{context}.stratum"),
        purpose=_require_text(value["purpose"], context=f"{context}.purpose"),
        history_actions=source_actions[:history_token_count],
        expected_state=dict(expected_state),
        source=dict(source),
    )


def load_calibration_plan(path: str | Path) -> CalibrationPlan:
    """Load a closed, content-addressed calibration plan."""
    plan_path = Path(path)
    try:
        raw = plan_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SanmillCalibrationError(f"cannot read calibration plan: {path}") from exc
    if not isinstance(payload, Mapping):
        raise SanmillCalibrationError("calibration plan must be an object")
    _require_exact_keys(payload, _PLAN_KEYS, context="calibration plan")
    if payload["schema_version"] != PLAN_SCHEMA:
        raise SanmillCalibrationError("unsupported calibration plan schema")
    if payload["status"] != "designed_unlaunched":
        raise SanmillCalibrationError("calibration plan status is not unlaunched")

    identity = _require_text(payload["plan_identity"], context="plan_identity")
    body = dict(payload)
    body.pop("plan_identity")
    if canonical_sha256(body) != identity:
        raise SanmillCalibrationError("calibration plan identity mismatch")

    runtime = payload["sanmill_runtime"]
    if not isinstance(runtime, Mapping):
        raise SanmillCalibrationError("sanmill_runtime must be an object")
    expected_runtime = {
        "commit": TRAINING_SANMILL_COMMIT,
        "tree": TRAINING_SANMILL_TREE,
        "binary_sha256": TRAINING_SANMILL_BINARY_SHA256,
        "binary_size": TRAINING_SANMILL_BINARY_SIZE,
        "strict_referee_profile": TRAINING_REFEREE_PROFILE,
        "strict_referee_semantic_digest": TRAINING_REFEREE_SEMANTIC_DIGEST,
        "rules_identity_sha256": EXPECTED_RULES_IDENTITY_SHA256,
    }
    if dict(runtime) != expected_runtime:
        raise SanmillCalibrationError("calibration Sanmill runtime pin drifted")

    scope = payload["scope"]
    expected_scope = {
        "candidate_model": False,
        "trainer": False,
        "optimizer": False,
        "checkpoint": False,
        "human_db": False,
        "specialist_db": False,
        "malom_db": False,
        "perfect_db": False,
        "opening_book": False,
        "patches_or_random_fallback": False,
    }
    if not isinstance(scope, Mapping) or dict(scope) != expected_scope:
        raise SanmillCalibrationError("calibration scope is not engine-only")

    measurement = payload["measurement"]
    if not isinstance(measurement, Mapping):
        raise SanmillCalibrationError("measurement must be an object")
    expected_measurement_keys = {
        "modes",
        "node_budgets",
        "repetitions_per_cell",
        "seed",
        "depth_limit",
        "protocol_timeout_seconds",
        "search_timeout_seconds",
        "ordering",
        "percentile_method",
        "requested_node_ceiling_total",
        "process_launch_ceiling",
    }
    _require_exact_keys(
        measurement, expected_measurement_keys, context="measurement"
    )
    if measurement["modes"] != list(_MODES):
        raise SanmillCalibrationError("calibration modes changed")
    budgets_raw = measurement["node_budgets"]
    if not isinstance(budgets_raw, list):
        raise SanmillCalibrationError("node_budgets must be an array")
    budgets = tuple(
        _require_positive_int(value, context="node_budgets item")
        for value in budgets_raw
    )
    if tuple(sorted(set(budgets))) != budgets:
        raise SanmillCalibrationError(
            "node_budgets must be unique and strictly increasing"
        )
    repetitions = _require_positive_int(
        measurement["repetitions_per_cell"],
        context="repetitions_per_cell",
    )
    seed = measurement["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SanmillCalibrationError("measurement seed must be non-negative")
    if measurement["depth_limit"] is not None:
        raise SanmillCalibrationError(
            "node calibration must not add an independent depth ceiling"
        )
    if measurement["ordering"] != "chronological-roots-rotated-budgets-v1":
        raise SanmillCalibrationError("unknown calibration ordering")
    if measurement["percentile_method"] != "nearest-rank":
        raise SanmillCalibrationError("unknown percentile method")

    source_game = payload["source_game"]
    if not isinstance(source_game, Mapping):
        raise SanmillCalibrationError("source_game must be an object")
    _require_exact_keys(
        source_game,
        {
            "evidence_relative_path",
            "evidence_sha256",
            "semantic_identity",
            "actions",
            "actions_sha256",
        },
        context="source_game",
    )
    source_actions_raw = source_game["actions"]
    if not isinstance(source_actions_raw, list) or any(
        not isinstance(action, str) or not action for action in source_actions_raw
    ):
        raise SanmillCalibrationError(
            "source_game.actions must be an array of action tokens"
        )
    source_actions = tuple(source_actions_raw)
    if canonical_sha256(list(source_actions)) != source_game["actions_sha256"]:
        raise SanmillCalibrationError("source game action identity mismatch")
    evidence_relative = Path(_require_text(
        source_game["evidence_relative_path"],
        context="source_game.evidence_relative_path",
    ))
    if evidence_relative.is_absolute():
        raise SanmillCalibrationError("source game evidence path must be relative")
    evidence_path = (_ROOT / evidence_relative).resolve(strict=False)
    try:
        evidence_path.relative_to(_ROOT)
    except ValueError as exc:
        raise SanmillCalibrationError(
            "source game evidence must remain inside the repository"
        ) from exc
    if not evidence_path.is_file():
        raise SanmillCalibrationError("source game evidence is missing")
    if _sha256_file(evidence_path) != source_game["evidence_sha256"]:
        raise SanmillCalibrationError("source game evidence SHA-256 mismatch")
    try:
        source_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        source_run = source_evidence["reproducibility"]["run"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SanmillCalibrationError("source game evidence has the wrong shape") from exc
    if source_run.get("semantic_identity") != source_game["semantic_identity"]:
        raise SanmillCalibrationError("source game semantic identity mismatch")
    if source_run.get("replayed_action_tokens") != list(source_actions):
        raise SanmillCalibrationError("source game actions differ from evidence")

    positions_raw = payload["positions"]
    if not isinstance(positions_raw, list) or not positions_raw:
        raise SanmillCalibrationError("positions must be a non-empty array")
    positions = tuple(
        _parse_root(value, index=index, source_actions=source_actions)
        for index, value in enumerate(positions_raw)
    )
    ids = [position.root_id for position in positions]
    if len(ids) != len(set(ids)):
        raise SanmillCalibrationError("calibration position IDs must be unique")
    logical_counts = [
        int(position.expected_state["logical_ply_count"])
        for position in positions
    ]
    if logical_counts != sorted(logical_counts):
        raise SanmillCalibrationError(
            "calibration roots must remain in chronological order"
        )
    for position in positions:
        if position.expected_state["action_token_count"] != len(
            position.history_actions
        ):
            raise SanmillCalibrationError(
                f"calibration root action count drifted: {position.root_id}"
            )

    requested_total = sum(budgets) * len(positions) * repetitions * len(_MODES)
    process_ceiling = (
        len(positions) * len(budgets) * repetitions
        + len(budgets) * repetitions
    )
    if measurement["requested_node_ceiling_total"] != requested_total:
        raise SanmillCalibrationError("requested node ceiling total is inconsistent")
    if measurement["process_launch_ceiling"] != process_ceiling:
        raise SanmillCalibrationError("process launch ceiling is inconsistent")

    decision_rules = payload["decision_rules"]
    if not isinstance(decision_rules, Mapping):
        raise SanmillCalibrationError("decision_rules must be an object")
    if decision_rules.get("auto_select_training_ladder") is not False:
        raise SanmillCalibrationError("calibration must not auto-select a ladder")

    return CalibrationPlan(
        path=plan_path,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        identity=identity,
        experiment_id=_require_text(
            payload["experiment_id"], context="experiment_id"
        ),
        claim_boundary=_require_text(
            payload["claim_boundary"], context="claim_boundary"
        ),
        seed=seed,
        budgets=budgets,
        repetitions=repetitions,
        protocol_timeout_seconds=_require_positive_number(
            measurement["protocol_timeout_seconds"],
            context="protocol_timeout_seconds",
        ),
        search_timeout_seconds=_require_positive_number(
            measurement["search_timeout_seconds"],
            context="search_timeout_seconds",
        ),
        positions=positions,
        requested_node_ceiling_total=requested_total,
        process_launch_ceiling=process_ceiling,
        payload=dict(payload),
    )


def state_calibration_record(state: UciPositionState) -> dict[str, Any]:
    """Return the stable state fields frozen by a calibration fixture."""
    referee = state.strict_referee_identity
    if referee is None:
        raise SanmillBridgeError("Sanmill omitted strict-referee identity")
    if referee.portable_record() != {
        "format": TRAINING_REFEREE_FORMAT,
        "profile": TRAINING_REFEREE_PROFILE,
        "repetitionObservation": TRAINING_REPETITION_OBSERVATION,
        "originCounted": True,
        "semanticDigest": TRAINING_REFEREE_SEMANTIC_DIGEST,
    }:
        raise SanmillBridgeError("Sanmill strict-referee identity drifted")
    if state.rules_identity_sha256 != EXPECTED_RULES_IDENTITY_SHA256:
        raise SanmillBridgeError("Sanmill rule identity drifted")
    if state.terminal or state.removal_pending:
        raise SanmillBridgeError(
            "calibration roots must be ongoing complete-turn boundaries"
        )
    board = project_stable_sanmill_fen(state.fen)
    assert_stable_legal_parity(board, state.legal_actions)
    body = {
        "fen": state.fen,
        "side_to_move": state.side_to_move,
        "phase": state.phase,
        "action": state.action,
        "action_token_count": state.action_token_count,
        "logical_ply_count": state.logical_ply_count,
        "logical_plies_by_side": list(state.logical_plies_by_side),
        "no_capture_count": state.no_capture_count,
        "repetition_current_count": state.repetition_current_count,
        "repetition_history_length": state.repetition_history_length,
        "history_sha256": state.history_sha256,
        "legal_action_count": len(state.legal_actions),
        "legal_actions_sha256": canonical_sha256(list(state.legal_actions)),
    }
    return {**body, "state_identity": canonical_sha256(body)}


def _position_and_validate(
    session: SanmillUciSession, root: CalibrationRoot
) -> tuple[UciPositionState, dict[str, Any]]:
    session.position_startpos(root.history_actions)
    state = session.state_json()
    observed = state_calibration_record(state)
    if observed != dict(root.expected_state):
        raise SanmillBridgeError(
            f"calibration fixture state drifted: {root.root_id}"
        )
    return state, observed


def _resolve_training_checkout(paths_config: str | Path) -> Path:
    path = Path(paths_config)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SanmillCalibrationError(
            f"cannot read local path registry: {paths_config}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SanmillCalibrationError("local path registry must be an object")
    value = payload.get("sanmill_training_checkout")
    if not isinstance(value, str) or not value:
        raise SanmillCalibrationError(
            "local path registry omits sanmill_training_checkout"
        )
    checkout = Path(value)
    if not checkout.is_absolute():
        checkout = _ROOT / checkout
    return checkout.resolve(strict=False)


def inspect_calibration_fixtures(
    plan: CalibrationPlan,
    installation: SanmillInstallation,
    *,
    session_factory: SessionFactory = SanmillUciSession,
) -> list[dict[str, Any]]:
    """Replay and validate every root without launching a search."""
    observations: list[dict[str, Any]] = []
    with session_factory(
        installation,
        seed=plan.seed,
        protocol_timeout=plan.protocol_timeout_seconds,
        search_timeout=plan.search_timeout_seconds,
    ) as session:
        session.configure_strict_referee_profile(TRAINING_REFEREE_PROFILE)
        for root in plan.positions:
            session.new_game()
            _, observed = _position_and_validate(session, root)
            observations.append({"root_id": root.root_id, "state": observed})
    return observations


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SanmillCalibrationError(f"cannot inspect NMM_LLM Git state: {detail}")
    return result.stdout.strip()


def inspect_published_source(*, require_published: bool) -> dict[str, Any]:
    """Bind execution to a clean commit and, when required, its upstream."""
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise SanmillCalibrationError(
            "NMM_LLM tracked worktree must be clean for calibration"
        )
    branch = _git("branch", "--show-current")
    if branch != "dev":
        raise SanmillCalibrationError("calibration must run from dev")
    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    upstream = _git("rev-parse", "@{upstream}")
    published = commit == upstream
    if require_published and not published:
        raise SanmillCalibrationError(
            "calibration source commit must already be published"
        )
    return {
        "branch": branch,
        "commit": commit,
        "tree": tree,
        "upstream_commit": upstream,
        "tracked_worktree": "clean",
        "published": published,
    }


def tracked_plan_record(plan: CalibrationPlan) -> dict[str, str]:
    """Require the selected plan to be a tracked file under this repository."""
    try:
        relative = plan.path.resolve().relative_to(_ROOT).as_posix()
    except ValueError as exc:
        raise SanmillCalibrationError(
            "calibration plan must be inside the NMM_LLM repository"
        ) from exc
    _git("ls-files", "--error-unmatch", "--", relative)
    return {
        "relative_path": relative,
        "raw_sha256": plan.raw_sha256,
        "identity": plan.identity,
    }


def validate_calibration_output(path: str | Path) -> Path:
    """Confine generated evidence to the ignored diagnostics directory."""
    target = Path(path).resolve(strict=False)
    diagnostics = (_ROOT / "out" / "diagnostics").resolve(strict=False)
    try:
        target.relative_to(diagnostics)
    except ValueError as exc:
        raise SanmillCalibrationError(
            "calibration output must be under out/diagnostics"
        ) from exc
    if target.suffix.lower() != ".json":
        raise SanmillCalibrationError("calibration output must be a JSON file")
    if target.exists():
        raise FileExistsError(f"calibration result already exists: {target}")
    return target


def preflight_calibration(
    plan_path: str | Path,
    paths_config: str | Path,
    *,
    require_published: bool = True,
    session_factory: SessionFactory = SanmillUciSession,
) -> dict[str, Any]:
    """Validate source, runtime, and fixtures without running timed search."""
    plan = load_calibration_plan(plan_path)
    plan_record = tracked_plan_record(plan)
    source = inspect_published_source(require_published=require_published)
    checkout = _resolve_training_checkout(paths_config)
    installation = inspect_sanmill_training_installation(checkout)
    fixtures = inspect_calibration_fixtures(
        plan, installation, session_factory=session_factory
    )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready_for_authorized_calibration",
        "launch_authorized": False,
        "plan": plan_record,
        "source": source,
        "sanmill": training_installation_record(installation, seed=plan.seed),
        "fixtures": fixtures,
        "bounded_work": {
            "positions": len(plan.positions),
            "budgets": list(plan.budgets),
            "repetitions_per_cell": plan.repetitions,
            "requested_node_ceiling_total": plan.requested_node_ceiling_total,
            "process_launch_ceiling": plan.process_launch_ceiling,
        },
        "claim_boundary": plan.claim_boundary,
    }


def _validate_search_result(
    result: UciLogicalTurnResult, *, root_id: str, node_budget: int
) -> None:
    if result.status != "ok":
        raise SanmillBridgeError(
            f"calibration search did not return a legal turn: {root_id}"
        )
    if result.node_budget != node_budget:
        raise SanmillBridgeError("calibration search changed its node ceiling")
    if result.total_nodes != result.primary_nodes + result.removal_nodes:
        raise SanmillBridgeError("Sanmill node accounting is inconsistent")
    if result.total_nodes <= 0 or result.total_nodes > node_budget:
        raise SanmillBridgeError("Sanmill actual nodes fall outside the ceiling")
    if not result.full_turn_actions or result.search_calls <= 0:
        raise SanmillBridgeError("Sanmill omitted logical-turn search evidence")


def _sample_record(
    *,
    mode: str,
    repetition: int,
    root: CalibrationRoot,
    node_budget: int,
    result: UciLogicalTurnResult,
    process_start_seconds: float | None,
    hash_reset_seconds: float | None,
    position_replay_seconds: float,
) -> dict[str, Any]:
    _validate_search_result(result, root_id=root.root_id, node_budget=node_budget)
    semantic = result.semantic_record()
    elapsed = result.elapsed_seconds
    nps = result.total_nodes / elapsed if elapsed > 0 else None
    return {
        "mode": mode,
        "repetition": repetition,
        "root_id": root.root_id,
        "stratum": root.stratum,
        "node_ceiling": node_budget,
        "process_start_seconds": process_start_seconds,
        "hash_reset_seconds": hash_reset_seconds,
        "position_replay_seconds": position_replay_seconds,
        "search_seconds": elapsed,
        "actual_nodes": result.total_nodes,
        "node_utilization": result.total_nodes / node_budget,
        "nodes_per_second": nps,
        "compound_turn": len(result.full_turn_actions) == 2,
        "semantic_result": semantic,
        "semantic_result_sha256": canonical_sha256(semantic),
    }


def _open_session(
    installation: SanmillInstallation,
    plan: CalibrationPlan,
    session_factory: SessionFactory,
) -> tuple[SanmillUciSession, float]:
    started = time.perf_counter()
    session = session_factory(
        installation,
        seed=plan.seed,
        protocol_timeout=plan.protocol_timeout_seconds,
        search_timeout=plan.search_timeout_seconds,
    )
    try:
        session.configure_strict_referee_profile(TRAINING_REFEREE_PROFILE)
    except BaseException:
        session.close()
        raise
    return session, time.perf_counter() - started


def _run_cold_samples(
    plan: CalibrationPlan,
    installation: SanmillInstallation,
    *,
    session_factory: SessionFactory,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    roots = plan.positions
    for repetition in range(plan.repetitions):
        rotated_budgets = plan.budgets[repetition % len(plan.budgets) :] + (
            plan.budgets[: repetition % len(plan.budgets)]
        )
        for budget_index, budget in enumerate(rotated_budgets):
            offset = (repetition + budget_index) % len(roots)
            rotated_roots = roots[offset:] + roots[:offset]
            for root in rotated_roots:
                session, process_seconds = _open_session(
                    installation, plan, session_factory
                )
                try:
                    reset_started = time.perf_counter()
                    session.new_game()
                    reset_seconds = time.perf_counter() - reset_started
                    replay_started = time.perf_counter()
                    _, before = _position_and_validate(session, root)
                    replay_seconds = time.perf_counter() - replay_started
                    result = session.search_logical_turn(budget)
                    after = state_calibration_record(session.state_json())
                    if after != before:
                        raise SanmillBridgeError(
                            "Sanmill logical search mutated the calibration root"
                        )
                    samples.append(
                        _sample_record(
                            mode="cold_process",
                            repetition=repetition,
                            root=root,
                            node_budget=budget,
                            result=result,
                            process_start_seconds=process_seconds,
                            hash_reset_seconds=reset_seconds,
                            position_replay_seconds=replay_seconds,
                        )
                    )
                finally:
                    session.close()
    return samples


def _run_warm_samples(
    plan: CalibrationPlan,
    installation: SanmillInstallation,
    *,
    session_factory: SessionFactory,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for repetition in range(plan.repetitions):
        rotated_budgets = plan.budgets[repetition % len(plan.budgets) :] + (
            plan.budgets[: repetition % len(plan.budgets)]
        )
        for budget in rotated_budgets:
            session, process_seconds = _open_session(
                installation, plan, session_factory
            )
            try:
                reset_started = time.perf_counter()
                session.new_game()
                reset_seconds = time.perf_counter() - reset_started
                for root_index, root in enumerate(plan.positions):
                    replay_started = time.perf_counter()
                    _, before = _position_and_validate(session, root)
                    replay_seconds = time.perf_counter() - replay_started
                    result = session.search_logical_turn(budget)
                    after = state_calibration_record(session.state_json())
                    if after != before:
                        raise SanmillBridgeError(
                            "Sanmill logical search mutated the calibration root"
                        )
                    samples.append(
                        _sample_record(
                            mode="warm_sequence",
                            repetition=repetition,
                            root=root,
                            node_budget=budget,
                            result=result,
                            process_start_seconds=(
                                process_seconds if root_index == 0 else None
                            ),
                            hash_reset_seconds=(
                                reset_seconds if root_index == 0 else None
                            ),
                            position_replay_seconds=replay_seconds,
                        )
                    )
            finally:
                session.close()
    return samples


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise SanmillCalibrationError("cannot summarize an empty sample")
    if not 0 < percentile <= 1:
        raise SanmillCalibrationError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise SanmillCalibrationError("summary metrics must be finite and non-empty")
    median = statistics.median(values)
    return {
        "minimum": min(values),
        "median": median,
        "p90_nearest_rank": nearest_rank(values, 0.90),
        "maximum": max(values),
        "median_absolute_deviation": statistics.median(
            abs(value - median) for value in values
        ),
    }


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[
            (
                str(sample["mode"]),
                str(sample["root_id"]),
                int(sample["node_ceiling"]),
            )
        ].append(sample)
    summary: list[dict[str, Any]] = []
    for (mode, root_id, budget), group in sorted(groups.items()):
        semantic_hashes = sorted(
            {str(sample["semantic_result_sha256"]) for sample in group}
        )
        if len(semantic_hashes) != 1:
            raise SanmillBridgeError(
                f"non-deterministic semantic search result: {mode}/{root_id}/{budget}"
            )
        nps = [
            float(sample["nodes_per_second"])
            for sample in group
            if sample["nodes_per_second"] is not None
        ]
        if len(nps) != len(group):
            raise SanmillCalibrationError("search timing cannot be zero")
        summary.append(
            {
                "mode": mode,
                "root_id": root_id,
                "node_ceiling": budget,
                "samples": len(group),
                "semantic_result_sha256": semantic_hashes[0],
                "search_seconds": _metric_summary(
                    [float(sample["search_seconds"]) for sample in group]
                ),
                "actual_nodes": _metric_summary(
                    [float(sample["actual_nodes"]) for sample in group]
                ),
                "node_utilization": _metric_summary(
                    [float(sample["node_utilization"]) for sample in group]
                ),
                "nodes_per_second": _metric_summary(nps),
                "compound_turn_samples": sum(
                    bool(sample["compound_turn"]) for sample in group
                ),
            }
        )
    return summary


def summarize_overheads(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize process, hash-reset, and history-replay overhead separately."""
    process: dict[str, list[float]] = defaultdict(list)
    reset: dict[str, list[float]] = defaultdict(list)
    replay: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sample in samples:
        mode = str(sample["mode"])
        root_id = str(sample["root_id"])
        if sample["process_start_seconds"] is not None:
            process[mode].append(float(sample["process_start_seconds"]))
        if sample["hash_reset_seconds"] is not None:
            reset[mode].append(float(sample["hash_reset_seconds"]))
        replay[(mode, root_id)].append(float(sample["position_replay_seconds"]))
    return {
        "process_start_seconds": {
            mode: {"samples": len(values), **_metric_summary(values)}
            for mode, values in sorted(process.items())
        },
        "hash_reset_seconds": {
            mode: {"samples": len(values), **_metric_summary(values)}
            for mode, values in sorted(reset.items())
        },
        "position_replay_seconds": [
            {
                "mode": mode,
                "root_id": root_id,
                "samples": len(values),
                **_metric_summary(values),
            }
            for (mode, root_id), values in sorted(replay.items())
        ],
    }


def _host_record() -> dict[str, Any]:
    power_scheme: str | None = None
    if os.name == "nt":
        result = subprocess.run(
            ["powercfg", "/getactivescheme"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            power_scheme = result.stdout.strip()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "logical_cpu_count": os.cpu_count(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
        "active_power_scheme": power_scheme,
    }


def run_calibration(
    plan: CalibrationPlan,
    installation: SanmillInstallation,
    *,
    source: Mapping[str, Any],
    run_id: str,
    invocation: Sequence[str],
    session_factory: SessionFactory = SanmillUciSession,
) -> dict[str, Any]:
    """Run the bounded engine-only calibration after explicit authorization."""
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise SanmillCalibrationError("run_id has an invalid evidence identifier")
    if not invocation or any(
        not isinstance(item, str) or not item for item in invocation
    ):
        raise SanmillCalibrationError("calibration invocation must be non-empty")
    started_at = _utc_now()
    host_before = _host_record()
    started = time.perf_counter()
    cold = _run_cold_samples(
        plan, installation, session_factory=session_factory
    )
    warm = _run_warm_samples(
        plan, installation, session_factory=session_factory
    )
    elapsed = time.perf_counter() - started
    host_after = _host_record()
    if host_before["active_power_scheme"] != host_after["active_power_scheme"]:
        raise SanmillCalibrationError("active power scheme changed during calibration")
    samples = cold + warm
    expected_samples = (
        len(plan.positions) * len(plan.budgets) * plan.repetitions * len(_MODES)
    )
    if len(samples) != expected_samples:
        raise SanmillCalibrationError("calibration sample count is incomplete")
    summary = summarize_samples(samples)
    body = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "claim_boundary": plan.claim_boundary,
        "invocation": list(invocation),
        "plan": {
            "relative_path": plan.path.resolve().relative_to(_ROOT).as_posix(),
            "raw_sha256": plan.raw_sha256,
            "identity": plan.identity,
        },
        "source": dict(source),
        "sanmill": training_installation_record(installation, seed=plan.seed),
        "host_before": host_before,
        "host_after": host_after,
        "bounded_work": {
            "positions": len(plan.positions),
            "node_budgets": list(plan.budgets),
            "repetitions_per_cell": plan.repetitions,
            "samples": len(samples),
            "requested_node_ceiling_total": plan.requested_node_ceiling_total,
            "process_launch_ceiling": plan.process_launch_ceiling,
        },
        "samples": samples,
        "summary": summary,
        "overhead_summary": summarize_overheads(samples),
        "interpretation": {
            "auto_selected_training_ladder": False,
            "end_to_end_games_per_hour_measured": False,
            "model_or_optimizer_work_measured": False,
            "next_gate": (
                "review actual-node utilization, latency tails, cold/warm "
                "effects, then separately approve any integrated route probe"
            ),
        },
    }
    return {**body, "report_identity": canonical_sha256(body)}


def publish_calibration_result(path: str | Path, report: Mapping[str, Any]) -> None:
    """Atomically publish a new result without replacing prior evidence."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"calibration result already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise FileExistsError(f"calibration result already exists: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_local_installation(paths_config: str | Path) -> SanmillInstallation:
    """Resolve and verify the pinned training runtime from the ignored registry."""
    return inspect_sanmill_training_installation(
        _resolve_training_checkout(paths_config)
    )

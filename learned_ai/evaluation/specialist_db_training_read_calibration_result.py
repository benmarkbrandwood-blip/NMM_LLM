"""Fail-closed result analysis for SpecialistDB training-read calibration."""

from __future__ import annotations

import math
import sqlite3
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from ai.human_db import HumanDB
from game.board import BoardState
from game.notation import encode_move
from game.rules import get_game_phase
from learned_ai.data.data_contract import load_dataset_manifest
from learned_ai.evaluation.malom_policy_auxiliary_calibration_result import (
    summarize_game_rows,
    summarize_update_rows,
)
from learned_ai.evaluation.mill_bonus_ablation_result import (
    MillBonusAblationResultError,
    _artifact_record,
    _readiness_arm,
    _require_finite,
    _require_int,
    _sha256_file,
    _strict_json,
    _strict_jsonl,
    _validate_authorization,
    _validate_controller_completion,
    _validate_manifest,
    _validate_policy_health,
)
from learned_ai.models.lookahead_advisor import LookaheadAdvisor
from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.training import managed_generalist as managed
from learned_ai.training.checkpoint_envelope import load_checkpoint
from learned_ai.training.managed_generalist import (
    load_managed_authorization,
    load_managed_plan,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation.mill_bonus_ablation_readiness import _repository_path
from learned_ai.validation.specialist_db_training_read_calibration import (
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_REPORT as DEFAULT_READINESS_REPORT,
    READINESS_SCHEMA,
    RESULT_SCHEMA,
    _assert_plan_semantics,
    _ordered_arms,
    load_specialist_read_calibration_contract,
)
from scripts import train_s_gen_v2 as trainer


DEFAULT_RESULT = Path("out/specialist-db-training-read-calibration-v1/result.json")
DEFAULT_MALOM_MANIFEST = Path("data/manifests/malom-sector-corrected-v1.json")
READ_FIELDS = (
    "queries",
    "rows_present",
    "theoretical_available",
    "empirical_available",
    "projections_returned",
    "empirical_suppressed",
)
PHASES = ("placement", "movement", "flying")
ROLLING_GAME_WINDOW = 50


SpecialistReadCalibrationResultError = MillBonusAblationResultError


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-(positive * np.log(positive)).sum())


def _rolling_read_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    window: int = ROLLING_GAME_WINDOW,
) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for end in range(window, len(rows) + 1):
        sample = rows[end - window : end]
        curve.append(
            {
                "game": int(sample[-1]["game"]),
                "window_games": window,
                **{
                    field: sum(int(row[f"specialist_read_{field}"]) for row in sample)
                    / window
                    for field in READ_FIELDS
                },
            }
        )
    return curve


def summarize_specialist_read_game_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    expected_games: int,
    expected_schedule_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate game logs and preserve the read intervention telemetry."""
    base = summarize_game_rows(
        rows,
        coefficient=0.0,
        expected_games=expected_games,
        expected_schedule_counts=expected_schedule_counts,
    )
    totals = Counter({field: 0 for field in READ_FIELDS})
    raw: list[dict[str, Any]] = []
    for expected_game, row in enumerate(rows, start=1):
        if row.get("specialist_read_mode") != mode:
            raise SpecialistReadCalibrationResultError(
                f"game {expected_game} SpecialistDB read mode differs"
            )
        values = {
            field: _require_int(
                row.get(f"specialist_read_{field}"),
                field=f"game[{expected_game}].specialist_read_{field}",
            )
            for field in READ_FIELDS
        }
        if (
            values["rows_present"] > values["queries"]
            or values["theoretical_available"] > values["rows_present"]
            or values["empirical_available"] > values["rows_present"]
            or values["projections_returned"] > values["rows_present"]
            or values["empirical_suppressed"] > values["empirical_available"]
        ):
            raise SpecialistReadCalibrationResultError(
                f"game {expected_game} SpecialistDB read counters contradict"
            )
        if mode == "full" and values["empirical_suppressed"] != 0:
            raise SpecialistReadCalibrationResultError(
                "full read mode suppressed empirical evidence"
            )
        if mode == "theoretical-only" and (
            values["empirical_suppressed"] != values["empirical_available"]
        ):
            raise SpecialistReadCalibrationResultError(
                "theoretical-only mode did not suppress every empirical read"
            )
        totals.update(values)
        raw.append(
            {
                "game": expected_game,
                "mode": mode,
                **values,
            }
        )
    engaged = totals["empirical_available"] > 0 and (
        (mode == "full" and totals["empirical_suppressed"] == 0)
        or (
            mode == "theoretical-only"
            and totals["empirical_suppressed"] == totals["empirical_available"]
        )
    )
    base["specialist_read_intervention"] = {
        "mode": mode,
        "totals": dict(totals),
        "engaged": engaged,
        "curves": {
            "interpretation": (
                "observed per-game SpecialistDB reads only; incomplete "
                "leading rolling windows are omitted"
            ),
            "raw": raw,
            "rolling_50_complete_windows_only": _rolling_read_curve(rows),
        },
    }
    return base


def _load_policy(
    model_config: Mapping[str, Any],
    model_state: Mapping[str, torch.Tensor],
) -> ScaffoldedPolicyNet:
    model = ScaffoldedPolicyNet.from_config(dict(model_config)).to("cpu")
    model.load_state_dict(dict(model_state))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _scratch_policy(model_config: Mapping[str, Any], seed: int) -> ScaffoldedPolicyNet:
    trainer._initialize_training_rngs(seed)
    model, start_game, _best, difficulty, source = trainer._load_model(
        torch.device("cpu"),
        None,
        tuple(model_config["policy_hidden"]),
        start_mode="fresh",
    )
    if (start_game, difficulty, source) != (0, trainer.DIFF_START, "scratch"):
        raise SpecialistReadCalibrationResultError(
            "scratch reconstruction contract drifted"
        )
    if model.get_config() != dict(model_config):
        raise SpecialistReadCalibrationResultError(
            "scratch model configuration differs"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _sqlite_asset_identity(path: Path) -> str:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        checked = connection.execute("PRAGMA quick_check").fetchone()
        if checked is None or checked[0] != "ok":
            raise SpecialistReadCalibrationResultError(
                "HumanDB quick_check did not return ok"
            )
        stat = path.stat()
        return canonical_sha256(
            {
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "page_count": connection.execute("PRAGMA page_count").fetchone()[0],
                "page_size": connection.execute("PRAGMA page_size").fetchone()[0],
                "schema_version": connection.execute(
                    "PRAGMA schema_version"
                ).fetchone()[0],
                "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
            }
        )
    finally:
        connection.close()


def _resolve_config_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def _policy_probabilities(
    model: ScaffoldedPolicyNet,
    features: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    tensor = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logits = model.policy_logits(tensor)
        probabilities = torch.softmax(logits / temperature, dim=0)
    if (
        logits.ndim != 1
        or logits.shape[0] != features.shape[0]
        or not torch.isfinite(logits).all()
        or not torch.isfinite(probabilities).all()
    ):
        raise SpecialistReadCalibrationResultError(
            "fixed-state policy output is invalid"
        )
    result = probabilities.detach().cpu().numpy().astype(np.float64)
    if not math.isclose(float(result.sum()), 1.0, abs_tol=1e-6):
        raise SpecialistReadCalibrationResultError(
            "fixed-state policy probabilities do not sum to one"
        )
    return result


def _fixed_state_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise SpecialistReadCalibrationResultError(
            "fixed-state comparison contains no positions"
        )
    phases: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        phases[str(row["phase"])].append(row)

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        critical = [row for row in group if row["critical"]]
        preserving_full = [
            float(row["full_preserving_mass"])
            for row in group
            if row["full_preserving_mass"] is not None
        ]
        preserving_treatment = [
            float(row["theoretical_only_preserving_mass"])
            for row in group
            if row["theoretical_only_preserving_mass"] is not None
        ]
        preserving_scratch = [
            float(row["scratch_preserving_mass"])
            for row in group
            if row["scratch_preserving_mass"] is not None
        ]
        return {
            "states": len(group),
            "critical_states": len(critical),
            "argmax_changes": sum(bool(row["argmax_changed"]) for row in group),
            "mean_policy_total_variation": mean(
                float(row["policy_total_variation"]) for row in group
            ),
            "maximum_policy_total_variation": max(
                float(row["policy_total_variation"]) for row in group
            ),
            "mean_full_entropy": mean(float(row["full_entropy"]) for row in group),
            "mean_theoretical_only_entropy": mean(
                float(row["theoretical_only_entropy"]) for row in group
            ),
            "mean_scratch_entropy": mean(
                float(row["scratch_entropy"]) for row in group
            ),
            "mean_full_preserving_mass": _mean_or_none(preserving_full),
            "mean_theoretical_only_preserving_mass": _mean_or_none(
                preserving_treatment
            ),
            "mean_scratch_preserving_mass": _mean_or_none(preserving_scratch),
        }

    return {
        "all": summarize(rows),
        "by_phase": {
            phase: summarize(group) for phase, group in sorted(phases.items())
        },
    }


def _load_endpoint_corpora(
    root: Path, contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    corpora: list[dict[str, Any]] = []
    for record in contract["analysis"]["fixed_development_diagnostics"]:
        path = _repository_path(root, record["corpus"], field="fixed corpus")
        if _sha256_file(path) != record["corpus_sha256"]:
            raise SpecialistReadCalibrationResultError(
                "fixed development corpus identity differs"
            )
        payload = _strict_json(path)
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise SpecialistReadCalibrationResultError(
                "fixed development corpus has no entries"
            )
        normalized: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise SpecialistReadCalibrationResultError(
                    "fixed development corpus entry is invalid"
                )
            phase = entry.get("phase")
            fen = entry.get("fen")
            index = entry.get("index")
            if phase not in PHASES or not isinstance(fen, str):
                raise SpecialistReadCalibrationResultError(
                    "fixed development corpus entry semantics differ"
                )
            normalized.append({"index": int(index), "phase": phase, "fen": fen})
        corpora.append(
            {
                "path": record["corpus"],
                "sha256": record["corpus_sha256"],
                "role": record["role"],
                "entries": normalized,
            }
        )
    return corpora


def _evaluate_seed_pair(
    *,
    root: Path,
    contract: Mapping[str, Any],
    paths_config: Path,
    seed: int,
    full_checkpoint: Path,
    theoretical_checkpoint: Path,
) -> dict[str, Any]:
    full_envelope = load_checkpoint(full_checkpoint, map_location="cpu")
    theoretical_envelope = load_checkpoint(theoretical_checkpoint, map_location="cpu")
    full_state = full_envelope.payload.trainer_state
    theoretical_state = theoretical_envelope.payload.trainer_state
    model_config = dict(full_state["model_config"])
    if model_config != dict(theoretical_state["model_config"]):
        raise SpecialistReadCalibrationResultError(
            f"seed {seed} model configurations differ"
        )
    expected_games = contract["resources"]["completed_games_per_arm"]
    if (
        int(full_state["game_count"]) != expected_games
        or int(theoretical_state["game_count"]) != expected_games
    ):
        raise SpecialistReadCalibrationResultError(
            f"seed {seed} endpoint game count differs"
        )
    full_assets = dict(full_envelope.descriptor.asset_identities)
    theoretical_assets = dict(theoretical_envelope.descriptor.asset_identities)
    for name in ("human_db", "malom_tablebase"):
        if full_assets.get(name) != theoretical_assets.get(name):
            raise SpecialistReadCalibrationResultError(
                f"seed {seed} endpoint asset differs: {name}"
            )
    full_model = _load_policy(model_config, full_envelope.payload.model_state)
    theoretical_model = _load_policy(
        model_config, theoretical_envelope.payload.model_state
    )
    scratch = _scratch_policy(model_config, seed)
    temperature = trainer._compute_temperature(
        expected_games,
        contract["common_training_contract"]["max_games_schedule"],
        contract["common_training_contract"]["temperature_start"],
    )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise SpecialistReadCalibrationResultError(
            "fixed-state scheduled temperature is invalid"
        )

    config = _strict_json(paths_config)
    human_path = _resolve_config_path(root, str(config["human_db_path"]))
    malom_path = _resolve_config_path(root, str(config["malom_db_path"]))
    human_identity = _sqlite_asset_identity(human_path)
    if human_identity != contract["data_contract"]["human_db_identity"]:
        raise SpecialistReadCalibrationResultError(
            "fixed-state HumanDB identity differs"
        )
    manifest_path = root / DEFAULT_MALOM_MANIFEST
    manifest = load_dataset_manifest(manifest_path)
    if manifest.manifest_sha256 != contract["data_contract"]["malom_manifest_identity"]:
        raise SpecialistReadCalibrationResultError(
            "fixed-state Malom manifest identity differs"
        )
    anchor = next(
        (item for item in manifest.components if item.relative_path == "std.secval"),
        None,
    )
    if anchor is None or _sha256_file(malom_path / "std.secval") != anchor.sha256:
        raise SpecialistReadCalibrationResultError(
            "fixed-state Malom anchor identity differs"
        )

    human = HumanDB(human_path, read_only=True)
    malom = ExternalSolvedDB(str(malom_path), strict=True)
    try:
        if not human.is_available() or not malom.is_available():
            raise SpecialistReadCalibrationResultError(
                "fixed-state read-only data is unavailable"
            )
        advisor = LookaheadAdvisor(
            sentinel=None,
            evaluate_fn=trainer._simple_evaluate,
            value_net=None,
            gap_net=None,
            human_db=human,
            use_sentinel=True,
            ply_depth=12,
            sim_ply_depth=contract["common_training_contract"]["sim_ply_depth"],
            endgame_db=malom,
            strict=True,
        )
        # The same reconstructed scratch target fixes the feature route for
        # both arms. SpecialistDB is deliberately absent below.
        advisor.set_frozen_model(scratch, device=torch.device("cpu"))
        corpus_results: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []
        for corpus in _load_endpoint_corpora(root, contract):
            rows: list[dict[str, Any]] = []
            for entry in corpus["entries"]:
                board = BoardState.from_fen_string(entry["fen"])
                encoded = encode_position_with_lookahead(
                    board,
                    board.turn,
                    sentinel_advisor=None,
                    db=None,
                    value_net=None,
                    lookahead_advisor=advisor,
                    specialist_db=None,
                    sdb_min_samples=3,
                    strict=True,
                )
                if encoded is None or not encoded.legal_moves:
                    raise SpecialistReadCalibrationResultError(
                        f"fixed corpus state is not encodable: {entry['index']}"
                    )
                features = np.asarray(encoded.feat_matrix, dtype=np.float32)
                if features.ndim != 2 or not np.isfinite(features).all():
                    raise SpecialistReadCalibrationResultError(
                        "fixed-state feature matrix is invalid"
                    )
                actions = [
                    encode_move(move, get_game_phase(board, board.turn))
                    for move in encoded.legal_moves
                ]
                qualities: list[float | None] = []
                for move in encoded.legal_moves:
                    value = malom.query_move_quality(board, move)
                    quality = None if value is None else float(value)
                    if quality is not None and quality > 0.0:
                        raise SpecialistReadCalibrationResultError(
                            "Malom returned a positive fixed-state move quality"
                        )
                    qualities.append(quality)
                full_probabilities = _policy_probabilities(
                    full_model, features, temperature=temperature
                )
                theoretical_probabilities = _policy_probabilities(
                    theoretical_model, features, temperature=temperature
                )
                scratch_probabilities = _policy_probabilities(
                    scratch, features, temperature=temperature
                )
                full_argmax = int(np.argmax(full_probabilities))
                theoretical_argmax = int(np.argmax(theoretical_probabilities))
                preserving = np.asarray(
                    [quality == 0.0 for quality in qualities], dtype=bool
                )
                known = np.asarray(
                    [quality is not None for quality in qualities], dtype=bool
                )
                critical = bool(preserving.any() and (known & ~preserving).any())
                row = {
                    "corpus": corpus["path"],
                    "index": entry["index"],
                    "phase": entry["phase"],
                    "fen": entry["fen"],
                    "actions": actions,
                    "critical": critical,
                    "argmax_changed": full_argmax != theoretical_argmax,
                    "full_argmax": actions[full_argmax],
                    "theoretical_only_argmax": actions[theoretical_argmax],
                    "policy_total_variation": float(
                        0.5
                        * np.abs(theoretical_probabilities - full_probabilities).sum()
                    ),
                    "full_entropy": _entropy(full_probabilities),
                    "theoretical_only_entropy": _entropy(theoretical_probabilities),
                    "scratch_entropy": _entropy(scratch_probabilities),
                    "full_preserving_mass": (
                        float(full_probabilities[preserving].sum())
                        if preserving.any()
                        else None
                    ),
                    "theoretical_only_preserving_mass": (
                        float(theoretical_probabilities[preserving].sum())
                        if preserving.any()
                        else None
                    ),
                    "scratch_preserving_mass": (
                        float(scratch_probabilities[preserving].sum())
                        if preserving.any()
                        else None
                    ),
                }
                rows.append(row)
                all_rows.append(row)
            corpus_results.append(
                {
                    "path": corpus["path"],
                    "sha256": corpus["sha256"],
                    "role": corpus["role"],
                    "summary": _fixed_state_summary(rows),
                    "positions": rows,
                }
            )
    finally:
        human.close()
        malom.close()

    return {
        "seed": seed,
        "route": {
            "device": "cpu",
            "temperature": temperature,
            "specialist_db_projection": "disabled",
            "frozen_target": "same-seed reconstructed scratch policy",
            "scratch_initialization_shared_exactly": True,
            "sentinel": False,
            "value_net": False,
            "gap_net": False,
        },
        "checkpoint_sha256": {
            "full": _sha256_file(full_checkpoint),
            "theoretical-only": _sha256_file(theoretical_checkpoint),
        },
        "corpora": corpus_results,
        "aggregate": _fixed_state_summary(all_rows),
    }


def decide_specialist_read_calibration_result(
    arm_summaries: Sequence[Mapping[str, Any]],
    endpoint_pairs: Sequence[Mapping[str, Any]],
    *,
    decision_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen three-seed read-mechanism decision."""
    indexed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for arm in arm_summaries:
        key = (
            _require_int(arm.get("seed"), field="arm.seed"),
            str(arm.get("condition")),
        )
        if key in indexed:
            raise SpecialistReadCalibrationResultError(
                "SpecialistDB calibration arm is duplicated"
            )
        indexed[key] = arm
    expected = {
        (seed, condition)
        for seed in (61, 62, 63)
        for condition in ("full", "theoretical-only")
    }
    if set(indexed) != expected:
        raise SpecialistReadCalibrationResultError(
            "SpecialistDB calibration pairing differs"
        )
    pair_by_seed = {
        _require_int(pair.get("seed"), field="endpoint.seed"): pair
        for pair in endpoint_pairs
    }
    if set(pair_by_seed) != {61, 62, 63}:
        raise SpecialistReadCalibrationResultError(
            "endpoint comparison seed cohort differs"
        )
    minimum_argmax = _require_int(
        decision_rule["minimum_argmax_changes_for_detectable_pair"],
        field="minimum_argmax_changes_for_detectable_pair",
        minimum=1,
    )
    minimum_tv = _require_finite(
        decision_rule["minimum_mean_total_variation_for_detectable_pair"],
        field="minimum_mean_total_variation_for_detectable_pair",
    )
    minimum_pairs = _require_int(
        decision_rule["minimum_reproducible_seed_pairs"],
        field="minimum_reproducible_seed_pairs",
        minimum=1,
    )
    if decision_rule.get("training_wdl_is_not_a_selection_metric") is not True:
        raise SpecialistReadCalibrationResultError(
            "training W/D/L decision boundary differs"
        )

    pairs: list[dict[str, Any]] = []
    for seed in (61, 62, 63):
        full = indexed[(seed, "full")]
        theoretical = indexed[(seed, "theoretical-only")]
        endpoint = pair_by_seed[seed]
        aggregate = endpoint["aggregate"]["all"]
        full_engaged = bool(full["metrics"]["specialist_read_intervention"]["engaged"])
        theoretical_engaged = bool(
            theoretical["metrics"]["specialist_read_intervention"]["engaged"]
        )
        argmax_changes = int(aggregate["argmax_changes"])
        mean_tv = float(aggregate["mean_policy_total_variation"])
        detectable = argmax_changes >= minimum_argmax or mean_tv >= minimum_tv
        checks = {
            "full_intervention_engaged": full_engaged,
            "theoretical_only_intervention_engaged": theoretical_engaged,
            "full_policy_health_passed": full["policy_health"]["passed"] is True,
            "theoretical_only_policy_health_passed": (
                theoretical["policy_health"]["passed"] is True
            ),
            "scratch_initialization_shared_exactly": endpoint["route"][
                "scratch_initialization_shared_exactly"
            ]
            is True,
            "specialist_projection_disabled_for_endpoint": endpoint["route"][
                "specialist_db_projection"
            ]
            == "disabled",
        }
        pairs.append(
            {
                "seed": seed,
                "full_arm_id": full["arm_id"],
                "theoretical_only_arm_id": theoretical["arm_id"],
                "argmax_changes": argmax_changes,
                "mean_policy_total_variation": mean_tv,
                "detectable_learned_policy_effect": detectable,
                "checks": checks,
                "eligible_pair": all(checks.values()),
            }
        )
    all_safe = all(pair["eligible_pair"] for pair in pairs)
    detectable_pairs = sum(pair["detectable_learned_policy_effect"] for pair in pairs)
    eligible = all_safe and detectable_pairs >= minimum_pairs
    return {
        "verdict": (
            "reproducible_read_effect_eligible_for_heldout_design"
            if eligible
            else "inconclusive_no_retained_mode_selection"
        ),
        "eligible": eligible,
        "detectable_seed_pairs": detectable_pairs,
        "all_identity_and_safety_gates_passed": all_safe,
        "thresholds": dict(decision_rule),
        "pairs": pairs,
        "training_wdl_used_for_selection": False,
        "selected_read_mode": None,
        "claim_boundary": (
            "read-mechanism calibration only; eligibility permits designing, "
            "but not launching, a held-out effectiveness comparison"
        ),
    }


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SpecialistReadCalibrationResultError("Git evidence audit failed")
    return result.stdout.strip()


def _inspect_analysis_source(root: Path, expected_commit: str) -> dict[str, Any]:
    branch = _git_output(root, "branch", "--show-current")
    head = _git_output(root, "rev-parse", "HEAD")
    upstream = _git_output(root, "rev-parse", "origin/dev")
    dirty = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != "dev" or head != upstream or dirty:
        raise SpecialistReadCalibrationResultError(
            "result analysis requires a clean published dev"
        )
    if head != expected_commit:
        raise SpecialistReadCalibrationResultError(
            "result analysis must use the exact training source commit"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": upstream,
        "training_source_commit": expected_commit,
        "worktree_clean": True,
    }


def _validate_readiness(
    path: Path,
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    readiness = _strict_json(path)
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise SpecialistReadCalibrationResultError("readiness schema differs")
    identity = readiness.get("readiness_identity")
    body = dict(readiness)
    body.pop("readiness_identity", None)
    if identity != canonical_sha256(body):
        raise SpecialistReadCalibrationResultError("readiness identity is invalid")
    if (
        readiness.get("state") != "ready_for_product_authorization"
        or readiness.get("launch_authorized") is not False
    ):
        raise SpecialistReadCalibrationResultError("readiness state differs")
    record = readiness.get("contract")
    if not isinstance(record, Mapping) or (
        record.get("plan_identity") != contract["plan_identity"]
        or record.get("file_sha256") != _sha256_file(contract_path)
    ):
        raise SpecialistReadCalibrationResultError("readiness binds another contract")
    if (
        readiness.get("result_analysis")
        != contract["analysis"]["result_implementation"]
    ):
        raise SpecialistReadCalibrationResultError(
            "readiness result-analyzer identity differs"
        )
    if len(readiness.get("arms", ())) != 6:
        raise SpecialistReadCalibrationResultError("readiness does not bind six arms")
    return readiness


def _analyze_arm(
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    readiness: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
) -> dict[str, Any]:
    control_dir = _repository_path(root, arm["control_dir"], field="control_dir")
    plan_path = control_dir / "plan.json"
    authorization_path = control_dir / "authorization.json"
    plan = load_managed_plan(plan_path)
    args = _assert_plan_semantics(
        plan,
        root=root,
        contract=contract,
        arm=arm,
        paths_config=paths_config,
        source_commit=source_commit,
    )
    ready_arm = _readiness_arm(readiness, str(arm["arm_id"]))
    if (
        ready_arm.get("plan_sha256") != plan.plan_sha256
        or ready_arm.get("specialist_read_mode") != arm["specialist_read_mode"]
    ):
        raise SpecialistReadCalibrationResultError("readiness arm identity differs")
    preflight = ready_arm.get("preflight")
    if not isinstance(preflight, Mapping):
        raise SpecialistReadCalibrationResultError("arm preflight evidence is absent")
    preflight_path = Path(str(preflight.get("path", "")))
    if not preflight_path.is_file() or _sha256_file(preflight_path) != preflight.get(
        "sha256"
    ):
        raise SpecialistReadCalibrationResultError("arm preflight evidence changed")
    preflight_report = _strict_json(preflight_path)
    authorization = load_managed_authorization(authorization_path)
    _validate_authorization(authorization, plan)
    completed_details, checkpoint = _validate_controller_completion(plan)
    health = _validate_policy_health(
        plan,
        details=completed_details,
        checkpoint=checkpoint,
    )

    segment = control_dir / "segments" / "segment-0001"
    manifest_path = segment / "run-manifest.json"
    train_log_path = segment / "train_log.jsonl"
    update_log_path = segment / "update_log.jsonl"
    run_events_path = segment / managed.RUN_EVENT_LEDGER_NAME
    manifest = _strict_json(manifest_path)
    _validate_manifest(
        manifest,
        plan=plan,
        arm={
            **arm,
            "mill_bonus_mode": contract["common_training_contract"]["mill_bonus_mode"],
        },
        contract=contract,
        preflight=preflight_report,
    )
    config = manifest.get("resolved_config")
    if (
        not isinstance(config, Mapping)
        or config.get("specialist_read_mode") != arm["specialist_read_mode"]
    ):
        raise SpecialistReadCalibrationResultError(
            "run manifest SpecialistDB read mode differs"
        )
    run_events = managed.load_run_events(run_events_path)
    if not run_events or run_events[-1].event_type != "training_completed":
        raise SpecialistReadCalibrationResultError("trainer lifecycle is incomplete")

    schedule = contract["resources"]["schedule_counts_by_seed"][str(arm["seed"])]
    metrics = summarize_specialist_read_game_rows(
        _strict_jsonl(train_log_path),
        mode=str(arm["specialist_read_mode"]),
        expected_games=contract["resources"]["completed_games_per_arm"],
        expected_schedule_counts=schedule,
    )
    updates = summarize_update_rows(
        _strict_jsonl(update_log_path),
        coefficient=0.0,
        expected_games=contract["resources"]["completed_games_per_arm"],
    )
    specialist_db = Path(args.specialist_db).resolve(strict=True)
    full_health_path = Path(str(health["report"]))
    artifacts = {
        "plan": _artifact_record(root, plan_path),
        "authorization": _artifact_record(root, authorization_path),
        "controller_events": _artifact_record(
            root, control_dir / managed.CONTROLLER_LEDGER_NAME
        ),
        "preflight": _artifact_record(root, preflight_path),
        "run_manifest": _artifact_record(root, manifest_path),
        "run_events": _artifact_record(root, run_events_path),
        "train_log": _artifact_record(root, train_log_path),
        "update_log": _artifact_record(root, update_log_path),
        "checkpoint": _artifact_record(root, checkpoint),
        "specialist_db": _artifact_record(root, specialist_db),
        "policy_health": _artifact_record(root, full_health_path),
    }
    return {
        "arm_id": arm["arm_id"],
        "condition": arm["condition"],
        "seed": arm["seed"],
        "specialist_read_mode": arm["specialist_read_mode"],
        "plan_sha256": plan.plan_sha256,
        "authorization_file_sha256": _sha256_file(authorization_path),
        "experiment_id": plan.experiment_id,
        "source_commit": plan.git_commit,
        "schedule_max_games": plan.max_games,
        "completion_game_bound": plan.game_bound,
        "policy_health": health,
        "metrics": metrics,
        "optimizer_updates": updates,
        "runtime_identities": {
            "mif": manifest["checkpoint_policy"]["mifSuite"],
            "ruleset": manifest["checkpoint_policy"]["ruleset"],
            "assets": manifest["assets"],
            "experiment_digest": manifest["checkpoint_policy"]["experimentDigest"],
            "resume_config_sha256": plan.resume_config_sha256,
        },
        "artifacts": artifacts,
    }


def analyze_specialist_read_calibration_result(
    *,
    root: Path,
    contract_path: Path,
    readiness_path: Path,
    paths_config: Path,
) -> dict[str, Any]:
    """Validate six completed arms and produce one deterministic result."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    readiness_path = readiness_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    contract = load_specialist_read_calibration_contract(contract_path)
    readiness = _validate_readiness(
        readiness_path,
        contract=contract,
        contract_path=contract_path,
    )
    source_commit = str(readiness["source"]["head"])
    source = _inspect_analysis_source(root, source_commit)
    arms = [
        _analyze_arm(
            root=root,
            contract=contract,
            arm=arm,
            readiness=readiness,
            paths_config=paths_config,
            source_commit=source_commit,
        )
        for arm in _ordered_arms(contract)
    ]
    arm_by_key = {(int(arm["seed"]), str(arm["condition"])): arm for arm in arms}
    endpoint_pairs = []
    for seed in contract["pairing"]["seeds"]:
        endpoint_pairs.append(
            _evaluate_seed_pair(
                root=root,
                contract=contract,
                paths_config=paths_config,
                seed=int(seed),
                full_checkpoint=root
                / arm_by_key[(int(seed), "full")]["artifacts"]["checkpoint"]["path"],
                theoretical_checkpoint=root
                / arm_by_key[(int(seed), "theoretical-only")]["artifacts"][
                    "checkpoint"
                ]["path"],
            )
        )
    decision = decide_specialist_read_calibration_result(
        arms,
        endpoint_pairs,
        decision_rule=contract["analysis"]["decision_rule"],
    )
    body = {
        "schema_version": RESULT_SCHEMA,
        "claim_boundary": contract["claim_boundary"],
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "plan_identity": contract["plan_identity"],
            "file_sha256": _sha256_file(contract_path),
        },
        "readiness": {
            "path": readiness_path.relative_to(root).as_posix(),
            "readiness_identity": readiness["readiness_identity"],
            "file_sha256": _sha256_file(readiness_path),
        },
        "analysis_source": source,
        "data_and_runtime_versions": {
            "data_contract": contract["data_contract"],
            "rules_and_runtime": contract["rules_and_runtime"],
        },
        "hyperparameters": contract["common_training_contract"],
        "baseline": {
            "type": "matched full SpecialistDB read arm for each fresh seed",
            "seeds": contract["pairing"]["seeds"],
            "pairing": contract["pairing"],
        },
        "arms": arms,
        "endpoint_pairs": endpoint_pairs,
        "decision": decision,
        "interpretation": {
            "observation_facts": [
                "Training, validation and endpoint fields in this result are "
                "observations from the exact bound artifacts, not forecasts.",
                "Ordinary RL has no supervised validation-loss curve; the "
                "result records that absence rather than inventing one.",
                "SpecialistDB read counters prove whether the intended "
                "intervention was encountered during each arm.",
            ],
            "hypothesis": contract["hypothesis"],
            "supporting_evidence": decision["pairs"],
            "counter_evidence_and_limits": [
                "Two hundred and fifty games per arm test a mechanism, not "
                "long-run learning or playing strength.",
                "Both endpoint corpora are inspected development evidence, "
                "not held-out validation.",
                "Training W/D/L is stratified diagnostic evidence and is not "
                "a selection metric.",
                "A reproducible effect does not establish which read mode is "
                "better and authorizes no training, promotion or publication.",
            ],
            "next_verification_experiment": (
                "If and only if the effect is reproducible, freeze a separate "
                "held-out effectiveness comparison before selecting a mode."
            ),
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}


def publish_result(path: Path, report: Mapping[str, Any]) -> None:
    """Publish one immutable canonical result after all validation passes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise SpecialistReadCalibrationResultError(
            f"result output already exists: {path}"
        ) from exc


__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_PATHS_CONFIG",
    "DEFAULT_READINESS_REPORT",
    "DEFAULT_RESULT",
    "RESULT_SCHEMA",
    "SpecialistReadCalibrationResultError",
    "analyze_specialist_read_calibration_result",
    "decide_specialist_read_calibration_result",
    "publish_result",
    "summarize_specialist_read_game_rows",
]

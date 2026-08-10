"""Compare target-refresh policies on one fixed common-anchor feature corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.human_db import HumanDB  # noqa: E402
from game.board import BoardState  # noqa: E402
from learned_ai.data.data_contract import load_dataset_manifest  # noqa: E402
from learned_ai.evaluation.common_anchor_policy_distribution import (  # noqa: E402
    DEFAULT_DIVERGENCE_THRESHOLDS,
    PRIMARY_TEMPERATURE,
    classify_final_policy_divergence,
    compare_action_logits,
    summarize_state_comparisons,
)
from learned_ai.evaluation.phase_corpus import (  # noqa: E402
    validate_phase_corpus,
)
from learned_ai.models.lookahead_advisor import LookaheadAdvisor  # noqa: E402
from learned_ai.models.scaffolded_encoder import (  # noqa: E402
    encode_position_with_lookahead,
)
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet  # noqa: E402
from learned_ai.sentinel.db_teacher import ExternalSolvedDB  # noqa: E402
from learned_ai.training.checkpoint_envelope import (  # noqa: E402
    CheckpointEnvelope,
    load_checkpoint,
)
from learned_ai.training.generalist_preflight import (  # noqa: E402
    _probe_human_db,
)
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from scripts import train_s_gen_v2 as trainer  # noqa: E402


DEFAULT_ATTEMPT_CONTRACT = ROOT / (
    "docs/experiments/"
    "sanmill-target-refresh-common-anchor-diagnostic-v1-attempt-003.json"
)
DEFAULT_CORPUS = ROOT / "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
DEFAULT_PATHS_CONFIG = ROOT / "data/training_paths.local.json"
DEFAULT_MALOM_MANIFEST = (
    ROOT / "data/manifests/malom-sector-corrected-v1.json"
)
DEFAULT_OUTPUT = ROOT / (
    "out/target-refresh-common-anchor-policy-analysis-v1/result.json"
)
EXPECTED_ATTEMPT_PLAN_IDENTITY = (
    "8cc192f5152bb15957f5bc7860bce12d6db0200bc51c2d6766752ed4fc54c634"
)
EXPECTED_CORPUS_SHA256 = (
    "cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e"
)
EXPECTED_DELTAS = (4, 8, 12, 16)
EXPECTED_SEEDS = (64, 65)
EXPECTED_CONDITIONS = ("refresh", "no-refresh")


class CommonAnchorAnalysisError(RuntimeError):
    """Raised when immutable policy-analysis inputs cannot be established."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise CommonAnchorAnalysisError(
            "evidence path is outside the repository"
        ) from exc


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CommonAnchorAnalysisError(f"cannot read JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise CommonAnchorAnalysisError(f"JSON root is not an object: {path.name}")
    return value


def _git_identity() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise CommonAnchorAnalysisError(
            "tracked worktree must be clean before immutable analysis"
        )
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    origin_dev = subprocess.check_output(
        ["git", "rev-parse", "origin/dev"], cwd=ROOT, text=True
    ).strip()
    if branch != "dev":
        raise CommonAnchorAnalysisError(
            "analysis requires the dev branch"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin_dev,
        "published": head == origin_dev,
    }


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise CommonAnchorAnalysisError(
                f"model state contains a non-tensor: {name}"
            )
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_policy(
    envelope: CheckpointEnvelope,
    *,
    device: torch.device,
) -> ScaffoldedPolicyNet:
    config = dict(envelope.payload.trainer_state["model_config"])
    model = ScaffoldedPolicyNet.from_config(config).to(device)
    model.load_state_dict(envelope.payload.model_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _normalized_action(move: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "from": None if move.get("from") is None else str(move["from"]),
        "to": None if move.get("to") is None else str(move["to"]),
        "capture": (
            None if move.get("capture") is None else str(move["capture"])
        ),
    }


def _action_key(move: Mapping[str, Any]) -> str:
    return canonical_json_bytes(_normalized_action(move)).decode("utf-8")


def _feature_sha256(features: np.ndarray, action_keys: Sequence[str]) -> str:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(action_keys):
        raise CommonAnchorAnalysisError(
            "feature matrix does not align with legal actions"
        )
    if not np.isfinite(matrix).all():
        raise CommonAnchorAnalysisError("feature matrix is non-finite")
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(matrix.tobytes(order="C"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(list(action_keys)))
    return digest.hexdigest()


def _file_observation(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"exists": path.exists()}
    if path.exists():
        stat = path.stat()
        result.update({"size": stat.st_size, "modified_ns": stat.st_mtime_ns})
    return result


def _read_only_observations(
    *,
    human_db_path: Path,
    malom_path: Path,
) -> dict[str, Any]:
    return {
        "human_db_main": _file_observation(human_db_path),
        "human_db_wal": _file_observation(
            human_db_path.with_name(human_db_path.name + "-wal")
        ),
        "human_db_shm": _file_observation(
            human_db_path.with_name(human_db_path.name + "-shm")
        ),
        "malom_std_secval": _file_observation(malom_path / "std.secval"),
    }


def _open_immutable_human_db(path: Path) -> HumanDB:
    """Open the frozen main file without participating in WAL/SHM locking."""
    human_db = HumanDB(path, read_only=True, immutable=True)
    if not human_db.is_available():
        human_db.close()
        raise CommonAnchorAnalysisError("immutable HumanDB open failed")
    return human_db


def _validate_attempt_contract(path: Path) -> dict[str, Any]:
    contract = _strict_json(path)
    if contract.get("plan_identity") != EXPECTED_ATTEMPT_PLAN_IDENTITY:
        raise CommonAnchorAnalysisError("attempt-003 plan identity differs")
    arms = contract.get("arms")
    if not isinstance(arms, list) or len(arms) != 4:
        raise CommonAnchorAnalysisError("attempt-003 arms differ")
    observed = {
        (int(arm["seed"]), str(arm["condition"])) for arm in arms
    }
    expected = {
        (seed, condition)
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    }
    if observed != expected:
        raise CommonAnchorAnalysisError("attempt-003 seed/condition cells differ")
    measurement = contract.get("measurement_contract", {})
    if (
        measurement.get("measurement_temperature") != PRIMARY_TEMPERATURE
        or tuple(
            range(
                int(measurement["measurement_every_updates"]),
                int(measurement["post_anchor_optimizer_updates"]) + 1,
                int(measurement["measurement_every_updates"]),
            )
        )
        != EXPECTED_DELTAS
        or measurement.get("specialist_read_mode") != "disabled"
        or measurement.get("writes_training_data") is not False
    ):
        raise CommonAnchorAnalysisError("attempt-003 measurement contract differs")
    return contract


def _arm_by_cell(
    contract: Mapping[str, Any],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {
        (int(arm["seed"]), str(arm["condition"])): arm
        for arm in contract["arms"]
    }


def _segment(arm: Mapping[str, Any]) -> Path:
    path = ROOT / str(arm["control_dir"]) / "segments" / "segment-0001"
    if not path.is_dir():
        raise CommonAnchorAnalysisError(
            f"completed arm segment is absent: {arm['arm_id']}"
        )
    return path


def _load_anchor_pair(
    arms: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[CheckpointEnvelope, dict[str, Any]]:
    records: dict[str, Any] = {}
    envelopes: dict[str, CheckpointEnvelope] = {}
    for condition in EXPECTED_CONDITIONS:
        arm = arms[(seed, condition)]
        path = _segment(arm) / "development-measurement-anchor.pt"
        envelope = load_checkpoint(path, map_location="cpu")
        state = envelope.payload.trainer_state
        if (
            envelope.descriptor.role != "development_measurement_anchor"
            or envelope.descriptor.experiment_id != arm["experiment_id"]
            or state.get("game_count") != 50
            or state.get("update_count") != arm["anchor_expected_update_count"]
        ):
            raise CommonAnchorAnalysisError(
                f"game-50 anchor semantics differ: {arm['arm_id']}"
            )
        state_sha256 = _state_dict_sha256(envelope.payload.model_state)
        records[condition] = {
            "path": _relative(path),
            "file_sha256": _sha256_file(path),
            "checkpoint_id": envelope.descriptor.checkpoint_id,
            "model_state_sha256": state_sha256,
            "game_count": int(state["game_count"]),
            "update_count": int(state["update_count"]),
        }
        envelopes[condition] = envelope

    if (
        records["refresh"]["model_state_sha256"]
        != records["no-refresh"]["model_state_sha256"]
    ):
        raise CommonAnchorAnalysisError(
            f"seed {seed} game-50 anchor model states are not identical"
        )
    if dict(envelopes["refresh"].payload.trainer_state["model_config"]) != dict(
        envelopes["no-refresh"].payload.trainer_state["model_config"]
    ):
        raise CommonAnchorAnalysisError(
            f"seed {seed} game-50 model configurations differ"
        )
    return envelopes["refresh"], {
        "conditions": records,
        "model_state_identical": True,
        "canonical_condition": "refresh",
    }


def _load_candidate_pair(
    arms: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    seed: int,
    delta: int,
    device: torch.device,
) -> tuple[dict[str, ScaffoldedPolicyNet], dict[str, Any]]:
    models: dict[str, ScaffoldedPolicyNet] = {}
    records: dict[str, Any] = {}
    model_config: dict[str, Any] | None = None
    for condition in EXPECTED_CONDITIONS:
        arm = arms[(seed, condition)]
        update_count = int(arm["anchor_expected_update_count"]) + delta
        path = _segment(arm) / (
            f"development-measurement-update-{update_count:08d}.pt"
        )
        envelope = load_checkpoint(path, map_location=device)
        state = envelope.payload.trainer_state
        if (
            envelope.descriptor.role != "development_measurement_candidate"
            or envelope.descriptor.experiment_id != arm["experiment_id"]
            or state.get("update_count") != update_count
        ):
            raise CommonAnchorAnalysisError(
                f"candidate checkpoint semantics differ: {arm['arm_id']}:{delta}"
            )
        current_config = dict(state["model_config"])
        if model_config is None:
            model_config = current_config
        elif current_config != model_config:
            raise CommonAnchorAnalysisError(
                f"candidate model configurations differ for seed {seed}"
            )
        assets = dict(envelope.descriptor.asset_identities)
        models[condition] = _load_policy(envelope, device=device)
        records[condition] = {
            "path": _relative(path),
            "file_sha256": _sha256_file(path),
            "checkpoint_id": envelope.descriptor.checkpoint_id,
            "model_state_sha256": _state_dict_sha256(
                envelope.payload.model_state
            ),
            "game_count": int(state["game_count"]),
            "update_count": int(state["update_count"]),
            "asset_identities": {
                key: assets[key]
                for key in (
                    "human_db",
                    "malom_tablebase",
                    "mif_suite_1_0",
                    "sanmill_training_runtime",
                    "training_ruleset",
                )
            },
        }
    if records["refresh"]["asset_identities"] != records["no-refresh"][
        "asset_identities"
    ]:
        raise CommonAnchorAnalysisError(
            f"paired checkpoint read-only asset identities differ for seed {seed}"
        )
    return models, records


def _build_feature_corpus(
    *,
    corpus: Mapping[str, Any],
    anchor_model: ScaffoldedPolicyNet,
    human_db: HumanDB,
    malom: ExternalSolvedDB,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    advisor = LookaheadAdvisor(
        sentinel=None,
        evaluate_fn=trainer._simple_evaluate,
        value_net=None,
        gap_net=None,
        human_db=human_db,
        use_sentinel=True,
        endgame_db=malom,
        ply_depth=12,
        frozen_model=anchor_model,
        frozen_device=device,
        sim_ply_depth=5,
        strict=True,
    )
    states: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    action_count = 0
    feature_identities: list[dict[str, Any]] = []
    for entry in corpus["entries"]:
        board = BoardState.from_fen_string(str(entry["fen"]))
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
            raise CommonAnchorAnalysisError(
                f"fixed corpus entry is not encodable: {entry['index']}"
            )
        features = np.asarray(encoded.feat_matrix, dtype=np.float32)
        actions = [_normalized_action(move) for move in encoded.legal_moves]
        action_keys = [_action_key(move) for move in encoded.legal_moves]
        if len(set(action_keys)) != len(action_keys):
            raise CommonAnchorAnalysisError(
                f"duplicate legal action identity: {entry['index']}"
            )
        qualities: list[float] = []
        for move in encoded.legal_moves:
            quality = malom.query_move_quality(board, move)
            qualities.append(float("nan") if quality is None else float(quality))
        feature_sha256 = _feature_sha256(features, action_keys)
        phase = str(entry["phase"])
        phase_counts[phase] += 1
        action_count += len(actions)
        feature_identities.append(
            {"index": int(entry["index"]), "sha256": feature_sha256}
        )
        states.append(
            {
                "index": int(entry["index"]),
                "fen": str(entry["fen"]),
                "phase": phase,
                "turn": str(entry["turn"]),
                "features": features,
                "feature_sha256": feature_sha256,
                "actions": actions,
                "action_keys": action_keys,
                "malom_qualities": np.asarray(qualities, dtype=np.float64),
                "heuristic_top1_action_key": action_keys[
                    int(encoded.h_top1_idx)
                ],
            }
        )
    if phase_counts != {"placement": 22, "movement": 21, "flying": 21}:
        raise CommonAnchorAnalysisError("fixed corpus phase counts differ")
    return states, {
        "states": len(states),
        "legal_actions": action_count,
        "phase_counts": dict(sorted(phase_counts.items())),
        "feature_corpus_identity": canonical_sha256(feature_identities),
        "feature_schema": "s-gen-v2-move-134",
        "feature_route": {
            "common_game_50_anchor": True,
            "lookahead_ply_depth": 12,
            "sim_ply_depth": 5,
            "human_db_frequencies_and_outcomes": True,
            "human_db_malom_labels": False,
            "malom_tablebase": True,
            "specialist_db": False,
            "sentinel": False,
            "value_net": False,
            "gap_net": False,
            "strict_dependency_failures": True,
        },
    }


def _compare_checkpoint_pair(
    *,
    states: Sequence[Mapping[str, Any]],
    models: Mapping[str, ScaffoldedPolicyNet],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for state in states:
        feature_tensor = torch.as_tensor(
            state["features"], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            refresh_logits = models["refresh"].policy_logits(feature_tensor)
            no_refresh_logits = models["no-refresh"].policy_logits(feature_tensor)
        if refresh_logits.shape != (len(state["actions"]),) or (
            no_refresh_logits.shape != refresh_logits.shape
        ):
            raise CommonAnchorAnalysisError(
                f"policy logit shape differs at corpus entry {state['index']}"
            )
        comparison = compare_action_logits(
            refresh_logits=refresh_logits.detach().cpu().numpy(),
            no_refresh_logits=no_refresh_logits.detach().cpu().numpy(),
            action_keys=state["action_keys"],
            malom_qualities=state["malom_qualities"],
        )
        for action_record, action in zip(
            comparison["actions"], state["actions"], strict=True
        ):
            action_record["action"] = action
        records.append(
            {
                "index": state["index"],
                "fen": state["fen"],
                "phase": state["phase"],
                "turn": state["turn"],
                "feature_sha256": state["feature_sha256"],
                "heuristic_top1_action_key": state[
                    "heuristic_top1_action_key"
                ],
                "comparison": comparison,
            }
        )
    return records, summarize_state_comparisons(records)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt-contract", type=Path, default=DEFAULT_ATTEMPT_CONTRACT
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG
    )
    parser.add_argument(
        "--malom-manifest", type=Path, default=DEFAULT_MALOM_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = {
        "attempt_contract": args.attempt_contract.resolve(),
        "corpus": args.corpus.resolve(),
        "paths_config": args.paths_config.resolve(),
        "malom_manifest": args.malom_manifest.resolve(),
        "output": args.output.resolve(),
    }
    for label in ("attempt_contract", "corpus", "paths_config", "malom_manifest"):
        if not paths[label].is_file():
            raise CommonAnchorAnalysisError(f"{label} is not an existing file")
    if paths["output"].exists():
        raise CommonAnchorAnalysisError("analysis output already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise CommonAnchorAnalysisError("requested CUDA device is unavailable")
    device = torch.device(args.device)

    source = _git_identity()
    contract = _validate_attempt_contract(paths["attempt_contract"])
    corpus_sha256 = _sha256_file(paths["corpus"])
    if corpus_sha256 != EXPECTED_CORPUS_SHA256:
        raise CommonAnchorAnalysisError("fixed phase corpus SHA-256 differs")
    corpus = _strict_json(paths["corpus"])
    validate_phase_corpus(corpus)

    settings = _strict_json(paths["paths_config"])
    human_db_path = (ROOT / settings["human_db_path"]).resolve()
    malom_path_value = Path(str(settings["malom_db_path"]))
    malom_path = (
        malom_path_value.resolve()
        if malom_path_value.is_absolute()
        else (ROOT / malom_path_value).resolve()
    )
    if not human_db_path.is_file() or not malom_path.is_dir():
        raise CommonAnchorAnalysisError("read-only data source is unavailable")

    human_report = _probe_human_db(human_db_path)
    data_contract = contract["data_contract"]
    if (
        human_report.get("error")
        or human_report.get("identity") != data_contract["human_db_identity"]
        or human_report.get("malom_columns_policy")
        != "masked_historical_labels"
    ):
        raise CommonAnchorAnalysisError("HumanDB identity or label policy differs")
    malom_manifest = load_dataset_manifest(paths["malom_manifest"])
    if (
        malom_manifest.manifest_sha256
        != data_contract["malom_manifest_identity"]
    ):
        raise CommonAnchorAnalysisError("Malom manifest identity differs")
    std_anchor = next(
        (
            component
            for component in malom_manifest.components
            if component.relative_path == "std.secval"
        ),
        None,
    )
    if std_anchor is None or _sha256_file(malom_path / "std.secval") != (
        std_anchor.sha256
    ):
        raise CommonAnchorAnalysisError("Malom std.secval identity differs")

    before = _read_only_observations(
        human_db_path=human_db_path,
        malom_path=malom_path,
    )
    human_db = _open_immutable_human_db(human_db_path)
    malom = ExternalSolvedDB(str(malom_path), strict=True)
    if not malom.is_available():
        raise CommonAnchorAnalysisError("read-only policy dependency is unavailable")
    arms = _arm_by_cell(contract)
    seed_reports: dict[str, Any] = {}
    final_by_seed: dict[str, Any] = {}
    try:
        for seed in EXPECTED_SEEDS:
            anchor_envelope, anchor_record = _load_anchor_pair(
                arms, seed=seed
            )
            anchor_model = _load_policy(anchor_envelope, device=device)
            states, feature_record = _build_feature_corpus(
                corpus=corpus,
                anchor_model=anchor_model,
                human_db=human_db,
                malom=malom,
                device=device,
            )
            checkpoint_reports: list[dict[str, Any]] = []
            for delta in EXPECTED_DELTAS:
                models, checkpoint_record = _load_candidate_pair(
                    arms,
                    seed=seed,
                    delta=delta,
                    device=device,
                )
                state_records, summary = _compare_checkpoint_pair(
                    states=states,
                    models=models,
                    device=device,
                )
                checkpoint_reports.append(
                    {
                        "post_anchor_optimizer_updates": delta,
                        "checkpoints": checkpoint_record,
                        "summary": summary,
                        "states": state_records,
                    }
                )
                if delta == EXPECTED_DELTAS[-1]:
                    final_by_seed[str(seed)] = summary
            seed_reports[str(seed)] = {
                "anchor": anchor_record,
                "feature_corpus": feature_record,
                "checkpoints": checkpoint_reports,
            }
    finally:
        human_db.close()
        malom.close()

    after = _read_only_observations(
        human_db_path=human_db_path,
        malom_path=malom_path,
    )
    if before != after:
        raise CommonAnchorAnalysisError(
            "read-only data source observations changed during analysis"
        )

    decision = classify_final_policy_divergence(
        final_by_seed,
        thresholds=DEFAULT_DIVERGENCE_THRESHOLDS,
    )
    report_core = {
        "schema_version": "nmm.common-anchor-policy-distribution-analysis.v1",
        "scope": {
            "read_only": True,
            "training_games": 0,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "database_writes": 0,
            "candidate_model_loaded": True,
            "held_out_strength_claim": False,
            "promotion_or_long_run_authority": False,
        },
        "identities": {
            "source": source,
            "attempt_contract": {
                "path": _relative(paths["attempt_contract"]),
                "sha256": _sha256_file(paths["attempt_contract"]),
                "plan_identity": contract["plan_identity"],
            },
            "fixed_phase_corpus": {
                "path": _relative(paths["corpus"]),
                "sha256": corpus_sha256,
                "corpus_identity": corpus["corpus_identity"],
            },
            "paths_config_sha256": _sha256_file(paths["paths_config"]),
            "human_db": {
                "lookup_key": "human_db_path",
                "identity": human_report["identity"],
                "use": "frequencies and empirical outcomes only",
                "historical_malom_labels": "masked",
            },
            "malom": {
                "lookup_key": "malom_db_path",
                "manifest_path": _relative(paths["malom_manifest"]),
                "manifest_file_sha256": _sha256_file(paths["malom_manifest"]),
                "manifest_identity": malom_manifest.manifest_sha256,
                "std_secval_sha256": std_anchor.sha256,
            },
        },
        "comparison_contract": {
            "seeds": list(EXPECTED_SEEDS),
            "conditions": list(EXPECTED_CONDITIONS),
            "post_anchor_optimizer_update_deltas": list(EXPECTED_DELTAS),
            "temperatures": [1.0, PRIMARY_TEMPERATURE],
            "primary_temperature": PRIMARY_TEMPERATURE,
            "state_weighting": "each of the 64 fixed positions has equal weight",
            "kl_direction": "both directions reported in natural-log nats",
            "jensen_shannon_units": "natural-log nats",
            "malom_probability": (
                "absolute probability mass assigned to WDL-preserving actions"
            ),
            "rank_ties": "absolute logit difference <= 1e-7",
            "classification_thresholds_frozen_before_execution": (
                DEFAULT_DIVERGENCE_THRESHOLDS
            ),
        },
        "read_only_observations": {"before": before, "after": after},
        "by_seed": seed_reports,
        "decision": decision,
        "interpretation_boundaries": [
            "full legal-action logits are compared on common fixed features",
            "top-1 changes alone do not establish material policy divergence",
            "the draft phase corpus is development evidence, not held-out strength evidence",
            "no supervised train or validation curves exist for this online RL diagnostic",
            "this analysis cannot select, promote, publish or start a model",
        ],
    }
    report = dict(report_core)
    report["analysis_identity"] = canonical_sha256(report_core)
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["output"].with_name(paths["output"].name + ".tmp")
    if temporary.exists():
        raise CommonAnchorAnalysisError("temporary analysis output already exists")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(paths["output"])
    print(f"report={_relative(paths['output'])}")
    print(f"sha256={_sha256_file(paths['output'])}")
    print(f"analysis_identity={report['analysis_identity']}")
    print(f"classification={decision['classification']}")
    print(f"next_design={decision['next_design']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

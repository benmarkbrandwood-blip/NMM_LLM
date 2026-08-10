"""Audit final-policy sensitivity to read-only SpecialistDB projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.human_db import HumanDB  # noqa: E402
from game.board import BoardState  # noqa: E402
from game.notation import encode_move  # noqa: E402
from game.rules import get_game_phase  # noqa: E402
from learned_ai.data.data_contract import load_dataset_manifest  # noqa: E402
from learned_ai.data.specialist_db import SpecialistDB  # noqa: E402
from learned_ai.models.lookahead_advisor import LookaheadAdvisor  # noqa: E402
from learned_ai.models.scaffolded_encoder import (  # noqa: E402
    encode_position_with_lookahead,
)
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet  # noqa: E402
from learned_ai.sentinel.db_teacher import ExternalSolvedDB  # noqa: E402
from learned_ai.training.checkpoint_envelope import load_checkpoint  # noqa: E402
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402
from learned_ai.validation.specialist_db_policy_mechanism import (  # noqa: E402
    PROJECTION_MODES,
    SpecialistEvidenceProjection,
    evidence_record,
    project_wdl,
    summarize_primary_contrast,
)
from scripts import train_s_gen_v2 as trainer  # noqa: E402


DEFAULT_CORPUS = ROOT / "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
DEFAULT_PATHS_CONFIG = ROOT / "data/training_paths.local.json"
DEFAULT_MALOM_MANIFEST = ROOT / "data/manifests/malom-sector-corrected-v1.json"
SDB_MIN_SAMPLES = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_sha256(features: np.ndarray) -> str:
    array = np.asarray(features, dtype="<f4", order="C")
    digest = hashlib.sha256()
    digest.update(b"nmm.float32-c-matrix.v1\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sidecars(path: Path) -> list[str]:
    return [
        candidate.name
        for suffix in ("-wal", "-shm", "-journal")
        if (candidate := path.with_name(path.name + suffix)).exists()
    ]


def _git_commit() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise RuntimeError("tracked worktree must be clean for audit evidence")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _sqlite_asset_identity(path: Path) -> str:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or quick_check[0] != "ok":
            raise RuntimeError("SQLite quick_check did not return ok")
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


def _load_policy(
    model_config: dict[str, Any],
    state: dict[str, torch.Tensor],
) -> ScaffoldedPolicyNet:
    model = ScaffoldedPolicyNet.from_config(model_config).to("cpu")
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--specialist-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--malom-manifest", type=Path, default=DEFAULT_MALOM_MANIFEST)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-specialist-db-sha256", required=True)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--expected-experiment-id", required=True)
    parser.add_argument("--expected-game-count", type=int, required=True)
    parser.add_argument("--schedule-max-games", type=int, default=5000)
    parser.add_argument("--temp-start", type=float, default=0.90)
    return parser


def _mode_result(
    model: ScaffoldedPolicyNet,
    encoded: Any,
    *,
    temperature: float,
    malom_qualities: list[float | None],
    hit_count: int,
) -> dict[str, Any]:
    features = np.asarray(encoded.feat_matrix, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise RuntimeError("encoded feature matrix is invalid")
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logits_tensor = model.policy_logits(feature_tensor)
        probabilities_temp1 = torch.softmax(logits_tensor, dim=0)
        probabilities_scheduled = torch.softmax(
            logits_tensor / float(temperature), dim=0
        )
    if (
        logits_tensor.ndim != 1
        or logits_tensor.shape[0] != len(encoded.legal_moves)
        or not torch.isfinite(logits_tensor).all()
        or not torch.isfinite(probabilities_temp1).all()
        or not torch.isfinite(probabilities_scheduled).all()
    ):
        raise RuntimeError("policy output is invalid")
    argmax = int(torch.argmax(logits_tensor).item())
    return {
        "feature_matrix_sha256": _feature_sha256(features),
        "logits": [float(value) for value in logits_tensor.tolist()],
        "probabilities_temp1": [float(value) for value in probabilities_temp1.tolist()],
        "probabilities_scheduled": [
            float(value) for value in probabilities_scheduled.tolist()
        ],
        "argmax_index": argmax,
        "argmax_action": None,
        "argmax_malom_quality": malom_qualities[argmax],
        "specialist_db_hit_count": int(hit_count),
    }


def _mode_summaries(position_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in PROJECTION_MODES:
        qualities: Counter[str] = Counter()
        hits = 0
        states_with_hits = 0
        changes_from_full = 0
        by_phase: dict[str, Counter[str]] = defaultdict(Counter)
        for row in position_rows:
            mode_row = row["modes"][mode]
            quality = mode_row["argmax_malom_quality"]
            quality_key = "unknown" if quality is None else str(int(quality))
            qualities[quality_key] += 1
            hits += int(mode_row["specialist_db_hit_count"])
            if mode_row["specialist_db_hit_count"]:
                states_with_hits += 1
            if mode_row["argmax_index"] != row["modes"]["full"]["argmax_index"]:
                changes_from_full += 1
            by_phase[str(row["phase"])][quality_key] += 1
        result[mode] = {
            "states": len(position_rows),
            "states_with_specialist_db_hits": states_with_hits,
            "specialist_db_action_hits": hits,
            "argmax_changes_from_full": changes_from_full,
            "argmax_malom_quality_counts": dict(sorted(qualities.items())),
            "argmax_malom_quality_by_phase": {
                phase: dict(sorted(counter.items()))
                for phase, counter in sorted(by_phase.items())
            },
        }
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commit = _git_commit()
    checkpoint_path = _resolve(args.checkpoint)
    specialist_path = _resolve(args.specialist_db)
    corpus_path = _resolve(args.corpus)
    paths_config = _resolve(args.paths_config)
    malom_manifest_path = _resolve(args.malom_manifest)
    output_path = _resolve(args.output)
    for label, path in (
        ("checkpoint", checkpoint_path),
        ("SpecialistDB", specialist_path),
        ("corpus", corpus_path),
        ("paths config", paths_config),
        ("Malom manifest", malom_manifest_path),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} is not an existing file")
    if output_path.exists():
        raise RuntimeError("audit evidence output already exists")

    checkpoint_sha_before = _sha256(checkpoint_path)
    specialist_sha_before = _sha256(specialist_path)
    corpus_sha = _sha256(corpus_path)
    if checkpoint_sha_before != args.expected_checkpoint_sha256.lower():
        raise RuntimeError("checkpoint identity differs from the frozen contract")
    if specialist_sha_before != args.expected_specialist_db_sha256.lower():
        raise RuntimeError("SpecialistDB identity differs from the frozen contract")
    if corpus_sha != args.expected_corpus_sha256.lower():
        raise RuntimeError("corpus identity differs from the frozen contract")
    sidecars_before = _sidecars(specialist_path)
    if sidecars_before:
        raise RuntimeError("SpecialistDB has SQLite sidecars before the audit")

    settings = json.loads(paths_config.read_text(encoding="utf-8"))
    human_path = _resolve(settings["human_db_path"])
    malom_path = _resolve(settings["malom_db_path"])
    envelope = load_checkpoint(checkpoint_path, map_location="cpu")
    descriptor = envelope.descriptor
    checkpoint_state = envelope.payload.trainer_state
    if descriptor.implementation.get("trainer") != trainer.STAGE_TAG:
        raise RuntimeError("checkpoint is not an s_gen_v2 checkpoint")
    if descriptor.experiment_id != args.expected_experiment_id:
        raise RuntimeError("checkpoint experiment identity differs")
    if int(checkpoint_state["game_count"]) != args.expected_game_count:
        raise RuntimeError("checkpoint game count differs")
    model_config = dict(checkpoint_state["model_config"])
    candidate = _load_policy(model_config, envelope.payload.model_state)
    target = _load_policy(
        model_config,
        dict(checkpoint_state["target_network"]["model_state"]),
    )
    temperature = trainer._compute_temperature(
        int(checkpoint_state["game_count"]),
        args.schedule_max_games,
        args.temp_start,
    )
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("scheduled temperature is invalid")

    human = HumanDB(human_path, read_only=True)
    specialist = SpecialistDB(specialist_path, read_only=True)
    malom = ExternalSolvedDB(str(malom_path), strict=True)
    try:
        specialist.require_trusted_malom_labels()
        assets = dict(descriptor.asset_identities)
        if specialist_sha_before != assets.get("specialist_db"):
            raise RuntimeError("SpecialistDB identity differs from checkpoint")
        human_identity = _sqlite_asset_identity(human_path)
        if human_identity != assets.get("human_db"):
            raise RuntimeError("HumanDB identity differs from checkpoint")
        malom_manifest = load_dataset_manifest(malom_manifest_path)
        if malom_manifest.manifest_sha256 != assets.get("malom_tablebase"):
            raise RuntimeError("Malom identity differs from checkpoint")
        anchor = next(
            (
                component
                for component in malom_manifest.components
                if component.relative_path == "std.secval"
            ),
            None,
        )
        if anchor is None or _sha256(malom_path / "std.secval") != anchor.sha256:
            raise RuntimeError("Malom std.secval anchor differs")
        if not human.is_available() or not malom.is_available():
            raise RuntimeError("required read-only data is unavailable")

        advisor = LookaheadAdvisor(
            sentinel=None,
            evaluate_fn=trainer._simple_evaluate,
            value_net=None,
            gap_net=None,
            human_db=human,
            use_sentinel=True,
            ply_depth=12,
            sim_ply_depth=5,
            endgame_db=malom,
            strict=True,
        )
        advisor.set_frozen_model(target, device=torch.device("cpu"))
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        position_rows: list[dict[str, Any]] = []
        action_evidence_counts: Counter[str] = Counter()
        action_evidence_by_phase: dict[str, Counter[str]] = defaultdict(Counter)
        for entry in corpus["entries"]:
            board = BoardState.from_fen_string(entry["fen"])
            phase = str(entry["phase"])
            mode_encodings: dict[str, Any] = {}
            action_keys: list[str] | None = None
            for mode in PROJECTION_MODES:
                projection = SpecialistEvidenceProjection(specialist, mode)
                encoded = encode_position_with_lookahead(
                    board,
                    board.turn,
                    sentinel_advisor=None,
                    db=None,
                    value_net=None,
                    lookahead_advisor=advisor,
                    specialist_db=projection,
                    sdb_min_samples=SDB_MIN_SAMPLES,
                    strict=True,
                )
                if encoded is None or not encoded.legal_moves:
                    raise RuntimeError(
                        f"corpus entry {entry['index']} is not encodable"
                    )
                current_keys = [
                    encode_move(move, get_game_phase(board, board.turn))
                    for move in encoded.legal_moves
                ]
                if action_keys is None:
                    action_keys = current_keys
                elif current_keys != action_keys:
                    raise RuntimeError("legal-action order differs across modes")
                mode_encodings[mode] = encoded
            assert action_keys is not None

            action_rows: list[dict[str, Any]] = []
            qualities: list[float | None] = []
            hit_counts = {mode: 0 for mode in PROJECTION_MODES}
            for action_index, move in enumerate(mode_encodings["full"].legal_moves):
                after = board.apply_move(move)
                evidence = specialist.query_wdl_evidence(
                    after, min_samples=SDB_MIN_SAMPLES
                )
                record = evidence_record(evidence)
                projected = {
                    mode: (
                        None
                        if (value := project_wdl(evidence, mode)) is None
                        else [float(item) for item in value]
                    )
                    for mode in PROJECTION_MODES
                }
                for mode, hit in record["projection_hits"].items():
                    if hit:
                        hit_counts[mode] += 1
                quality_value = malom.query_move_quality(board, move)
                quality = None if quality_value is None else float(quality_value)
                if quality is not None and quality > 0:
                    raise RuntimeError("Malom returned a positive move quality")
                qualities.append(quality)
                if record["theoretical_wdl"] is not None:
                    action_evidence_counts["theoretical"] += 1
                    action_evidence_by_phase[phase]["theoretical"] += 1
                if record["empirical_distribution"] is not None:
                    action_evidence_counts["empirical"] += 1
                    action_evidence_by_phase[phase]["empirical"] += 1
                if record["theoretical_empirical_disagreement"]:
                    action_evidence_counts["disagreement"] += 1
                    action_evidence_by_phase[phase]["disagreement"] += 1
                action_evidence_counts["actions"] += 1
                action_evidence_by_phase[phase]["actions"] += 1
                action_rows.append(
                    {
                        "action_index": action_index,
                        "action": action_keys[action_index],
                        "resulting_fen": after.to_fen_string(),
                        "malom_quality": quality,
                        "specialist_db": record,
                        "projected_wdl": projected,
                    }
                )

            mode_rows: dict[str, dict[str, Any]] = {}
            for mode in PROJECTION_MODES:
                mode_row = _mode_result(
                    candidate,
                    mode_encodings[mode],
                    temperature=temperature,
                    malom_qualities=qualities,
                    hit_count=hit_counts[mode],
                )
                index = int(mode_row["argmax_index"])
                mode_row["argmax_action"] = action_keys[index]
                mode_rows[mode] = mode_row
            critical = any(value == 0 for value in qualities) and any(
                value is not None and value < 0 for value in qualities
            )
            position_rows.append(
                {
                    "index": int(entry["index"]),
                    "phase": phase,
                    "turn": str(board.turn),
                    "fen": str(entry["fen"]),
                    "critical": bool(critical),
                    "legal_actions": action_rows,
                    "modes": mode_rows,
                }
            )

        primary = summarize_primary_contrast(position_rows)
        report_core = {
            "schema_version": "nmm.specialist-db-policy-mechanism-audit.v1",
            "audit_id": "specialist-db-policy-mechanism-audit-v1",
            "scope": {
                "diagnostic_only": True,
                "development_corpus_not_held_out": True,
                "no_model_updates": True,
                "no_checkpoint_writes": True,
                "no_database_writes": True,
                "no_strength_or_promotion_claim": True,
            },
            "identities": {
                "git_commit": commit,
                "audit_script": _relative(Path(__file__)),
                "audit_script_sha256": _sha256(Path(__file__)),
                "checkpoint": _relative(checkpoint_path),
                "checkpoint_sha256_before": checkpoint_sha_before,
                "checkpoint_id": descriptor.checkpoint_id,
                "experiment_id": descriptor.experiment_id,
                "specialist_db": _relative(specialist_path),
                "specialist_db_sha256_before": specialist_sha_before,
                "specialist_db_label_version": specialist.malom_label_version,
                "specialist_db_sidecars_before": sidecars_before,
                "corpus": _relative(corpus_path),
                "corpus_sha256": corpus_sha,
                "paths_config_sha256": _sha256(paths_config),
                "human_db_identity": human_identity,
                "human_db_sha256": _sha256(human_path),
                "malom_manifest": _relative(malom_manifest_path),
                "malom_manifest_identity": malom_manifest.manifest_sha256,
                "malom_std_secval_sha256": anchor.sha256,
                "checkpoint_asset_identities": assets,
            },
            "route": {
                "device": "cpu",
                "projection_modes": list(PROJECTION_MODES),
                "specialist_db_min_samples": SDB_MIN_SAMPLES,
                "lookahead_ply_depth": 12,
                "simulation_ply_depth": 5,
                "temperature_scheduled": float(temperature),
                "sentinel": False,
                "value_net": False,
                "gap_net": False,
                "human_db_historical_malom_masked": True,
            },
            "checkpoint_state": {
                "game_count": int(checkpoint_state["game_count"]),
                "update_count": int(checkpoint_state["update_count"]),
                "target_games_since_update": int(
                    checkpoint_state["target_network"]["games_since_update"]
                ),
                "model_config": model_config,
            },
            "database_coverage": {
                "all": dict(sorted(action_evidence_counts.items())),
                "by_phase": {
                    phase: dict(sorted(counts.items()))
                    for phase, counts in sorted(action_evidence_by_phase.items())
                },
            },
            "mode_summaries": _mode_summaries(position_rows),
            "primary_contrast": primary,
            "positions": position_rows,
        }
    finally:
        human.close()
        specialist.close()

    checkpoint_sha_after = _sha256(checkpoint_path)
    specialist_sha_after = _sha256(specialist_path)
    sidecars_after = _sidecars(specialist_path)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("checkpoint changed during the read-only audit")
    if specialist_sha_after != specialist_sha_before:
        raise RuntimeError("SpecialistDB changed during the read-only audit")
    if sidecars_after:
        raise RuntimeError("SpecialistDB sidecars appeared during the audit")
    report_core["identities"].update(
        {
            "checkpoint_sha256_after": checkpoint_sha_after,
            "specialist_db_sha256_after": specialist_sha_after,
            "specialist_db_sidecars_after": sidecars_after,
        }
    )
    report = dict(report_core)
    report["evidence_id"] = canonical_sha256(report_core)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError("temporary audit output already exists")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output_path)
    print(f"report={_relative(output_path)}")
    print(f"sha256={_sha256(output_path)}")
    print(f"evidence_id={report['evidence_id']}")
    print(f"decision={report['primary_contrast']['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

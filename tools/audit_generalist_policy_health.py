"""Create a read-only, fixed-state health report for a Generalist checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
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
from learned_ai.data.specialist_db import SpecialistDB  # noqa: E402
from learned_ai.models.lookahead_advisor import LookaheadAdvisor  # noqa: E402
from learned_ai.models.scaffolded_encoder import (  # noqa: E402
    MOVE_FEAT_DIM,
    encode_position_with_lookahead,
)
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet  # noqa: E402
from learned_ai.sentinel.db_teacher import ExternalSolvedDB  # noqa: E402
from learned_ai.training.checkpoint_envelope import load_checkpoint  # noqa: E402
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402
from learned_ai.validation.generalist_policy_health import (  # noqa: E402
    PolicyHealthState,
    summarize_direct_lookahead_signal,
    summarize_policy_health,
)
from scripts import train_s_gen_v2 as trainer  # noqa: E402


DEFAULT_CORPUS = ROOT / "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
DEFAULT_PATHS_CONFIG = ROOT / "data/training_paths.local.json"
DEFAULT_MALOM_MANIFEST = (
    ROOT / "data/manifests/malom-sector-corrected-v1.json"
)
DEFAULT_CORPUS_SHA256 = (
    "cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _git_commit() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise RuntimeError("tracked worktree must be clean for policy-health evidence")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _load_policy(
    model_config: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    device: torch.device,
) -> ScaffoldedPolicyNet:
    model = ScaffoldedPolicyNet.from_config(model_config).to(device)
    model.load_state_dict(model_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _scratch_policy(
    model_config: dict[str, Any], seed: int, device: torch.device
) -> ScaffoldedPolicyNet:
    trainer._initialize_training_rngs(seed)
    model, start_game, _best, difficulty, source = trainer._load_model(
        device,
        None,
        tuple(model_config["policy_hidden"]),
        start_mode="fresh",
    )
    if (start_game, difficulty, source) != (0, trainer.DIFF_START, "scratch"):
        raise RuntimeError("scratch reconstruction contract drifted")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--specialist-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--expected-corpus-sha256",
        default=DEFAULT_CORPUS_SHA256,
    )
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument(
        "--malom-manifest",
        type=Path,
        default=DEFAULT_MALOM_MANIFEST,
    )
    parser.add_argument("--expected-experiment-id", required=True)
    parser.add_argument("--expected-game-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schedule-max-games", type=int, default=5000)
    parser.add_argument("--temp-start", type=float, default=0.90)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    commit = _git_commit()
    checkpoint_path = _resolve(str(args.checkpoint))
    specialist_path = _resolve(str(args.specialist_db))
    corpus_path = _resolve(str(args.corpus))
    paths_config = _resolve(str(args.paths_config))
    malom_manifest_path = _resolve(str(args.malom_manifest))
    output_path = _resolve(str(args.output))
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
        raise RuntimeError("policy-health evidence output already exists")
    corpus_sha256 = _sha256(corpus_path)
    if corpus_sha256 != args.expected_corpus_sha256.lower():
        raise RuntimeError("policy-health corpus identity does not match")

    settings = json.loads(paths_config.read_text(encoding="utf-8"))
    human_path = _resolve(settings["human_db_path"])
    malom_path = _resolve(settings["malom_db_path"])
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA policy-health audit is unavailable")

    envelope = load_checkpoint(checkpoint_path, map_location=device)
    descriptor = envelope.descriptor
    state = envelope.payload.trainer_state
    if descriptor.implementation.get("trainer") != trainer.STAGE_TAG:
        raise RuntimeError("checkpoint is not an s_gen_v2 checkpoint")
    if descriptor.experiment_id != args.expected_experiment_id:
        raise RuntimeError("checkpoint experiment identity does not match")
    if int(state["game_count"]) != args.expected_game_count:
        raise RuntimeError("checkpoint game count does not match")
    assets = dict(descriptor.asset_identities)
    model_config = dict(state["model_config"])
    candidate = _load_policy(model_config, envelope.payload.model_state, device)
    target = _load_policy(
        model_config,
        dict(state["target_network"]["model_state"]),
        device,
    )
    scratch = _scratch_policy(model_config, args.seed, device)
    temperature = trainer._compute_temperature(
        int(state["game_count"]), args.schedule_max_games, args.temp_start
    )

    human = HumanDB(human_path, read_only=True)
    specialist = SpecialistDB(specialist_path, read_only=True)
    malom = ExternalSolvedDB(str(malom_path), strict=True)
    try:
        specialist.require_trusted_malom_labels()
        specialist_sha256 = _sha256(specialist_path)
        expected_specialist = assets.get("specialist_db")
        if specialist_sha256 != expected_specialist:
            raise RuntimeError("SpecialistDB identity differs from the checkpoint")
        human_sha256 = _sha256(human_path)
        if human_sha256 != assets.get("human_db"):
            raise RuntimeError("HumanDB identity differs from the checkpoint")
        malom_manifest = load_dataset_manifest(malom_manifest_path)
        if malom_manifest.manifest_sha256 != assets.get("malom_tablebase"):
            raise RuntimeError("Malom manifest identity differs from the checkpoint")
        anchor = next(
            (
                component
                for component in malom_manifest.components
                if component.relative_path == "std.secval"
            ),
            None,
        )
        if anchor is None or _sha256(malom_path / "std.secval") != anchor.sha256:
            raise RuntimeError("Malom std.secval anchor identity differs")
        if not human.is_available() or not malom.is_available():
            raise RuntimeError("required read-only policy-health data is unavailable")

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
        )
        advisor.set_frozen_model(target, device=device)
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        states: list[PolicyHealthState] = []
        quality_counts: Counter[str] = Counter()
        for entry in corpus["entries"]:
            board = BoardState.from_fen_string(entry["fen"])
            encoded = encode_position_with_lookahead(
                board,
                board.turn,
                sentinel_advisor=None,
                db=None,
                value_net=None,
                lookahead_advisor=advisor,
                specialist_db=specialist,
                sdb_min_samples=3,
                strict=True,
            )
            if encoded is None or not encoded.legal_moves:
                raise RuntimeError(f"corpus entry {entry['index']} is not encodable")
            qualities: list[float] = []
            for move in encoded.legal_moves:
                quality = malom.query_move_quality(board, move)
                qualities.append(float("nan") if quality is None else float(quality))
                quality_counts[
                    "unknown" if quality is None else str(int(quality))
                ] += 1
            states.append(
                PolicyHealthState(
                    phase=str(entry["phase"]),
                    features=encoded.feat_matrix,
                    malom_qualities=np.asarray(qualities, dtype=np.float64),
                    heuristic_top1_idx=int(encoded.h_top1_idx),
                )
            )

        candidate_summary = summarize_policy_health(
            candidate, states, temperature=temperature, device=device
        )
        scratch_summary = summarize_policy_health(
            scratch, states, temperature=temperature, device=device
        )
        direct_signal = summarize_direct_lookahead_signal(
            states, signal_column=MOVE_FEAT_DIM
        )
        report_core = {
            "schema_version": "nmm.generalist-policy-health.v1",
            "scope": {
                "diagnostic_only": True,
                "no_model_updates": True,
                "no_checkpoint_writes": True,
                "no_database_writes": True,
                "no_strength_or_promotion_claim": True,
                "draft_phase_corpus_is_not_formal_evaluation": True,
            },
            "identities": {
                "git_commit": commit,
                "checkpoint": _relative(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "checkpoint_id": descriptor.checkpoint_id,
                "run_id": descriptor.run_id,
                "experiment_id": descriptor.experiment_id,
                "corpus": _relative(corpus_path),
                "corpus_sha256": corpus_sha256,
                "paths_config_sha256": _sha256(paths_config),
                "malom_manifest": _relative(malom_manifest_path),
                "malom_manifest_sha256": _sha256(malom_manifest_path),
                "malom_manifest_identity": malom_manifest.manifest_sha256,
                "malom_std_secval_sha256": anchor.sha256,
                "specialist_db": _relative(specialist_path),
                "specialist_db_sha256": specialist_sha256,
                "human_db_sha256": human_sha256,
                "checkpoint_asset_identities": assets,
            },
            "checkpoint_state": {
                "game_count": int(state["game_count"]),
                "update_count": int(state["update_count"]),
                "target_games_since_update": int(
                    state["target_network"]["games_since_update"]
                ),
                "temperature": float(temperature),
                "model_config": model_config,
            },
            "fixed_state_diagnostic": {
                "design": (
                    "candidate and reconstructed scratch weights on the same 64 "
                    "fixed positions, final target, production feature route, and "
                    "read-only data identities"
                ),
                "malom_move_quality_counts": dict(sorted(quality_counts.items())),
                "direct_lookahead_signal": direct_signal,
                "scratch": scratch_summary,
                "candidate": candidate_summary,
            },
        }
        report = dict(report_core)
        report["evidence_id"] = canonical_sha256(report_core)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        if temporary.exists():
            raise RuntimeError("temporary policy-health evidence already exists")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output_path)
        print(f"report={_relative(output_path)}")
        print(f"sha256={_sha256(output_path)}")
        print(f"evidence_id={report['evidence_id']}")
        return 0
    finally:
        human.close()
        specialist.close()
        malom.close()


if __name__ == "__main__":
    raise SystemExit(main())

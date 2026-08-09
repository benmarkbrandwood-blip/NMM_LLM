"""Probe exact-WDL preserving-set labels and gradients without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

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
    encode_position_with_lookahead,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB  # noqa: E402
from learned_ai.training.malom_policy_labels import (  # noqa: E402
    label_malom_preserving_actions,
)
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402
from learned_ai.validation.malom_policy_auxiliary_probe import (  # noqa: E402
    MalomPolicyAuxiliaryProbeState,
    run_in_memory_auxiliary_probe,
)
from scripts import train_s_gen_v2 as trainer  # noqa: E402


SCHEMA_VERSION = "nmm.malom-policy-auxiliary-gradient-probe.v2"
DEFAULT_CORPUS = ROOT / "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
DEFAULT_CORPUS_SHA256 = (
    "cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e"
)
DEFAULT_PATHS_CONFIG = ROOT / "data/training_paths.local.json"
DEFAULT_MALOM_MANIFEST = ROOT / "data/manifests/malom-sector-corrected-v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _git_commit(expected: str) -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise RuntimeError("tracked worktree must be clean")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if commit != expected:
        raise RuntimeError(
            f"source commit differs: expected {expected}, observed {commit}"
        )
    return commit


def _sqlite_identity(path: Path) -> str:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        row = connection.execute("PRAGMA quick_check").fetchone()
        if row is None or row[0] != "ok":
            raise RuntimeError(f"SQLite quick_check failed for {path}")
        stat = path.stat()
        return canonical_sha256(
            {
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "page_count": connection.execute(
                    "PRAGMA page_count"
                ).fetchone()[0],
                "page_size": connection.execute(
                    "PRAGMA page_size"
                ).fetchone()[0],
                "schema_version": connection.execute(
                    "PRAGMA schema_version"
                ).fetchone()[0],
                "user_version": connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0],
            }
        )
    finally:
        connection.close()


def _finite_positive(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _positive_float_tuple(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be comma-separated finite positive numbers"
        ) from exc
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError(
            "must be comma-separated finite positive numbers"
        )
    return result


def _integer_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be comma-separated integers"
        ) from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("must contain unique integers")
    return result


def _positive_integer_tuple(value: str) -> tuple[int, ...]:
    result = _integer_tuple(value)
    if any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("must contain positive integers")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
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
    parser.add_argument("--seeds", type=_integer_tuple, default=(48, 49, 50))
    parser.add_argument(
        "--coefficients",
        type=_positive_float_tuple,
        default=(0.03, 0.1, 0.3),
    )
    parser.add_argument("--step-size", type=_finite_positive, default=0.0001)
    parser.add_argument("--temperature", type=_finite_positive, default=0.9)
    parser.add_argument(
        "--policy-hidden",
        type=_positive_integer_tuple,
        default=(256, 128),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _fresh_policy(
    seed: int,
    policy_hidden: tuple[int, ...],
    device: torch.device,
):
    trainer._initialize_training_rngs(seed)
    model, start_game, _best, difficulty, source = trainer._load_model(
        device,
        None,
        policy_hidden,
        start_mode="fresh",
    )
    if (start_game, difficulty, source) != (0, trainer.DIFF_START, "scratch"):
        raise RuntimeError("fresh policy reconstruction contract drifted")
    return model


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_commit = _git_commit(args.expected_source_commit)
    output_path = _resolve(args.output)
    corpus_path = _resolve(args.corpus)
    paths_config_path = _resolve(args.paths_config)
    malom_manifest_path = _resolve(args.malom_manifest)
    for label, path in (
        ("corpus", corpus_path),
        ("paths config", paths_config_path),
        ("Malom manifest", malom_manifest_path),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} is not an existing file: {path}")
    if output_path.exists():
        raise RuntimeError("probe output already exists")
    corpus_sha256 = _sha256(corpus_path)
    if corpus_sha256 != args.expected_corpus_sha256.lower():
        raise RuntimeError("probe corpus identity differs")

    settings = json.loads(paths_config_path.read_text(encoding="utf-8"))
    human_path = _resolve(settings["human_db_route_probe_snapshot_path"])
    specialist_path = _resolve(settings["specialist_db_route_probe_snapshot_path"])
    malom_path = _resolve(settings["malom_db_path"])
    for label, path in (
        ("HumanDB snapshot", human_path),
        ("SpecialistDB snapshot", specialist_path),
        ("Malom directory", malom_path),
    ):
        if not path.exists():
            raise RuntimeError(f"{label} does not exist: {path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA probe is unavailable")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(corpus.get("entries"), list) or not corpus["entries"]:
        raise RuntimeError("probe corpus has no entries")
    human_sha_before = _sha256(human_path)
    specialist_sha_before = _sha256(specialist_path)
    malom_anchor = malom_path / "std.secval"
    malom_anchor_sha_before = _sha256(malom_anchor)
    human_identity = _sqlite_identity(human_path)
    specialist_identity = _sqlite_identity(specialist_path)
    malom_manifest = load_dataset_manifest(malom_manifest_path)
    anchor = next(
        (
            component
            for component in malom_manifest.components
            if component.relative_path == "std.secval"
        ),
        None,
    )
    if anchor is None or anchor.sha256 != malom_anchor_sha_before:
        raise RuntimeError("Malom std.secval differs from its manifest")

    human = HumanDB(human_path, read_only=True)
    specialist = SpecialistDB(specialist_path, read_only=True)
    malom = ExternalSolvedDB(str(malom_path), strict=True)
    seed_reports: list[dict[str, Any]] = []
    try:
        if not human.is_available() or not malom.is_available():
            raise RuntimeError("required read-only probe data is unavailable")
        specialist.require_trusted_malom_labels()
        for seed in args.seeds:
            model = _fresh_policy(seed, args.policy_hidden, device)
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
            advisor.set_frozen_model(model, device=device)
            states: list[MalomPolicyAuxiliaryProbeState] = []
            root_counts: Counter[str] = Counter()
            phase_states: Counter[str] = Counter()
            phase_actions: Counter[str] = Counter()
            phase_informative: Counter[str] = Counter()
            preserving_actions = 0
            downgrading_actions = 0
            feature_seconds = 0.0
            label_seconds = 0.0
            for entry in corpus["entries"]:
                board = BoardState.from_fen_string(entry["fen"])
                started = time.perf_counter()
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
                feature_seconds += time.perf_counter() - started
                if encoded is None or not encoded.legal_moves:
                    raise RuntimeError(
                        f"corpus entry {entry['index']} is not encodable"
                    )
                started = time.perf_counter()
                labels = label_malom_preserving_actions(
                    malom,
                    board=board,
                    player=board.turn,
                    legal_moves=encoded.legal_moves,
                )
                label_seconds += time.perf_counter() - started
                phase = str(entry["phase"])
                states.append(
                    MalomPolicyAuxiliaryProbeState(
                        phase=phase,
                        features=encoded.feat_matrix,
                        preserving_mask=labels.preserving_mask,
                    )
                )
                root_counts[labels.root_wdl] += 1
                phase_states[phase] += 1
                phase_actions[phase] += len(encoded.legal_moves)
                if labels.downgrading_count > 0:
                    phase_informative[phase] += 1
                preserving_actions += labels.preserving_count
                downgrading_actions += labels.downgrading_count

            probe = run_in_memory_auxiliary_probe(
                model,
                states,
                temperature=args.temperature,
                device=device,
                coefficients=args.coefficients,
                step_size=args.step_size,
            )
            seed_reports.append(
                {
                    "seed": seed,
                    "coverage": {
                        "states": len(states),
                        "actions": preserving_actions + downgrading_actions,
                        "preserving_actions": preserving_actions,
                        "downgrading_actions": downgrading_actions,
                        "root_wdl_counts": dict(sorted(root_counts.items())),
                        "phase_states": dict(sorted(phase_states.items())),
                        "phase_actions": dict(sorted(phase_actions.items())),
                        "phase_informative_states": dict(
                            sorted(phase_informative.items())
                        ),
                        "feature_seconds": feature_seconds,
                        "label_seconds": label_seconds,
                        "labelled_actions_per_second": (
                            (preserving_actions + downgrading_actions)
                            / max(label_seconds, 1e-12)
                        ),
                    },
                    "gradient_probe": probe,
                }
            )
    finally:
        for component in (human, specialist):
            close = getattr(component, "close", None)
            if callable(close):
                close()

    mutation_checks = {
        "human_db_unchanged": _sha256(human_path) == human_sha_before,
        "specialist_db_unchanged": (
            _sha256(specialist_path) == specialist_sha_before
        ),
        "malom_anchor_unchanged": (
            _sha256(malom_anchor) == malom_anchor_sha_before
        ),
        "tracked_worktree_clean_after": (
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                text=True,
            ).strip()
            == ""
        ),
    }
    if not all(mutation_checks.values()):
        raise RuntimeError("probe mutated a protected source or tracked file")

    body = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "development_corpus": True,
            "no_training_run": True,
            "no_persistent_model_update": True,
            "no_checkpoint_write": True,
            "no_database_write": True,
            "no_strength_or_promotion_claim": True,
            "in_memory_sgd_direction_is_not_adam_trajectory": True,
        },
        "identities": {
            "source_commit": source_commit,
            "corpus": _portable(corpus_path),
            "corpus_sha256": corpus_sha256,
            "paths_config_sha256": _sha256(paths_config_path),
            "human_db": _portable(human_path),
            "human_db_sha256": human_sha_before,
            "human_db_identity": human_identity,
            "specialist_db": _portable(specialist_path),
            "specialist_db_sha256": specialist_sha_before,
            "specialist_db_identity": specialist_identity,
            "malom_manifest": _portable(malom_manifest_path),
            "malom_manifest_sha256": _sha256(malom_manifest_path),
            "malom_manifest_identity": malom_manifest.manifest_sha256,
            "malom_std_secval_sha256": malom_anchor_sha_before,
        },
        "configuration": {
            "seeds": list(args.seeds),
            "coefficients": list(args.coefficients),
            "directional_step_size": args.step_size,
            "temperature": args.temperature,
            "policy_hidden": list(args.policy_hidden),
            "device": str(device),
            "feature_route": {
                "sentinel": False,
                "value_net": False,
                "gap_net": False,
                "human_db": True,
                "specialist_db": True,
                "malom_features": False,
                "malom_labels": True,
                "lookahead_ply_depth": 12,
                "lookahead_sim_ply_depth": 5,
                "frozen_target": "same fresh initialization as learner",
            },
        },
        "mutation_checks": mutation_checks,
        "seed_reports": seed_reports,
    }
    report = {**body, "probe_identity": canonical_sha256(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

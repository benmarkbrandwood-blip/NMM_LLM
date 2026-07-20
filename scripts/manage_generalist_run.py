"""Prepare, authorize, inspect, and supervise bounded Generalist runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.training.managed_generalist import (
    ManagedContractError,
    ManagedPlan,
    authorize_plan,
    managed_status,
    publish_managed_plan,
    run_authorized_plan,
    run_next_segment,
)
from learned_ai.training.generalist_preflight import (
    resume_config_sha256,
    validate_generalist_configuration,
)
from learned_ai.training.generalist_run_manifest import utc_now_text
from scripts import train_s_gen_v2 as trainer


DEFAULT_NODE_BUDGET = 500_000
DEFAULT_MAX_GAMES = 5_000
DEFAULT_SEGMENT_GAMES = 250


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if Path(top).resolve() != _ROOT.resolve():
        raise ManagedContractError("the primary workspace is not the repository root")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _default_plan_id(commit: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"managed-v4-{stamp}-{commit[:8]}"


def _common_trainer_args(args: argparse.Namespace, paths_config: Path) -> list[str]:
    return [
        "--experiment-id",
        args.experiment_id,
        "--paths-config",
        str(paths_config),
        "--max-games",
        str(args.max_games),
        "--seed",
        "42",
        "--temp-start",
        "0.90",
        "--self-play-ratio",
        "0.50",
        "--update-target-every",
        "50",
        "--max-ply",
        "60",
        "--max-ply-branch",
        "60",
        "--max-branches-per-game",
        "0",
        "--sim-ply-depth",
        "5",
        "--batch-games",
        "1",
        "--log-every",
        "50",
        "--heuristic-node-budget",
        str(args.heuristic_node_budget),
        "--no-sentinel",
        "--no-value-net",
        "--no-gap-net",
        "--no-s1a-warmstart",
        "--no-imitation-mix",
        "--no-s1b-refresher",
    ]


def _prepare(args: argparse.Namespace) -> dict:
    commit, dirty = _git_state()
    if dirty:
        raise ManagedContractError("prepare requires a clean Git worktree")
    paths_config = Path(args.paths_config).resolve(strict=True)
    control_dir = Path(args.control_dir).resolve(strict=False)
    plan_path = control_dir / "plan.json"
    common_args = _common_trainer_args(args, paths_config)

    parser = trainer._build_argument_parser()
    semantic_args = parser.parse_args(["--preflight", "long-run", *common_args])
    trainer._configure_paths(semantic_args)
    validate_generalist_configuration(semantic_args)
    plan = ManagedPlan(
        plan_id=args.plan_id or _default_plan_id(commit),
        created_at_utc=utc_now_text(),
        objective="corrected-v4-single-machine-single-GPU-baseline",
        experiment_id=args.experiment_id,
        git_commit=commit,
        control_dir=str(control_dir),
        paths_config=str(paths_config),
        paths_config_sha256=_file_sha256(paths_config),
        resume_config_sha256=resume_config_sha256(semantic_args),
        max_games=args.max_games,
        segment_games=args.segment_games,
        max_wall_hours=args.max_wall_hours,
        common_trainer_args=tuple(common_args),
        allow_safe_exact_resume=True,
        publication_allowed=False,
        promotion_allowed=False,
    )
    publish_managed_plan(plan_path, plan)
    return {
        "state": "awaiting_product_authorization",
        "summary": "The bounded technical plan is frozen; no training was started.",
        "needs_product_decision": True,
        "product_decision": "Approve or reject the objective and resource envelope.",
        "plan_path": str(plan_path),
        "authorization_path": str(control_dir / "authorization.json"),
        "plan_sha256": plan.plan_sha256,
        "resource_envelope": {
            "max_games": plan.max_games,
            "segment_games": plan.segment_games,
            "max_wall_hours": plan.max_wall_hours,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--control-dir", required=True)
    prepare.add_argument("--max-wall-hours", required=True, type=float)
    prepare.add_argument("--plan-id")
    prepare.add_argument(
        "--paths-config",
        default=str(_ROOT / "data" / "training_paths.local.json"),
    )
    prepare.add_argument(
        "--experiment-id",
        default="dev-v4-managed-baseline-v1",
    )
    prepare.add_argument("--max-games", type=int, default=DEFAULT_MAX_GAMES)
    prepare.add_argument(
        "--segment-games",
        type=int,
        default=DEFAULT_SEGMENT_GAMES,
    )
    prepare.add_argument(
        "--heuristic-node-budget",
        type=int,
        default=DEFAULT_NODE_BUDGET,
        help="Technical fixed-work setting; normally selected by the Agent",
    )

    authorize = commands.add_parser("authorize")
    authorize.add_argument("--plan", required=True)
    authorize.add_argument("--authorization", required=True)
    authorize.add_argument("--authorized-by", required=True)
    authorize.add_argument("--decision-note", required=True)

    status = commands.add_parser("status")
    status.add_argument("--plan", required=True)
    status.add_argument("--authorization", required=True)

    run_next = commands.add_parser("run-next")
    run_next.add_argument("--plan", required=True)
    run_next.add_argument("--authorization", required=True)

    run_all = commands.add_parser("run-authorized")
    run_all.add_argument("--plan", required=True)
    run_all.add_argument("--authorization", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = _prepare(args)
        elif args.command == "authorize":
            authorize_plan(
                args.plan,
                args.authorization,
                authorized_by=args.authorized_by,
                decision_note=args.decision_note,
            )
            result = managed_status(args.plan, args.authorization)
        elif args.command == "status":
            result = managed_status(args.plan, args.authorization)
        elif args.command == "run-next":
            result = run_next_segment(args.plan, args.authorization)
        else:
            result = run_authorized_plan(args.plan, args.authorization)
    except (ManagedContractError, FileNotFoundError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "state": "stopped",
                    "summary": str(exc),
                    "needs_product_decision": False,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

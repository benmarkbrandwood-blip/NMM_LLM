"""Prepare, authorize, inspect, and supervise bounded Generalist runs."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.training.managed_generalist import (  # noqa: E402
    ManagedContractError,
    ManagedPlan,
    PolicyHealthGate,
    authorize_plan,
    managed_status,
    publish_managed_plan,
    recover_failed_segment,
    recover_interrupted_segment,
    run_authorized_plan,
    run_next_segment,
)
from learned_ai.training.generalist_preflight import (  # noqa: E402
    resume_config_sha256,
    validate_generalist_configuration,
)
from learned_ai.training.generalist_run_manifest import utc_now_text  # noqa: E402
from scripts import train_s_gen_v2 as trainer  # noqa: E402


DEFAULT_NODE_BUDGET = 500_000
DEFAULT_MAX_GAMES = 5_000
DEFAULT_SEGMENT_GAMES = 250
DEFAULT_SANMILL_NODE_LADDER = "1000,5000,25000,100000,500000"
DEFAULT_SANMILL_STAGE_GAMES = "500,500,500,1000,2500"
DEFAULT_POLICY_HEALTH_CORPUS = (
    _ROOT / "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
)
DEFAULT_POLICY_HEALTH_CORPUS_SHA256 = (
    "cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e"
)
DEFAULT_POLICY_HEALTH_AUDIT = _ROOT / "tools/audit_generalist_policy_health.py"


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
    self_play_ratio = args.self_play_ratio
    if self_play_ratio is None:
        self_play_ratio = (
            0.60 if args.engine_profile == "sanmill-fixed-resource" else 0.50
        )
    common_args = [
        "--experiment-id",
        args.experiment_id,
        "--paths-config",
        str(paths_config),
        "--max-games",
        str(args.max_games),
        "--seed",
        str(args.seed),
        "--temp-start",
        "0.90",
        "--mill-bonus-mode",
        args.mill_bonus_mode,
        "--malom-policy-aux-coef",
        str(args.malom_policy_aux_coef),
        "--malom-policy-aux-mode",
        args.malom_policy_aux_mode,
        "--malom-policy-aux-target-ratio",
        str(args.malom_policy_aux_target_ratio),
        "--malom-policy-aux-coef-cap",
        str(args.malom_policy_aux_coef_cap),
        "--malom-policy-aux-denominator-floor",
        str(args.malom_policy_aux_denominator_floor),
        "--specialist-read-mode",
        args.specialist_read_mode,
        "--self-play-ratio",
        str(self_play_ratio),
        "--update-target-every",
        "50",
        "--max-ply",
        str(args.max_ply),
        "--max-ply-branch",
        str(args.max_ply),
        "--max-branches-per-game",
        "0",
        "--sim-ply-depth",
        "5",
        "--batch-games",
        "1",
        "--log-every",
        "50",
        "--no-sentinel",
        "--no-value-net",
        "--no-gap-net",
        "--no-s1a-warmstart",
        "--no-imitation-mix",
        "--no-s1b-refresher",
        "--no-opening-forcing",
    ]
    if args.engine_profile == "sanmill-fixed-resource":
        level_count = len(args.sanmill_node_ladder.split(","))
        common_args.extend(
            (
                "--referee-engine",
                "sanmill",
                "--opponent-engine",
                "sanmill",
                "--sanmill-node-ladder",
                args.sanmill_node_ladder,
                "--sanmill-stage-games",
                args.sanmill_stage_games,
                "--curriculum-advance-policy",
                "fixed-resource",
                "--diff-start",
                "1",
                "--diff-max",
                str(level_count),
                "--minimal-rollouts",
                "--no-recovery",
            )
        )
    else:
        common_args.extend(
            ("--heuristic-node-budget", str(args.heuristic_node_budget))
        )
    specialist_db = getattr(args, "specialist_db", None)
    if specialist_db:
        common_args.extend(("--specialist-db", str(Path(specialist_db).resolve())))
    return common_args


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
    policy_health = None
    if args.policy_health_gate:
        corpus = Path(args.policy_health_corpus).resolve(strict=True)
        audit_script = DEFAULT_POLICY_HEALTH_AUDIT.resolve(strict=True)
        corpus_sha256 = _file_sha256(corpus)
        if corpus_sha256 != DEFAULT_POLICY_HEALTH_CORPUS_SHA256:
            raise ManagedContractError(
                "the fixed policy-health corpus identity differs"
            )
        policy_health = PolicyHealthGate(
            corpus_path=str(corpus),
            corpus_sha256=corpus_sha256,
            audit_script_path=str(audit_script),
            audit_script_sha256=_file_sha256(audit_script),
            exact_critical_states=29,
            required_direct_preserving_rate=1.0,
            min_candidate_preserving_rate=0.50,
            min_candidate_logit_margin=-0.10,
            device=args.policy_health_device,
        )
    plan = ManagedPlan(
        plan_id=args.plan_id or _default_plan_id(commit),
        created_at_utc=utc_now_text(),
        objective=args.objective,
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
        policy_health=policy_health,
        completion_game_bound=args.completion_game_bound,
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
            "max_games": plan.game_bound,
            "schedule_max_games": plan.max_games,
            "segment_games": plan.segment_games,
            "max_wall_hours": plan.max_wall_hours,
        },
        "policy_health_gate": (
            None if policy_health is None else policy_health.to_dict()
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--control-dir", required=True)
    prepare.add_argument("--max-wall-hours", required=True, type=float)
    prepare.add_argument("--plan-id")
    prepare.add_argument(
        "--objective",
        required=True,
        help="Product-approved purpose of this successor experiment",
    )
    prepare.add_argument(
        "--paths-config",
        default=str(_ROOT / "data" / "training_paths.local.json"),
    )
    prepare.add_argument(
        "--experiment-id",
        required=True,
        help="New experiment identity; do not reuse a completed run ID",
    )
    prepare.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Explicit trainer seed recorded in the immutable run plan",
    )
    prepare.add_argument("--max-games", type=int, default=DEFAULT_MAX_GAMES)
    prepare.add_argument(
        "--completion-game-bound",
        type=int,
        default=None,
        help=(
            "Optional controller completion ceiling. The trainer still uses "
            "--max-games as its global schedule horizon."
        ),
    )
    prepare.add_argument(
        "--segment-games",
        type=int,
        default=DEFAULT_SEGMENT_GAMES,
    )
    prepare.add_argument(
        "--engine-profile",
        choices=("local-game-ai", "sanmill-fixed-resource"),
        default="local-game-ai",
        help="Explicit opponent/referee and resource-curriculum profile",
    )
    prepare.add_argument(
        "--self-play-ratio",
        type=float,
        default=None,
        help="Frozen-target share; defaults to 0.50 local or 0.60 Sanmill",
    )
    prepare.add_argument(
        "--heuristic-node-budget",
        type=int,
        default=DEFAULT_NODE_BUDGET,
        help="Technical fixed-work setting; normally selected by the Agent",
    )
    prepare.add_argument(
        "--sanmill-node-ladder",
        default=DEFAULT_SANMILL_NODE_LADDER,
        help="Comma-separated fixed-node ceilings for the Sanmill profile",
    )
    prepare.add_argument(
        "--sanmill-stage-games",
        default=DEFAULT_SANMILL_STAGE_GAMES,
        help="Comma-separated global game durations for the Sanmill profile",
    )
    prepare.add_argument(
        "--max-ply",
        required=True,
        type=int,
        help=(
            "Explicit experiment truncation ceiling in complete logical plies; "
            "this is not a rules draw"
        ),
    )
    prepare.add_argument(
        "--mill-bonus-mode",
        required=True,
        choices=trainer.MILL_BONUS_MODES,
        help=(
            "Explicit reward-shaping contract. New retained successors must "
            "not inherit the trainer's legacy compatibility default."
        ),
    )
    prepare.add_argument(
        "--malom-policy-aux-coef",
        type=trainer._finite_nonnegative_float,
        default=0.0,
        help=(
            "Explicit A2C preserving-set auxiliary coefficient; zero keeps "
            "the historical update"
        ),
    )
    prepare.add_argument(
        "--malom-policy-aux-mode",
        choices=trainer.MALOM_POLICY_AUX_MODES,
        default="fixed",
        help="Explicit fixed or per-batch policy-head-normalized scaling rule",
    )
    prepare.add_argument(
        "--malom-policy-aux-target-ratio",
        type=trainer._finite_positive_float,
        default=trainer.DEFAULT_MALOM_POLICY_AUX_TARGET_RATIO,
        help="Normalized applied-gradient target relative to the policy head",
    )
    prepare.add_argument(
        "--malom-policy-aux-coef-cap",
        type=trainer._finite_positive_float,
        default=trainer.DEFAULT_MALOM_POLICY_AUX_COEF_CAP,
        help="Maximum detached coefficient in normalized mode",
    )
    prepare.add_argument(
        "--malom-policy-aux-denominator-floor",
        type=trainer._finite_positive_float,
        default=trainer.DEFAULT_MALOM_POLICY_AUX_DENOMINATOR_FLOOR,
        help="Fail-closed raw auxiliary gradient denominator floor",
    )
    prepare.add_argument(
        "--specialist-db",
        default=None,
        help=(
            "Optional disposable SpecialistDB path. Required for smoke so the "
            "active sector-corrected baseline database is never reused."
        ),
    )
    prepare.add_argument(
        "--specialist-read-mode",
        choices=trainer.SPECIALIST_READ_MODES,
        default="full",
        help=(
            "Explicit training read projection for the writable SpecialistDB; "
            "defaults to the historical empirical-first route"
        ),
    )
    prepare.add_argument(
        "--policy-health-gate",
        action="store_true",
        help=(
            "Require the frozen fixed-state policy-health audit after every "
            "completed segment"
        ),
    )
    prepare.add_argument(
        "--policy-health-corpus",
        default=str(DEFAULT_POLICY_HEALTH_CORPUS),
        help="Fixed phase-covered corpus used by the policy-health gate",
    )
    prepare.add_argument(
        "--policy-health-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for the read-only segment-boundary policy audit",
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

    recover = commands.add_parser("recover-interrupted")
    recover.add_argument("--plan", required=True)
    recover.add_argument("--authorization", required=True)

    recover_failed = commands.add_parser("recover-failed")
    recover_failed.add_argument("--plan", required=True)
    recover_failed.add_argument("--authorization", required=True)
    recover_failed.add_argument("--technical-evidence", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        # This CLI is consumed as a JSON subprocess protocol. Keep incidental
        # trainer, path-resolution, and supervisor diagnostics on stderr so
        # stdout always contains exactly one machine-readable document.
        with contextlib.redirect_stdout(sys.stderr):
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
            elif args.command == "recover-interrupted":
                result = recover_interrupted_segment(args.plan, args.authorization)
            elif args.command == "recover-failed":
                result = recover_failed_segment(
                    args.plan,
                    args.authorization,
                    technical_evidence_path=args.technical_evidence,
                )
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

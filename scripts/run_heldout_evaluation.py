#!/usr/bin/env python3
"""Preflight, run, resume, or recompute the frozen retained-v2 evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.heldout_evaluation import (  # noqa: E402
    HeldoutEvaluationError,
    build_readiness_report,
    load_frozen_heldout_contract,
    load_game_ledger,
    load_runtime_spec,
    recompute_heldout_evaluation,
    require_ready,
    resolve_heldout_paths,
    run_frozen_heldout_evaluation,
)


def _write_stdout(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _load() -> tuple[object, object]:
    contract = load_frozen_heldout_contract()
    return contract, resolve_heldout_paths(contract, _ARGS.paths_config)


def _preflight(*, resume: bool) -> int:
    contract, paths = _load()
    report = build_readiness_report(contract, paths, resume=resume)
    _write_stdout(report)
    return 0 if report["ready"] else 2


def _run(*, resume: bool, launch: bool) -> int:
    if not launch:
        raise HeldoutEvaluationError("run requires the explicit --launch flag")
    contract, paths = _load()
    readiness = build_readiness_report(contract, paths, resume=resume)
    require_ready(readiness)
    result = run_frozen_heldout_evaluation(
        contract,
        paths,
        readiness,
        resume=resume,
    )
    _write_stdout(result)
    return 0


def _recompute() -> int:
    _contract, paths = _load()
    result = recompute_heldout_evaluation(paths.output_spec, paths.output_ledger)
    if paths.output_report.is_file():
        persisted = json.loads(paths.output_report.read_text(encoding="utf-8"))
        if persisted != result:
            raise HeldoutEvaluationError("persisted report differs from recomputation")
    _write_stdout(result)
    return 0


def _status() -> int:
    _contract, paths = _load()
    if not paths.output_spec.is_file():
        _write_stdout(
            {
                "evaluation_id": ("dev-v4-sanmill-corrected-retained-v2-heldout-v1"),
                "status": "not_started",
                "authorization_consumed": False,
                "completed_games": 0,
            }
        )
        return 0
    spec = load_runtime_spec(paths.output_spec)
    records, tail = load_game_ledger(spec, paths.output_ledger)
    failure = paths.output_root / "failure.json"
    launch = paths.output_root / "launch.json"
    status = (
        "completed"
        if paths.output_report.is_file()
        else "failed"
        if failure.is_file()
        else "partial"
    )
    _write_stdout(
        {
            "evaluation_id": spec["evaluation_id"],
            "spec_identity": spec["spec_identity"],
            "status": status,
            "authorization_consumed": launch.is_file(),
            "completed_games": len(records),
            "ledger_tail_record_sha256": tail,
            "failure_recorded": failure.is_file(),
        }
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        default=str(_ROOT / "data/training_paths.local.json"),
        help="ignored per-machine path registry",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="run all read-only launch gates")
    resume_preflight = commands.add_parser(
        "preflight-resume", help="audit one existing same-spec partial run"
    )
    resume_preflight.set_defaults(resume=True)
    run = commands.add_parser("run", help="consume the grant and run once")
    run.add_argument("--launch", action="store_true")
    resume = commands.add_parser(
        "resume", help="resume only the missing suffix of the same run"
    )
    resume.add_argument("--launch", action="store_true")
    commands.add_parser("recompute", help="recompute a complete ledger")
    commands.add_parser("status", help="inspect local run state")
    return parser


def main() -> int:
    global _ARGS
    _ARGS = _parser().parse_args()
    try:
        if _ARGS.command == "preflight":
            return _preflight(resume=False)
        if _ARGS.command == "preflight-resume":
            return _preflight(resume=True)
        if _ARGS.command == "run":
            return _run(resume=False, launch=_ARGS.launch)
        if _ARGS.command == "resume":
            return _run(resume=True, launch=_ARGS.launch)
        if _ARGS.command == "recompute":
            return _recompute()
        return _status()
    except HeldoutEvaluationError as exc:
        print(f"held-out evaluation stopped: {exc}", file=sys.stderr)
        return 2


_ARGS: argparse.Namespace


if __name__ == "__main__":
    raise SystemExit(main())

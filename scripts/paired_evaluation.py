#!/usr/bin/env python3
"""Freeze, run, and recompute fixed-N paired model-bundle evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.delivery.model_bundle import verify_model_bundle
from learned_ai.evaluation.paired_protocol import (
    EvaluationSpec,
    build_runtime_identity,
    freeze_evaluation_spec,
    recompute_evaluation,
    run_paired_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("candidate")
    freeze.add_argument("baseline")
    freeze.add_argument("corpus", help="JSON list of BoardState FEN strings")
    freeze.add_argument("output")
    freeze.add_argument("--evaluation-id", required=True)
    freeze.add_argument("--pairs", type=int, required=True)
    freeze.add_argument("--seed", type=int, required=True)
    freeze.add_argument("--max-ply", type=int, default=200)
    freeze.add_argument("--acceptance-margin", type=float, default=0.0)
    freeze.add_argument("--rejection-margin", type=float, default=0.0)
    freeze.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run = commands.add_parser("run")
    run.add_argument("spec")
    run.add_argument("candidate")
    run.add_argument("baseline")
    run.add_argument("output")
    run.add_argument("--device", choices=("cpu", "cuda"), required=True)
    recompute = commands.add_parser("recompute")
    recompute.add_argument("spec")
    recompute.add_argument("records")
    args = parser.parse_args()
    if args.command == "freeze":
        runtime = build_runtime_identity(args.device)
        candidate = verify_model_bundle(args.candidate, device=args.device)
        baseline = verify_model_bundle(args.baseline, device=args.device)
        positions = tuple(json.loads(Path(args.corpus).read_text(encoding="utf-8")))
        spec = EvaluationSpec(
            evaluation_id=args.evaluation_id, candidate_bundle=candidate["bundle_identity"],
            baseline_bundle=baseline["bundle_identity"], start_positions=positions,
            pairs=args.pairs, seed=args.seed, work_budget={"lookahead_rollouts_per_move": 0},
            max_ply=args.max_ply, rules_version="nmm-v4-corrected", confidence_z=1.96,
            acceptance_margin=args.acceptance_margin, rejection_margin=args.rejection_margin,
            runtime=runtime,
        )
        freeze_evaluation_spec(args.output, spec)
        result = spec.to_dict()
    elif args.command == "run":
        result = run_paired_evaluation(args.spec, args.candidate, args.baseline, args.output, device=args.device)
    else:
        result = recompute_evaluation(args.spec, args.records)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

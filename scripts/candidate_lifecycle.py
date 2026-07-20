#!/usr/bin/env python3
"""Manage local candidate validation, evaluation, and atomic promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.delivery.candidate_lifecycle import candidate_status, decide_candidate, register_candidate, transition_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("candidate_id")
    register.add_argument("bundle")
    for name in ("validate", "evaluate", "quarantine", "status"):
        command = commands.add_parser(name)
        command.add_argument("candidate_id")
        if name == "quarantine":
            command.add_argument("--reason", required=True)
    decide = commands.add_parser("decide")
    decide.add_argument("candidate_id")
    decide.add_argument("spec")
    decide.add_argument("records")
    decide.add_argument("--accepted-root", required=True)
    args = parser.parse_args()
    if args.command == "register":
        result = register_candidate(args.registry, args.candidate_id, args.bundle)
    elif args.command == "validate":
        result = transition_candidate(args.registry, args.candidate_id, "validating")
    elif args.command == "evaluate":
        result = transition_candidate(args.registry, args.candidate_id, "evaluating")
    elif args.command == "quarantine":
        result = transition_candidate(args.registry, args.candidate_id, "quarantined", reason=args.reason)
    elif args.command == "status":
        result = candidate_status(args.registry, args.candidate_id)
    else:
        result = decide_candidate(args.registry, args.candidate_id, args.spec, args.records, args.accepted_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

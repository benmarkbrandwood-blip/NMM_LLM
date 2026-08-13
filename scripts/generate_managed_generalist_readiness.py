"""Generate or verify one managed Generalist readiness evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.validation.managed_generalist_readiness import (  # noqa: E402
    ManagedReadinessError,
    generate_readiness,
    verify_persisted_readiness,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate")
    generate.add_argument("--plan", required=True, type=Path)
    generate.add_argument("--experiment-document", required=True, type=Path)
    generate.add_argument("--reviewed-main", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--readiness", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            report = generate_readiness(
                root=_ROOT,
                plan_path=args.plan,
                experiment_document=args.experiment_document,
                reviewed_main=args.reviewed_main,
                python_executable=sys.executable,
            )
        else:
            report = verify_persisted_readiness(
                root=_ROOT,
                readiness_path=args.readiness,
            )
    except (ManagedReadinessError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit or prepare the frozen six-arm mill-bonus ablation smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.validation.mill_bonus_ablation_readiness import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_REPORT,
    MillBonusAblationReadinessError,
    build_prepare_commands,
    inspect_published_source,
    inspect_preparation_evidence,
    inspect_runtime_identities,
    inspect_template,
    load_ablation_contract,
    prepare_ablation,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve(strict=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT),
        help="tracked source-only ablation contract",
    )
    parser.add_argument(
        "--paths-config",
        default=str(DEFAULT_PATHS_CONFIG),
        help="ignored machine-local path registry",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="new ignored combined readiness report",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="copy isolated databases and create plans; never authorizes training",
    )
    args = parser.parse_args(argv)
    contract_path = _resolve(args.contract)
    paths_config = _resolve(args.paths_config)
    report_path = _resolve(args.report)
    try:
        if args.prepare:
            result = prepare_ablation(
                root=ROOT,
                contract_path=contract_path,
                paths_config=paths_config,
                report_path=report_path,
                python_executable=sys.executable,
            )
        else:
            contract = load_ablation_contract(contract_path)
            source = inspect_published_source(ROOT, contract)
            result = {
                "state": "source_ready_for_local_preparation",
                "launch_authorized": False,
                "contract_identity": contract["plan_identity"],
                "source": source,
                "preparation_evidence": inspect_preparation_evidence(
                    ROOT, contract, source=source
                ),
                "template": inspect_template(ROOT, contract),
                "runtime": inspect_runtime_identities(
                    ROOT, paths_config, contract
                ),
                "commands": build_prepare_commands(
                    root=ROOT,
                    contract=contract,
                    paths_config=paths_config,
                    python_executable=sys.executable,
                ),
            }
    except (MillBonusAblationReadinessError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "state": "not_ready",
                    "launch_authorized": False,
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

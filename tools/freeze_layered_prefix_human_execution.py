"""Freeze strict Sanmill execution records for 21 HumanDB source members."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.layered_human_execution import (
    build_layered_human_execution,
    verify_layered_human_execution,
)
from learned_ai.evaluation.sanmill_uci import (
    PREFIX12_REPLAY_INSTALLATION_CONTRACT,
    inspect_sanmill_installation,
)
from learned_ai.training.run_contract import canonical_json_bytes


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=Path("data/training_paths.local.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    experiments = ROOT / "docs" / "experiments"
    evidence = ROOT / "docs" / "evidence"
    human_core = _load(
        experiments
        / "sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json"
    )
    source_core = _load(
        experiments
        / "sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json"
    )
    human_audit = _load(
        evidence / "sanmill-layered-human-source-audit-2026-07-25.json"
    )
    runtime = _load(
        experiments / "sanmill-prefix12-human-replay-runtime-2026-08-01.json"
    )
    installation = inspect_sanmill_installation(
        ROOT / args.paths_config,
        contract=PREFIX12_REPLAY_INSTALLATION_CONTRACT,
    )
    payload = build_layered_human_execution(
        human_core_decision=human_core,
        source_core_decision=source_core,
        human_audit=human_audit,
        runtime_decision=runtime,
        installation=installation,
    )
    summary = verify_layered_human_execution(
        payload,
        human_core_decision=human_core,
        source_core_decision=source_core,
        human_audit=human_audit,
        runtime_decision=runtime,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload) + b"\n")
    print(json.dumps(summary, sort_keys=True))
    print(f"human_execution_identity={payload['human_execution_identity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

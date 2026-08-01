"""Freeze the accepted 64-member twelve-ply executable prefix corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.layered_executable_corpus import (
    build_layered_executable_corpus,
    verify_layered_executable_corpus,
)
from learned_ai.training.run_contract import canonical_json_bytes


EXPERIMENTS = ROOT / "docs" / "experiments"
EVIDENCE = ROOT / "docs" / "evidence"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> dict[str, dict]:
    return {
        "composition_decision": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-composition-decision-2026-08-01.json"
        ),
        "book_core_decision": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json"
        ),
        "human_core_decision": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json"
        ),
        "perfect_core_decision": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-perfect-core-2026-08-01.json"
        ),
        "source_core_decision": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json"
        ),
        "sanmill_book_audit": _load(
            EVIDENCE / "sanmill-layered-book-source-audit-2026-07-25.json"
        ),
        "expert_book_audit": _load(
            EVIDENCE
            / "sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json"
        ),
        "human_audit": _load(
            EVIDENCE / "sanmill-layered-human-source-audit-2026-07-25.json"
        ),
        "perfect_audit": _load(
            EVIDENCE / "sanmill-layered-perfect-source-audit-2026-07-25.json"
        ),
        "human_execution": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-human-execution-2026-08-01.json"
        ),
        "runtime_decision": _load(
            EXPERIMENTS
            / "sanmill-prefix12-human-replay-runtime-2026-08-01.json"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    inputs = _inputs()
    payload = build_layered_executable_corpus(**inputs)
    summary = verify_layered_executable_corpus(payload, **inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload) + b"\n")
    print(json.dumps(summary, sort_keys=True))
    print(
        "executable_corpus_identity="
        f"{payload['executable_corpus_identity']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit the phase-development histories through pinned strict Sanmill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.phase_replay_development_corpus import (  # noqa: E402
    audit_phase_replays_with_sanmill,
    write_phase_replay_sanmill_audit,
)
from learned_ai.validation.sanmill_node_calibration import (  # noqa: E402
    load_local_installation,
)


DEFAULT_CORPUS = Path(
    "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json"
)
DEFAULT_OUTPUT = Path(
    "docs/evidence/phase-replay-development-corpus-sanmill-audit-2026-08-11.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=Path("data/training_paths.local.json"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus_path = (ROOT / args.corpus).resolve()
    output = (ROOT / args.output).resolve()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    installation = load_local_installation((ROOT / args.paths_config).resolve())
    report = audit_phase_replays_with_sanmill(corpus, installation)
    write_phase_replay_sanmill_audit(report, output, corpus=corpus)
    print(
        json.dumps(
            {
                "output": output.relative_to(ROOT).as_posix(),
                "audit_identity": report["audit_identity"],
                "fresh_process_count": report["fresh_process_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

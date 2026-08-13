#!/usr/bin/env python3
"""Freeze the zero-game retained-v3/v4 phase-process source corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.phase_corpus import load_pinned_tgf_fixture  # noqa: E402
from learned_ai.evaluation.retained_phase_process_corpus import (  # noqa: E402
    build_retained_phase_process_corpus,
    sha256_file,
    write_retained_phase_process_corpus,
)
from learned_ai.validation.sanmill_node_calibration import (  # noqa: E402
    load_local_installation,
)


DEFAULT_PHASE = Path("docs/experiments/dev-v4-phase-covered-corpus-v1.json")
DEFAULT_PRIOR_REPLAY = Path(
    "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json"
)
DEFAULT_OPENING = Path(
    "docs/experiments/"
    "sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.json"
)
DEFAULT_PLAN = Path(
    "docs/experiments/sanmill-retained-v3-v4-passivity-diagnostic-v1.json"
)
DEFAULT_OUTPUT = Path(
    "docs/experiments/sanmill-retained-v3-v4-phase-process-corpus-v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=Path("data/training_paths.local.json"),
    )
    parser.add_argument("--phase-corpus", type=Path, default=DEFAULT_PHASE)
    parser.add_argument("--prior-replay", type=Path, default=DEFAULT_PRIOR_REPLAY)
    parser.add_argument("--opening-corpus", type=Path, default=DEFAULT_OPENING)
    parser.add_argument("--passivity-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths_config = (ROOT / args.paths_config).resolve()
    phase_path = (ROOT / args.phase_corpus).resolve()
    replay_path = (ROOT / args.prior_replay).resolve()
    opening_path = (ROOT / args.opening_corpus).resolve()
    plan_path = (ROOT / args.passivity_plan).resolve()
    output = (ROOT / args.output).resolve()
    local_paths = _load(paths_config)
    human_db_path = Path(local_paths["human_db_path"])
    if not human_db_path.is_absolute():
        human_db_path = (ROOT / human_db_path).resolve()
    fixture, fixture_source = load_pinned_tgf_fixture(paths_config)
    payload = build_retained_phase_process_corpus(
        phase_corpus=_load(phase_path),
        phase_corpus_file_sha256=sha256_file(phase_path),
        prior_replay_corpus=_load(replay_path),
        prior_replay_file_sha256=sha256_file(replay_path),
        fixture=fixture,
        fixture_source=fixture_source,
        opening_corpus=_load(opening_path),
        opening_corpus_file_sha256=sha256_file(opening_path),
        passivity_plan=_load(plan_path),
        passivity_plan_file_sha256=sha256_file(plan_path),
        installation=load_local_installation(paths_config),
        human_db_path=human_db_path,
        repository_root=ROOT,
    )
    write_retained_phase_process_corpus(payload, output)
    print(
        json.dumps(
            {
                "output": output.relative_to(ROOT).as_posix(),
                "corpus_identity": payload["corpus_identity"],
                "records_identity": payload["records_identity"],
                "records": len(payload["records"]),
                "strict_replay_audit_identity": payload["strict_replay_audit"][
                    "audit_identity"
                ],
                "exposure_audit_identity": payload["exposure_audit"][
                    "audit_identity"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

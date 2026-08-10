"""Generate the candidate-blind replayable phase-development corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.phase_corpus import (  # noqa: E402
    load_pinned_tgf_fixture,
)
from learned_ai.evaluation.phase_replay_development_corpus import (  # noqa: E402
    SOURCE_CORPUS_PATH,
    build_phase_replay_development_corpus,
    source_corpus_sha256,
    write_phase_replay_development_corpus,
)


DEFAULT_OUTPUT = Path(
    "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=Path("data/training_paths.local.json"),
    )
    parser.add_argument("--source-corpus", type=Path, default=Path(SOURCE_CORPUS_PATH))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_path = (ROOT / args.source_corpus).resolve()
    output = (ROOT / args.output).resolve()
    phase_corpus = json.loads(source_path.read_text(encoding="utf-8"))
    fixture, fixture_source = load_pinned_tgf_fixture(
        (ROOT / args.paths_config).resolve()
    )
    payload = build_phase_replay_development_corpus(
        phase_corpus,
        fixture,
        fixture_source,
        source_corpus_sha256=source_corpus_sha256(source_path),
    )
    write_phase_replay_development_corpus(payload, output)
    print(
        json.dumps(
            {
                "output": output.relative_to(ROOT).as_posix(),
                "corpus_identity": payload["corpus_identity"],
                "record_count": len(payload["records"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

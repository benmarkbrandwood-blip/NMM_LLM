"""tools/build_human_db.py — Build or incrementally update data/human_db.sqlite.

Reads human-vs-human game JSONL files, aggregates per-position move statistics,
annotates each resulting position with Malom WDL + DTW, and writes a SQLite
database that HumanDB (ai/human_db.py) reads at server startup in milliseconds
instead of scanning tens of thousands of files.

Delegates all pipeline logic to `tools/_human_db_build.py` so this entry point
and `tools/build_human_db_sha.py` cannot drift.  The `_sha` variant only
differs in that it additionally emits `<output>.sha256` for download
verification; use that one whenever you're producing a distribution artefact.

Usage
-----
    # Full build (first time, or after --rebuild):
    .venv/bin/python tools/build_human_db.py \\
        --games-dir data/human_games \\
        --output data/human_db.sqlite \\
        --malom-db /path/to/Malom_Standard/Std_DD_89adjusted

    # Incremental update (only processes new/changed files):
    .venv/bin/python tools/build_human_db.py --update

    # Skip Malom annotation (faster when DB is not mounted):
    .venv/bin/python tools/build_human_db.py --update --no-malom

    # Force full rebuild from scratch:
    .venv/bin/python tools/build_human_db.py --rebuild

    # Fixture / smoke run:
    .venv/bin/python tools/build_human_db.py --limit-files 10
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from tools._human_db_build import build_argparser, run


def main() -> None:
    args = build_argparser().parse_args()
    run(args, emit_sha_sidecar=False)


if __name__ == "__main__":
    main()

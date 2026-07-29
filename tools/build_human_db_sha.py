"""tools/build_human_db_sha.py — build_human_db + SHA-256 sidecar.

Identical to `tools/build_human_db.py` except it writes a SHA-256 sidecar
next to the finished DB (`data/human_db.sqlite.sha256`) so users can
verify downloads from the Internet Archive or any other distribution
channel.

All pipeline logic lives in `tools/_human_db_build.py`.  This wrapper
and its sibling `build_human_db.py` differ only in the `emit_sha_sidecar`
flag they pass to `run()`.

Usage
-----
    # Full build (first time, or after --rebuild):
    .venv/bin/python tools/build_human_db_sha.py \\
        --games-dir data/human_games \\
        --output data/human_db.sqlite \\
        --malom-db /path/to/Malom_Standard/Std_DD_89adjusted

    # Incremental update (only processes new/changed files):
    .venv/bin/python tools/build_human_db_sha.py --update

    # Skip Malom annotation (faster when DB is not mounted):
    .venv/bin/python tools/build_human_db_sha.py --update --no-malom

    # Force full rebuild from scratch:
    .venv/bin/python tools/build_human_db_sha.py --rebuild

    # Fixture / smoke run:
    .venv/bin/python tools/build_human_db_sha.py --limit-files 10
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from tools._human_db_build import build_argparser, run


def main() -> None:
    args = build_argparser().parse_args()
    run(args, emit_sha_sidecar=True)


if __name__ == "__main__":
    main()

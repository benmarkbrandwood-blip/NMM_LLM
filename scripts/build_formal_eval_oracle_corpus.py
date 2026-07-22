#!/usr/bin/env python3
"""Build the review-only Oracle start corpus and its board-image package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.oracle_corpus import (
    load_pinned_sanmill_book,
    validate_review_manifest,
    write_review_package,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=_ROOT / "data" / "training_paths.local.json",
        help="ignored local path registry containing sanmill_checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            _ROOT
            / "docs"
            / "experiments"
            / "dev-v4-formal-paired-eval-v1-oracle-corpus.json"
        ),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=(
            _ROOT
            / "docs"
            / "experiments"
            / "assets"
            / "dev-v4-formal-paired-eval-v1-oracle-corpus"
        ),
    )
    parser.add_argument(
        "--start-positions-output",
        type=Path,
        default=(
            _ROOT
            / "docs"
            / "experiments"
            / "dev-v4-formal-paired-eval-v1-start-positions.json"
        ),
        help="freeze-compatible JSON array paired with the audit artifact",
    )
    parser.add_argument(
        "--review-asset-directory",
        default="assets/dev-v4-formal-paired-eval-v1-oracle-corpus",
        help="POSIX path relative to the corpus JSON directory",
    )
    args = parser.parse_args()

    book, source = load_pinned_sanmill_book(args.paths_config)
    payload = write_review_package(
        book,
        source,
        output_path=args.output,
        start_positions_path=args.start_positions_output,
        asset_root=args.asset_root,
        review_asset_directory=args.review_asset_directory,
    )
    image_audit = validate_review_manifest(args.asset_root)
    result = {
        "status": payload["status"],
        "starts": len(payload["start_positions"]),
        "start_positions_sha256": payload["start_positions_sha256"],
        "corpus_identity": payload["corpus_identity"],
        "corpus_file_sha256": _sha256_file(args.output),
        "start_positions_file_sha256": _sha256_file(args.start_positions_output),
        **image_audit,
        "review_manifest_identity": payload["review_assets"]["manifest_identity"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

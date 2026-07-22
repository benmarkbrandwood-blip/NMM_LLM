#!/usr/bin/env python3
"""Build the unfrozen 64-position phase-covered review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.phase_corpus import (
    build_phase_corpus,
    load_pinned_tgf_fixture,
    write_phase_review_package,
)
from learned_ai.evaluation.training_aligned_policy import (
    load_training_aligned_policy,
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
    )
    parser.add_argument(
        "--route-bundle",
        type=Path,
        default=(
            _ROOT
            / "learned_ai"
            / "checkpoints"
            / "evaluation"
            / "dev-v4-route-aligned-preparation"
            / "candidate-route-bundle"
        ),
    )
    parser.add_argument(
        "--malom-manifest",
        type=Path,
        default=_ROOT / "data" / "manifests" / "malom-sector-corrected-v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            _ROOT
            / "docs"
            / "experiments"
            / "dev-v4-phase-covered-corpus-v1.json"
        ),
    )
    parser.add_argument(
        "--start-positions-output",
        type=Path,
        default=(
            _ROOT
            / "docs"
            / "experiments"
            / "dev-v4-phase-covered-corpus-v1-start-positions.json"
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
            / "dev-v4-phase-covered-corpus-v1"
        ),
    )
    parser.add_argument(
        "--review-asset-directory",
        default="assets/dev-v4-phase-covered-corpus-v1",
    )
    args = parser.parse_args()

    config = json.loads(args.paths_config.read_text(encoding="utf-8"))
    fixture, source = load_pinned_tgf_fixture(args.paths_config)
    with load_training_aligned_policy(
        args.route_bundle,
        human_db_path=config["human_db_path"],
        specialist_db_path=config["specialist_db_path"],
        malom_path=config["malom_db_path"],
        malom_manifest_path=args.malom_manifest,
        device="cpu",
    ) as policy:
        payload = build_phase_corpus(
            fixture,
            source,
            policy,
            review_asset_directory=args.review_asset_directory,
        )
    complete = write_phase_review_package(
        payload,
        output_path=args.output,
        start_positions_path=args.start_positions_output,
        asset_root=args.asset_root,
    )
    print(
        json.dumps(
            {
                "status": complete["status"],
                "starts": len(complete["start_positions"]),
                "start_positions_sha256": complete["start_positions_sha256"],
                "corpus_identity": complete["corpus_identity"],
                "corpus_file_sha256": _sha256_file(args.output),
                "start_positions_file_sha256": _sha256_file(
                    args.start_positions_output
                ),
                "review_manifest_identity": complete["review_assets"][
                    "manifest_identity"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

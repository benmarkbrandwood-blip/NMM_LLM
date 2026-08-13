#!/usr/bin/env python3
"""Freeze the candidate-blind late-import held-out source pool once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.retained_late_import_heldout_pool import (  # noqa: E402
    PRIOR_CORPUS_PATHS,
    build_retained_late_import_pool,
    validate_retained_late_import_pool,
)
from learned_ai.training.run_contract import canonical_json_bytes  # noqa: E402
from learned_ai.training.sanmill_referee import (  # noqa: E402
    inspect_sanmill_training_installation,
)


DEFAULT_OUTPUT = ROOT / (
    "docs/experiments/"
    "sanmill-retained-v3-v4-late-import-heldout-pool-v1.json"
)
PHASE_PLAN = ROOT / (
    "docs/experiments/"
    "sanmill-retained-v3-v4-phase-process-generalization-v1.json"
)


def _local_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{field} is absent")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        payload = json.loads(output.read_bytes())
        records = validate_retained_late_import_pool(payload)
        print(f"pool_identity={payload['pool_identity']}")
        print(f"records={len(records)}")
        print("status=existing_validated_no_rebuild")
        return 0

    paths = json.loads((ROOT / "data/training_paths.local.json").read_bytes())
    plan = json.loads(PHASE_PLAN.read_bytes())
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise RuntimeError("phase-process candidate records differ")
    specialist_paths = {
        str(candidate["candidate_id"]): (
            ROOT / str(candidate["specialist_db"]["path"]),
            str(candidate["specialist_db"]["file_sha256"]),
        )
        for candidate in candidates
    }
    installation = inspect_sanmill_training_installation(
        _local_path(
            paths.get("sanmill_training_checkout"),
            field="sanmill_training_checkout",
        )
    )
    payload = build_retained_late_import_pool(
        repository_root=ROOT,
        human_games_root=ROOT / "data/human_games",
        imported_manifest_path=ROOT / "data/human_games/imported.json",
        human_db_path=_local_path(paths.get("human_db_path"), field="human_db_path"),
        human_db_identity=str(plan["data"]["human_db_identity"]),
        specialist_paths=specialist_paths,
        installation=installation,
        prior_corpus_paths=PRIOR_CORPUS_PATHS,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
    print(f"pool_identity={payload['pool_identity']}")
    print(f"records={len(payload['records'])}")
    print("status=frozen_source_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

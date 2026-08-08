#!/usr/bin/env python3
"""Audit frozen evaluation starts against trainer-visible position databases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.human_db import HumanDB  # noqa: E402
from learned_ai.data.specialist_db import SpecialistDB  # noqa: E402
from learned_ai.evaluation.heldout_exposure import (  # noqa: E402
    HeldoutExposureError,
    build_exposure_audit,
    validate_executable_corpus,
)
from learned_ai.training.generalist_preflight import (  # noqa: E402
    _probe_human_db,
    _probe_specialist_db,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--human-db", required=True)
    parser.add_argument("--specialist-db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-file-sha256", required=True)
    parser.add_argument("--corpus-identity", required=True)
    parser.add_argument("--records-identity", required=True)
    parser.add_argument("--human-db-identity", required=True)
    parser.add_argument("--specialist-db-identity", required=True)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    human_path = Path(args.human_db)
    specialist_path = Path(args.specialist_db)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"exposure audit already exists: {output}")
    if _sha256_file(corpus_path) != args.corpus_file_sha256:
        raise HeldoutExposureError("executable corpus file differs from the pin")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    records = validate_executable_corpus(
        corpus,
        expected_corpus_identity=args.corpus_identity,
        expected_records_identity=args.records_identity,
    )
    human_report = _probe_human_db(human_path)
    specialist_report = _probe_specialist_db(specialist_path)
    if human_report.get("error"):
        raise HeldoutExposureError(str(human_report["error"]))
    if human_report.get("identity") != args.human_db_identity:
        raise HeldoutExposureError("HumanDB identity differs from the pin")
    if human_report.get("malom_columns_policy") != "masked_historical_labels":
        raise HeldoutExposureError("HumanDB label policy differs from training")
    if specialist_report.get("error"):
        raise HeldoutExposureError(str(specialist_report["error"]))
    if specialist_report.get("content_sha256") != args.specialist_db_identity:
        raise HeldoutExposureError("SpecialistDB identity differs from the pin")
    if specialist_report.get("label_version") != "sector-corrected-v1":
        raise HeldoutExposureError("SpecialistDB label version is untrusted")

    human_db = HumanDB(human_path, read_only=True)
    specialist_db = SpecialistDB(specialist_path, read_only=True)
    try:
        specialist_db.require_trusted_malom_labels()
        audit = build_exposure_audit(
            records,
            human_db=human_db,
            specialist_db=specialist_db,
            corpus_identity=args.corpus_identity,
            records_identity=args.records_identity,
            human_db_identity=args.human_db_identity,
            specialist_db_identity=args.specialist_db_identity,
        )
    finally:
        specialist_db.close()
        human_db.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(audit["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

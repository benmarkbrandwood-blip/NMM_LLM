"""Build a candidate-blind, coverage-positive SpecialistDB audit corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.data.specialist_db import SpecialistDB  # noqa: E402
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402
from learned_ai.validation.specialist_db_coverage_corpus import (  # noqa: E402
    build_empirical_coverage_corpus,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _git_commit() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise RuntimeError("tracked worktree must be clean for corpus evidence")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _sidecars(path: Path) -> list[str]:
    return [
        candidate.name
        for suffix in ("-wal", "-shm", "-journal")
        if (candidate := path.with_name(path.name + suffix)).exists()
    ]


def _quick_check(path: Path) -> None:
    uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        result = connection.execute("PRAGMA quick_check").fetchone()
        if result is None or result[0] != "ok":
            raise RuntimeError("SpecialistDB quick_check did not return ok")
    finally:
        connection.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--specialist-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-specialist-db-sha256", required=True)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--minimum-selected-states", type=int, default=64)
    parser.add_argument("--minimum-empirical-actions", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commit = _git_commit()
    source_path = _resolve(args.source_corpus)
    specialist_path = _resolve(args.specialist_db)
    output_path = _resolve(args.output)
    if not source_path.is_file() or not specialist_path.is_file():
        raise RuntimeError("source corpus and SpecialistDB must exist")
    if output_path.exists():
        raise RuntimeError("coverage corpus output already exists")
    source_sha = _sha256(source_path)
    specialist_sha_before = _sha256(specialist_path)
    if source_sha != args.expected_source_sha256.lower():
        raise RuntimeError("source corpus identity differs from the frozen contract")
    if specialist_sha_before != args.expected_specialist_db_sha256.lower():
        raise RuntimeError("SpecialistDB identity differs from the frozen contract")
    sidecars_before = _sidecars(specialist_path)
    if sidecars_before:
        raise RuntimeError("SpecialistDB has SQLite sidecars before corpus build")
    _quick_check(specialist_path)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = source.get("corpus", {}).get("records")
    if not isinstance(records, list):
        raise RuntimeError("source corpus does not contain executable records")
    specialist = SpecialistDB(specialist_path, read_only=True)
    try:
        specialist.require_trusted_malom_labels()
        corpus = build_empirical_coverage_corpus(
            records,
            specialist,
            min_samples=args.min_samples,
        )
        coverage = corpus["source_summary"]["coverage"]
        selected = len(corpus["entries"])
        empirical_actions = int(coverage.get("empirical_actions", 0))
        if selected < args.minimum_selected_states:
            raise RuntimeError("coverage corpus has too few selected states")
        if empirical_actions < args.minimum_empirical_actions:
            raise RuntimeError("coverage corpus has too few empirical action hits")
        report_core = {
            "schema_version": "nmm.specialist-db-coverage-corpus.v1",
            "corpus_id": "specialist-db-policy-mechanism-placement-coverage-v2",
            "status": "frozen",
            "scope": {
                "candidate_loaded": False,
                "no_model_or_checkpoint_read": True,
                "no_database_writes": True,
                "placement_only": True,
                "development_evidence_not_held_out": True,
                "no_strength_or_promotion_claim": True,
            },
            "identities": {
                "git_commit": commit,
                "builder": _relative(Path(__file__)),
                "builder_sha256": _sha256(Path(__file__)),
                "source_corpus": _relative(source_path),
                "source_corpus_sha256": source_sha,
                "source_records_identity": source.get("corpus", {}).get(
                    "records_identity"
                ),
                "specialist_db": _relative(specialist_path),
                "specialist_db_sha256_before": specialist_sha_before,
                "specialist_db_label_version": specialist.malom_label_version,
                "specialist_db_sidecars_before": sidecars_before,
            },
            "sufficiency_gate": {
                "minimum_selected_states": args.minimum_selected_states,
                "minimum_empirical_actions": args.minimum_empirical_actions,
                "observed_selected_states": selected,
                "observed_empirical_actions": empirical_actions,
                "passed": True,
            },
            **corpus,
        }
    finally:
        specialist.close()

    specialist_sha_after = _sha256(specialist_path)
    sidecars_after = _sidecars(specialist_path)
    if specialist_sha_after != specialist_sha_before:
        raise RuntimeError("SpecialistDB changed during corpus build")
    if sidecars_after:
        raise RuntimeError("SpecialistDB sidecars appeared during corpus build")
    report_core["identities"].update(
        {
            "specialist_db_sha256_after": specialist_sha_after,
            "specialist_db_sidecars_after": sidecars_after,
        }
    )
    report = dict(report_core)
    report["evidence_id"] = canonical_sha256(report_core)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError("temporary coverage output already exists")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output_path)
    print(f"report={_relative(output_path)}")
    print(f"sha256={_sha256(output_path)}")
    print(f"evidence_id={report['evidence_id']}")
    print(f"selected_states={len(report['entries'])}")
    print(
        f"empirical_actions={report['source_summary']['coverage']['empirical_actions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

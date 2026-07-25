#!/usr/bin/env python3
"""Audit book and StrictSteps prefix diversity without playing any game."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.oracle_corpus import ring16_canonical_fen
from learned_ai.evaluation.phase_corpus import project_tgf_fen
from learned_ai.evaluation.sanmill_book_paths import (
    BookPathCorpus,
    load_book_path_corpus,
)
from learned_ai.evaluation.sanmill_data_query import (
    SanmillDataQuerySession,
    portable_source_identity,
)
from learned_ai.evaluation.sanmill_prefix import (
    PairedPrefix,
    PrefixSourceSpec,
    generate_paired_prefix,
)
from learned_ai.evaluation.sanmill_uci import (
    SanmillInstallation,
    inspect_sanmill_installation,
)
from learned_ai.training.run_contract import canonical_sha256


_SCHEMA = "nmm.sanmill-prefix-diversity-audit.v1"
_EXPERIMENT_ID = "perfect-prefix-diversity-probe-v1"


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Git inspection failed: {detail}")
    return result.stdout.strip()


def _clean_commit() -> str:
    top = Path(_git("rev-parse", "--show-toplevel")).resolve()
    if top != _ROOT.resolve():
        raise RuntimeError(f"unexpected Git top-level: {top}")
    dirty = _git("status", "--short", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "prefix-diversity audit requires a clean Git worktree:\n"
            f"{dirty}"
        )
    return _git("rev-parse", "HEAD")


def _strict_paths(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot read the local path registry") from exc
    if not isinstance(value, dict):
        raise RuntimeError("local path registry must be an object")
    malom_path = value.get("malom_db_path")
    if not isinstance(malom_path, str) or not malom_path:
        raise RuntimeError("local path registry lacks malom_db_path")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ring16(tgf_fen: str) -> str:
    projected = project_tgf_fen(tgf_fen)
    if projected is None:
        raise RuntimeError("stable prefix unexpectedly projects as pending removal")
    return ring16_canonical_fen(projected.fen)


def _histogram(values: Iterable[int], *, field: str) -> list[dict[str, int]]:
    return [
        {field: value, "count": count}
        for value, count in sorted(Counter(values).items())
    ]


def book_diversity_record(corpus: BookPathCorpus) -> dict[str, Any]:
    histories = [path.action_tokens for path in corpus.paths]
    final_fens = [path.final_fen for path in corpus.paths]
    exact_counts = Counter(final_fens)
    orbit_by_fen = {fen: _ring16(fen) for fen in exact_counts}
    orbit_counts = Counter(orbit_by_fen[fen] for fen in final_fens)
    rank_counts = Counter(
        step.candidate_source_rank
        for path in corpus.paths
        for step in path.steps
    )
    compound_counts = Counter(
        sum(step.compound_turn for step in path.steps)
        for path in corpus.paths
    )
    return {
        "corpus_identity": corpus.corpus_identity,
        "complete_history_count": len(histories),
        "unique_history_count": len(set(histories)),
        "unique_exact_final_fen_count": len(exact_counts),
        "unique_ring16_final_orbit_count": len(orbit_counts),
        "exact_fen_history_multiplicity": _histogram(
            exact_counts.values(),
            field="histories_per_exact_fen",
        ),
        "ring16_history_multiplicity": _histogram(
            orbit_counts.values(),
            field="histories_per_ring16_orbit",
        ),
        "candidate_source_rank_histogram": [
            {"rank": rank, "count": count}
            for rank, count in sorted(rank_counts.items())
        ],
        "compound_turns_per_history": [
            {"compound_turns": count, "history_count": histories_at_count}
            for count, histories_at_count in sorted(compound_counts.items())
        ],
        "ring16_orbit_set_sha256": canonical_sha256(sorted(orbit_counts)),
    }


def _perfect_pass(
    *,
    installation: SanmillInstallation,
    database_path: Path,
    samples: int,
    seed: int,
    pass_index: int,
) -> tuple[dict[str, Any], list[PairedPrefix]]:
    with SanmillDataQuerySession(installation) as session:
        root = session.query_perfect_db(
            (),
            database_path=database_path,
            request_id=f"perfect-diversity-root-{pass_index}",
            cache_sectors=8,
        )
        source_identity = portable_source_identity(
            root,
            path_lookup_key="malom_db_path",
        )
        spec = PrefixSourceSpec(
            kind="perfect_db",
            candidate_policy="uniform_candidate",
            database_path=database_path,
            path_lookup_key="malom_db_path",
            expected_identity_sha256=source_identity["identity_sha256"],
            cache_sectors=8,
        )
        prefixes = [
            generate_paired_prefix(
                session,
                installation,
                experiment_id=_EXPERIMENT_ID,
                pair_id=f"pair-{index:03d}",
                seed=seed,
                source_spec=spec,
            )
            for index in range(samples)
        ]
    return source_identity, prefixes


def perfect_diversity_record(
    prefixes: list[PairedPrefix],
    *,
    book_orbits: set[str],
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    records = [prefix.to_dict() for prefix in prefixes]
    histories = [prefix.action_tokens for prefix in prefixes]
    final_fens = [prefix.final_fen for prefix in prefixes]
    orbits = [_ring16(fen) for fen in final_fens]
    orbit_counts = Counter(orbits)
    return {
        "source_identity": source_identity,
        "sample_count": len(prefixes),
        "unique_history_count": len(set(histories)),
        "unique_exact_final_fen_count": len(set(final_fens)),
        "unique_ring16_final_orbit_count": len(orbit_counts),
        "maximum_ring16_orbit_multiplicity": max(orbit_counts.values()),
        "book_ring16_overlap_count": len(set(orbits) & book_orbits),
        "record_set_sha256": canonical_sha256(records),
        "prefix_identity_set_sha256": canonical_sha256(
            [prefix.prefix_identity for prefix in prefixes]
        ),
        "ring16_orbit_set_sha256": canonical_sha256(sorted(orbit_counts)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=_ROOT / "data" / "training_paths.local.json",
    )
    parser.add_argument(
        "--book-corpus",
        type=Path,
        default=(
            _ROOT
            / "docs"
            / "experiments"
            / "sanmill-book-path-corpus-v1.json"
        ),
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    generator_commit = _clean_commit()
    paths = _strict_paths(args.paths_config)
    installation = inspect_sanmill_installation(args.paths_config)
    corpus = load_book_path_corpus(args.book_corpus)
    book_record = book_diversity_record(corpus)
    book_orbits = {
        _ring16(path.final_fen)
        for path in corpus.paths
    }
    database_path = Path(paths["malom_db_path"]).resolve()

    first_source, first = _perfect_pass(
        installation=installation,
        database_path=database_path,
        samples=args.samples,
        seed=args.seed,
        pass_index=0,
    )
    second_source, second = _perfect_pass(
        installation=installation,
        database_path=database_path,
        samples=args.samples,
        seed=args.seed,
        pass_index=1,
    )
    first_records = [prefix.to_dict() for prefix in first]
    second_records = [prefix.to_dict() for prefix in second]
    if first_source != second_source or first_records != second_records:
        raise RuntimeError("fresh Perfect DB prefix processes disagree")

    body = {
        "schema_version": _SCHEMA,
        "status": "source-only-pre-result-audit",
        "policy_frozen": False,
        "candidate_loaded": False,
        "games_played": 0,
        "generator": {
            "nmm_llm_commit": generator_commit,
            "algorithm": "ring16-book-vs-perfect-prefix-diversity-v1",
        },
        "inputs": {
            "book_corpus_file_sha256": _sha256_file(args.book_corpus),
            "book_corpus_identity": corpus.corpus_identity,
            "sanmill": installation.portable_record(),
            "perfect_probe": {
                "experiment_id": _EXPERIMENT_ID,
                "pair_ids": f"pair-000..pair-{args.samples - 1:03d}",
                "seed": args.seed,
                "candidate_policy": "uniform_candidate",
                "cache_sectors": 8,
                "logical_plies": 8,
                "fresh_processes": 2,
            },
        },
        "book": book_record,
        "perfect": perfect_diversity_record(
            first,
            book_orbits=book_orbits,
            source_identity=first_source,
        ),
    }
    payload = {**body, "audit_identity": canonical_sha256(body)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

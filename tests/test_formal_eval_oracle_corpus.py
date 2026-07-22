from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
import learned_ai.evaluation.oracle_corpus as oracle_corpus
from learned_ai.evaluation.oracle_corpus import (
    EXPECTED_INVALID_ORACLE_MOVE,
    load_pinned_sanmill_book,
    project_oracle_source,
    ring16_canonical_fen,
    validate_corpus_artifact,
    validate_review_manifest,
)


_ROOT = Path(__file__).resolve().parents[1]
_CORPUS = (
    _ROOT
    / "docs"
    / "experiments"
    / "dev-v4-formal-paired-eval-v1-oracle-corpus.json"
)
_ASSETS = (
    _ROOT
    / "docs"
    / "experiments"
    / "assets"
    / "dev-v4-formal-paired-eval-v1-oracle-corpus"
)
_STARTS = (
    _ROOT
    / "docs"
    / "experiments"
    / "dev-v4-formal-paired-eval-v1-start-positions.json"
)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def test_pinned_book_loads_after_the_reference_checkout_advances(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkout = tmp_path / "sanmill"
    asset = checkout.joinpath(*oracle_corpus.SOURCE_ASSET.parts)
    asset.parent.mkdir(parents=True)
    reviewed_book = {"schemaVersion": 1, "variant": "nmm", "symmetry": "ring16"}
    reviewed_bytes = (json.dumps(reviewed_book, sort_keys=True) + "\n").encode("utf-8")
    asset.write_bytes(reviewed_bytes)
    _git(checkout, "init")
    _git(checkout, "config", "user.name", "Oracle Test")
    _git(checkout, "config", "user.email", "oracle-test@example.invalid")
    _git(checkout, "add", oracle_corpus.SOURCE_ASSET.as_posix())
    _git(checkout, "commit", "-m", "Add reviewed opening book")
    reviewed_commit = _git(checkout, "rev-parse", "HEAD")

    asset.write_text(
        json.dumps({**reviewed_book, "variant": "changed"}) + "\n",
        encoding="utf-8",
    )
    _git(checkout, "add", oracle_corpus.SOURCE_ASSET.as_posix())
    _git(checkout, "commit", "-m", "Advance reference checkout")
    assert _git(checkout, "rev-parse", "HEAD") != reviewed_commit

    paths_config = tmp_path / "training_paths.local.json"
    paths_config.write_text(
        json.dumps({"sanmill_checkout": str(checkout)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(oracle_corpus, "EXPECTED_SANMILL_COMMIT", reviewed_commit)
    monkeypatch.setattr(
        oracle_corpus,
        "EXPECTED_SOURCE_SHA256",
        hashlib.sha256(reviewed_bytes).hexdigest(),
    )

    book, source = load_pinned_sanmill_book(paths_config)

    assert book == reviewed_book
    assert source["sanmill_commit"] == reviewed_commit


def test_pending_removal_is_not_treated_as_a_direct_start() -> None:
    projection = project_oracle_source(
        "********/@*@@OOOO/******** w p r 4 5 3 6 1 0 "
        "-1 -1 -1 -1 0 0 4 ids:nodes",
        ["xf2"],
    )

    assert projection.action == "r"
    assert projection.direct_fen is None
    assert projection.stable_fen == "........WB.B.WWW........|B|4|3"
    assert projection.legal_oracle_moves == ("xf2",)
    assert projection.illegal_oracle_moves == ()


def test_inner_outer_ring_swap_has_the_same_ring16_identity() -> None:
    outer = "W.......................|W|1|0"
    inner = "................W.......|W|1|0"

    assert ring16_canonical_fen(outer) == ring16_canonical_fen(inner)


def test_known_illegal_source_recommendation_is_audited_not_selected() -> None:
    projection = project_oracle_source(
        "****OO*O/O@O*@OO@/@@**@*O* b p p 8 1 6 2 0 0 "
        "-1 -1 -1 -1 0 0 8 ids:nodes",
        ["c3"],
    )

    assert projection.raw_key_sha256 == EXPECTED_INVALID_ORACLE_MOVE["raw_key_sha256"]
    assert projection.direct_fen == ".BB..B.WBWBW.BWWW....WW.|B|8|7"
    assert projection.legal_oracle_moves == ()
    assert projection.illegal_oracle_moves == ("c3",)


def test_generated_oracle_corpus_reproduces_all_projection_evidence() -> None:
    payload = json.loads(_CORPUS.read_text(encoding="utf-8"))
    starts = json.loads(_STARTS.read_text(encoding="utf-8"))
    boards = [BoardState.from_fen_string(fen) for fen in starts]

    audit = validate_corpus_artifact(payload)

    assert audit == {
        "starts": 107,
        "ring16_orbits": 107,
        "direct_sources": 108,
        "pending_removals": 2,
    }
    assert payload["status"] == "generated_for_owner_review"
    assert starts == payload["start_positions"]
    assert hashlib.sha256(_STARTS.read_bytes()).hexdigest() == payload[
        "start_positions_artifact"
    ]["file_sha256"]
    manifest_path = _ASSETS / payload["review_assets"]["manifest"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == payload[
        "review_assets"
    ]["manifest_file_sha256"]
    assert payload["automated_audit"]["phase_counts"] == {
        "placement": 107,
        "movement": 0,
        "flying": 0,
    }
    assert payload["automated_audit"]["oracle_move_legality"][
        "invalid_records"
    ] == [EXPECTED_INVALID_ORACLE_MOVE]
    assert all(not is_terminal(board)[0] for board in boards)
    assert all(get_all_legal_moves(board) for board in boards)


def test_generated_review_images_match_their_manifest() -> None:
    assert validate_review_manifest(_ASSETS) == {
        "individual_images": 107,
        "contact_sheets": 9,
    }

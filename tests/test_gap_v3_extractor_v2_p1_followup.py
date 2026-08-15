"""tests/test_gap_v3_extractor_v2_p1_followup.py — Codex 2026-08-14 P1 + P2 fixes.

Regressions for the second Codex review of fdd5a97:
  P1(1) Teacher probs_strict + validation — no silent uniform fallback.
  P1(2) Full-run manifest set equality — deleted ledger file detected.
  P1(3) Teacher lineage verification.
  P1(4) Non-production / smoke / sub-floor outputs are non-ready in provenance
        AND the Stage E trainer refuses to consume them without an explicit
        --allow-non-production-dataset override.
  P1(5) Empirical P_h drops illegal-notation counts from the denominator.
  P2(1) Per-(band, phase) state discard counts recorded in provenance.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load(mod_name: str, rel: str):
    spec = importlib.util.spec_from_file_location(mod_name, _ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def extractor():
    return _load("extract_gap_v3_dataset_v2", "tools/extract_gap_v3_dataset_v2.py")


@pytest.fixture(scope="module")
def trainer():
    return _load("train_gap_net_v3", "tools/train_gap_net_v3.py")


@pytest.fixture(scope="module")
def ledger_builder():
    return _load("build_gap_v3_session_ledger", "tools/build_gap_v3_session_ledger.py")


def _initial_fen() -> str:
    from game.board import BoardState
    return BoardState.new_game().to_fen_string()


def _write_game(dir_path: Path, filename: str, session_id: str) -> None:
    rec = {
        "session_id": session_id,
        "moves": [{"board_fen_before": _initial_fen(),
                   "to": "a7", "color": "white"}],
    }
    (dir_path / filename).write_text(json.dumps(rec) + "\n", encoding="utf-8")


# ── P1(5) empirical P_h: illegal-notation counts dropped ─────────────────────

def test_empirical_ph_drops_illegal_notations(extractor):
    """Codex P1(5): denominator must be legal-only.  {legal: 25, illegal: 25}
    at min_support=25 previously produced a legal-move vector summing to 0.5;
    now must return a valid P_h summing to 1 plus n_illegal_events=25."""
    legal_moves = [{"from": None, "to": "a7"}, {"from": None, "to": "d7"}]
    counts = {"a7": 20, "d7": 5, "z9": 25}   # z9 is illegal
    ph, illegal = extractor._empirical_ph(counts, legal_moves, min_support=25)
    assert ph is not None
    assert abs(float(ph.sum()) - 1.0) < 1e-6, f"ph sum {ph.sum()} != 1"
    # 20/25 + 5/25 = 1.0
    assert abs(float(ph[0]) - 20 / 25) < 1e-6
    assert abs(float(ph[1]) - 5 / 25) < 1e-6
    assert illegal == 25


def test_empirical_ph_returns_none_when_legal_below_min(extractor):
    """Illegal counts do NOT prop up min_support."""
    legal_moves = [{"from": None, "to": "a7"}]
    counts = {"a7": 10, "z9": 100}   # legal_total = 10 < min_support 25
    ph, illegal = extractor._empirical_ph(counts, legal_moves, min_support=25)
    assert ph is None
    assert illegal == 100


def test_empirical_ph_no_illegal_returns_zero_count(extractor):
    legal_moves = [{"from": None, "to": "a7"}, {"from": None, "to": "d7"}]
    counts = {"a7": 20, "d7": 30}
    ph, illegal = extractor._empirical_ph(counts, legal_moves, min_support=25)
    assert ph is not None
    assert illegal == 0
    assert abs(float(ph.sum()) - 1.0) < 1e-6


# ── P1(2) full-run manifest set equality ────────────────────────────────────

def test_manifest_verify_detects_deleted_file(extractor, ledger_builder, tmp_path):
    """Codex P1(2): file present in ledger manifest but missing from disk
    must fail-closed on a full run."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "a.jsonl", "sa")
    _write_game(games_dir, "b.jsonl", "sb")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    # Delete one file after ledger build
    (games_dir / "b.jsonl").unlink()

    with pytest.raises(RuntimeError, match="missing from disk"):
        extractor._verify_files_match_manifest(
            games_dir, info["ledger_file_shas"], limit_files=None,
        )


def test_manifest_verify_detects_extra_file(extractor, ledger_builder, tmp_path):
    """Codex P1(2): file present on disk but not in ledger manifest is fatal
    on a full run."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "a.jsonl", "sa")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    _write_game(games_dir, "extra.jsonl", "s_extra")

    with pytest.raises(RuntimeError, match="extra on disk"):
        extractor._verify_files_match_manifest(
            games_dir, info["ledger_file_shas"], limit_files=None,
        )


def test_manifest_verify_smoke_run_tolerates_subset(extractor, ledger_builder, tmp_path):
    """Under smoke run (limit_files set), deleted files are tolerated because
    the smoke may only scan a subset of the manifest."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "a.jsonl", "sa")
    _write_game(games_dir, "b.jsonl", "sb")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    (games_dir / "b.jsonl").unlink()
    # Must not raise
    extractor._verify_files_match_manifest(
        games_dir, info["ledger_file_shas"], limit_files=5,
    )


# ── Codex 2026-08-15 P2: probs_strict abstain regression ────────────────────

def test_compute_teacher_ph_abstains_on_degenerate_teacher(extractor):
    """Codex 2026-08-15 P2: prior code silently returned a uniform teacher
    P_h when `advisor.probs()` fell back on non-finite scores.  probs_strict()
    now raises, and the extractor's emit loop must catch that and mark the
    row abstained with reason prefix `teacher_probs_failed:`.

    Proves (a) probs() would silently return uniform on the same input,
    (b) probs_strict() raises, and (c) _compute_teacher_ph — the seam the
    extractor emit loop uses — returns (None, teacher_probs_failed:…)."""
    from ai.human_move_policy_advisor import HumanMovePolicyAdvisor    # noqa: WPS433
    from game.board import BoardState                                  # noqa: WPS433

    class DegenerateAdvisor(HumanMovePolicyAdvisor):
        """Simulates a teacher whose forward pass yields non-finite logits.
        probs() falls back to uniform (silent); probs_strict() raises."""
        def __init__(self):                                             # noqa: D401
            self.temperature = 1.0
            self.board_feature_dim = 79

        def _score_batch(self, x):
            return np.full((x.shape[0],), np.nan, dtype=np.float32)

        def _band_row(self, elo_band, n):
            return np.zeros((n, 3), dtype=np.float32)

        def _successor_features(self, board, legal_moves):
            return np.zeros(
                (len(legal_moves), self.board_feature_dim), dtype=np.float32,
            )

    advisor = DegenerateAdvisor()
    board = BoardState.new_game()
    legal_moves = [{"from": None, "to": "a7"},
                   {"from": None, "to": "d7"}]

    # (a) probs() silently degrades — this is the exact bug we hardened against.
    ph_soft = advisor.probs(board, legal_moves, elo_band="middle")
    assert np.allclose(ph_soft, [0.5, 0.5]), \
        f"probs() must return uniform on NaN scores, got {ph_soft}"

    # (b) probs_strict() raises rather than silently returning uniform.
    with pytest.raises(ValueError):
        advisor.probs_strict(board, legal_moves, elo_band="middle")

    # (c) The extractor emit-loop seam abstains the row and surfaces the
    #     teacher_probs_failed: reason exactly as written to abstained.jsonl.
    ph_model, reason = extractor._compute_teacher_ph(
        advisor, board, legal_moves, "middle",
    )
    assert ph_model is None, "abstained row must return None ph"
    assert reason is not None
    assert reason.startswith("teacher_probs_failed:"), \
        f"reason must be teacher_probs_failed prefix, got {reason!r}"


def test_compute_teacher_ph_happy_path_returns_valid_ph(extractor):
    """Codex 2026-08-15 P2: a well-formed teacher returns (ph, None)."""
    from ai.human_move_policy_advisor import HumanMovePolicyAdvisor    # noqa: WPS433
    from game.board import BoardState                                  # noqa: WPS433

    class UniformAdvisor(HumanMovePolicyAdvisor):
        """Returns constant zero logits — softmax yields exact uniform."""
        def __init__(self):
            self.temperature = 1.0
            self.board_feature_dim = 79

        def _score_batch(self, x):
            return np.zeros((x.shape[0],), dtype=np.float32)

        def _band_row(self, elo_band, n):
            return np.zeros((n, 3), dtype=np.float32)

        def _successor_features(self, board, legal_moves):
            return np.zeros(
                (len(legal_moves), self.board_feature_dim), dtype=np.float32,
            )

    board = BoardState.new_game()
    legal_moves = [{"from": None, "to": "a7"},
                   {"from": None, "to": "d7"}]

    ph_model, reason = extractor._compute_teacher_ph(
        UniformAdvisor(), board, legal_moves, "middle",
    )
    assert reason is None
    assert ph_model is not None
    assert ph_model.shape == (2,)
    assert abs(float(ph_model.sum()) - 1.0) < 1e-5


# ── Codex 2026-08-15 P1: between-pass file-deletion detection ───────────────

def test_iter_events_detects_between_pass_deletion(extractor, ledger_builder, tmp_path):
    """Codex 2026-08-15 P1: `_verify_files_match_manifest` runs once before
    pass 1.  If a ledger-listed file is deleted after that precheck but before
    pass 2 starts, the disk-glob-driven iteration silently completes with fewer
    events while provenance still binds the full manifest.

    Fix: `_iter_jsonl_events` now iterates the frozen manifest keys, not a
    fresh disk glob.  Missing-on-disk becomes fatal via FileNotFoundError →
    RuntimeError with a between-pass-deletion hint."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "a.jsonl", "sa")
    _write_game(games_dir, "b.jsonl", "sb")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    # Simulate between-pass deletion: precheck already ran successfully, and
    # now we're about to start pass 2 …
    (games_dir / "b.jsonl").unlink()

    with pytest.raises(RuntimeError, match="missing from disk"):
        list(extractor._iter_jsonl_events(
            games_dir, info["session_meta"],
            ledger_file_shas=info["ledger_file_shas"],
        ))


def test_iter_events_manifest_driven_ignores_late_added_disk_file(
    extractor, ledger_builder, tmp_path,
):
    """Codex 2026-08-15 P1 corollary: iteration is driven by the frozen
    manifest, so a file added on disk after the ledger was built is a full-run
    fatal (extra-on-disk detection preserved) rather than being silently
    scanned."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "a.jsonl", "sa")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    _write_game(games_dir, "late.jsonl", "s_late")

    with pytest.raises(RuntimeError, match="not in ledger file manifest"):
        list(extractor._iter_jsonl_events(
            games_dir, info["session_meta"],
            ledger_file_shas=info["ledger_file_shas"],
        ))


# ── P1(3) teacher lineage ───────────────────────────────────────────────────

def _write_synthetic_teacher_npz(
    tmp_path: Path,
    filename: str = "teacher.npz",
    dataset_split_scheme: str | None = "session_ledger_strict_single_tier",
    session_ledger_sha256: str | None = "0" * 64,
    session_ledger_files_manifest_sha256: str | None = "b" * 64,
) -> Path:
    """Write a fake teacher .npz with the specified provenance fields.  Any
    None field is omitted from provenance to simulate missing-provenance cases."""
    prov: dict = {}
    if dataset_split_scheme is not None:
        prov["dataset_split_scheme"] = dataset_split_scheme
    if session_ledger_sha256 is not None:
        prov["session_ledger_sha256"] = session_ledger_sha256
    if session_ledger_files_manifest_sha256 is not None:
        prov["session_ledger_files_manifest_sha256"] = session_ledger_files_manifest_sha256

    path = tmp_path / filename
    np.savez(str(path), provenance_json=np.array(json.dumps(prov), dtype=object))
    return path


def test_teacher_lineage_rejects_v2_scheme(extractor, ledger_builder, tmp_path):
    """A teacher without dataset_split_scheme=session_ledger_strict_single_tier
    is refused (covers the v2 candidate + any state-key-split teacher)."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "a.jsonl", "sa")
    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    teacher = _write_synthetic_teacher_npz(
        tmp_path, dataset_split_scheme="state_key_three_way",
    )
    with pytest.raises(RuntimeError, match="split_scheme"):
        extractor._verify_teacher_lineage(teacher, ledger_path, info)


def test_teacher_lineage_rejects_missing_provenance(extractor, ledger_builder, tmp_path):
    """A teacher with no provenance_json at all is refused."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "a.jsonl", "sa")
    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    teacher_path = tmp_path / "teacher.npz"
    np.savez(str(teacher_path), weights=np.zeros(1))   # no provenance_json
    with pytest.raises(RuntimeError, match="no provenance_json"):
        extractor._verify_teacher_lineage(teacher_path, ledger_path, info)


def test_teacher_lineage_rejects_ledger_sha_mismatch(extractor, ledger_builder, tmp_path):
    """Teacher trained on a different ledger SHA is refused."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "a.jsonl", "sa")
    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    teacher = _write_synthetic_teacher_npz(
        tmp_path, session_ledger_sha256="deadbeef" * 8,
    )
    with pytest.raises(RuntimeError, match="ledger sha"):
        extractor._verify_teacher_lineage(teacher, ledger_path, info)


def test_teacher_lineage_rejects_manifest_mismatch(extractor, ledger_builder, tmp_path):
    """Teacher trained on ledger with different files_manifest_sha256 is refused."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "a.jsonl", "sa")
    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    actual_ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    teacher = _write_synthetic_teacher_npz(
        tmp_path,
        session_ledger_sha256=actual_ledger_sha,
        session_ledger_files_manifest_sha256="c" * 64,   # wrong manifest
    )
    with pytest.raises(RuntimeError, match="files_manifest_sha256"):
        extractor._verify_teacher_lineage(teacher, ledger_path, info)


def test_teacher_lineage_happy_path(extractor, ledger_builder, tmp_path):
    """Matching teacher provenance passes verification."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "a.jsonl", "sa")
    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    actual_ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    manifest_sha = info["provenance"]["files_manifest_sha256"]
    teacher = _write_synthetic_teacher_npz(
        tmp_path,
        session_ledger_sha256=actual_ledger_sha,
        session_ledger_files_manifest_sha256=manifest_sha,
    )
    prov = extractor._verify_teacher_lineage(teacher, ledger_path, info)
    assert prov["dataset_split_scheme"] == "session_ledger_strict_single_tier"


# ── P1(4) production_ready flag + trainer refusal ───────────────────────────

def test_provenance_production_ready_true_on_clean_run(extractor):
    prov = extractor._make_provenance(
        counters={"n_emitted": 2_000_000, "n_hybrid_rows": 0,
                  "n_model_only_rows": 2_000_000},
        disposition={},
        ledger_info={"provenance": {"files_manifest_sha256": "b" * 64}},
        pass1_stats={},
        teacher_net=Path("/dev/null"), malom_db_dir="",
        games_dir=Path("/dev/null"), out_dir=Path("/dev/null"),
        min_empirical_support=25, temperature=1.0,
        coverage_floor_rows=1_275_400, limit_files=None,
        gate_status="ok", elapsed_wall=1.0,
        require_ready=True, strict=True, allow_partial_ledger=False,
    )
    assert prov["production_ready"] is True
    assert prov["non_ready_reasons"] == []


@pytest.mark.parametrize("flags,expected_reason_fragment", [
    ({"limit_files": 5}, "limit_files"),
    ({"strict": False}, "strict=False"),
    ({"allow_partial_ledger": True}, "allow_partial_ledger"),
    ({"require_ready": False}, "require_ready=False"),
    ({"coverage_floor_rows": 5_000_000}, "coverage_floor not met"),  # emitted 2M < 5M
])
def test_provenance_non_ready_reasons_flagged(extractor, flags, expected_reason_fragment):
    base = dict(
        counters={"n_emitted": 2_000_000, "n_hybrid_rows": 0,
                  "n_model_only_rows": 2_000_000},
        disposition={},
        ledger_info={"provenance": {"files_manifest_sha256": "b" * 64}},
        pass1_stats={},
        teacher_net=Path("/dev/null"), malom_db_dir="",
        games_dir=Path("/dev/null"), out_dir=Path("/dev/null"),
        min_empirical_support=25, temperature=1.0,
        coverage_floor_rows=1_275_400, limit_files=None,
        gate_status="ok", elapsed_wall=1.0,
        require_ready=True, strict=True, allow_partial_ledger=False,
    )
    base.update(flags)
    prov = extractor._make_provenance(**base)
    assert prov["production_ready"] is False, f"expected non-ready under {flags}"
    assert any(expected_reason_fragment in r for r in prov["non_ready_reasons"]), \
        f"reason fragment {expected_reason_fragment!r} not in {prov['non_ready_reasons']}"


def _write_synth_dataset_metadata(dir_path: Path, provenance: dict) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    meta_path = dir_path / "metadata.npz"
    np.savez(
        str(meta_path),
        state_keys=np.array(["sk"], dtype=object),
        provenance=np.array(json.dumps(provenance), dtype=object),
    )
    return meta_path


def test_trainer_refuses_non_production_dataset(trainer, tmp_path):
    """Codex P1(4): Stage E hard-rejects non-production datasets by default."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready": False,
        "non_ready_reasons": ["limit_files=5"],
    })
    with pytest.raises(SystemExit, match="NOT production_ready"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_trainer_allow_non_production_overrides(trainer, tmp_path):
    """--allow-non-production-dataset lets the run proceed (with warning).

    Codex 2026-08-15 P1: return dict carries structured taint fields."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready": False,
        "non_ready_reasons": ["limit_files=5"],
    })
    taint = trainer._verify_dataset_production_ready(
        dataset_dir, allow_non_production=True,
    )
    assert taint["dataset_production_ready"] is False
    assert taint["non_production_override"] is True
    assert taint["promotion_eligible"] is False
    assert "limit_files=5" in taint["dataset_non_ready_reasons"]


def test_trainer_accepts_production_ready_dataset(trainer, tmp_path):
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready": True,
        "non_ready_reasons": [],
    })
    taint = trainer._verify_dataset_production_ready(dataset_dir)
    assert taint["dataset_production_ready"] is True
    assert taint["non_production_override"] is False
    assert taint["promotion_eligible"] is True
    assert taint["dataset_non_ready_reasons"] == []


def test_trainer_refuses_missing_provenance(trainer, tmp_path):
    """metadata.npz without a provenance field → refuse."""
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    np.savez(
        str(dataset_dir / "metadata.npz"),
        state_keys=np.array(["sk"], dtype=object),   # no provenance
    )
    with pytest.raises(SystemExit, match="no provenance"):
        trainer._verify_dataset_production_ready(dataset_dir)


# ── P2(1) per-(band, phase) state discard counts ────────────────────────────

def test_trainer_refuses_missing_metadata_file(trainer, tmp_path):
    """Trainer must refuse when metadata.npz itself is absent."""
    dataset_dir = tmp_path / "empty_ds"
    dataset_dir.mkdir()
    with pytest.raises(SystemExit, match="metadata.npz not found"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_trainer_taint_missing_provenance_under_override(trainer, tmp_path):
    """Codex 2026-08-15 P1 edge case: dataset with no provenance under
    --allow-non-production-dataset must still yield a non-empty
    dataset_non_ready_reasons list (synthetic 'no_dataset_provenance')."""
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    np.savez(
        str(dataset_dir / "metadata.npz"),
        state_keys=np.array(["sk"], dtype=object),   # no provenance field
    )
    taint = trainer._verify_dataset_production_ready(
        dataset_dir, allow_non_production=True,
    )
    assert taint["dataset_production_ready"] is False
    assert taint["non_production_override"] is True
    assert taint["promotion_eligible"] is False
    assert "no_dataset_provenance" in taint["dataset_non_ready_reasons"]


def test_trainer_saved_npz_round_trip_carries_taint(trainer, tmp_path):
    """Codex 2026-08-15 P1: the saved model NPZ must record taint fields
    (dataset_production_ready, dataset_non_ready_reasons,
    non_production_override, promotion_eligible) AND the model label must
    reflect the taint.  Round-trip: build provenance the way `main()`
    would, hand it to `_save`, reload with `np.load`, and assert taint
    survived and the model label switched to *_nonproduction."""
    model = trainer.GapNetV3()

    # Simulate main() computing taint for a non-production dataset.
    taint_nonprod = {
        "dataset_production_ready":  False,
        "dataset_non_ready_reasons": ["limit_files=5", "coverage_floor not met"],
        "non_production_override":   True,
        "promotion_eligible":        False,
        "dataset_provenance":        {"production_ready": False},
    }
    model_label = ("gap_net_v3_candidate"
                   if taint_nonprod["promotion_eligible"]
                   else "gap_net_v3_candidate_nonproduction")
    provenance = {
        "model":                     model_label,
        "dataset_production_ready":  taint_nonprod["dataset_production_ready"],
        "dataset_non_ready_reasons": taint_nonprod["dataset_non_ready_reasons"],
        "non_production_override":   taint_nonprod["non_production_override"],
        "promotion_eligible":        taint_nonprod["promotion_eligible"],
    }
    out = tmp_path / "nonprod.npz"
    trainer._save(model, out, provenance)

    z = np.load(str(out), allow_pickle=True)
    reloaded_prov = json.loads(str(z["provenance"]))
    assert reloaded_prov["model"] == "gap_net_v3_candidate_nonproduction"
    assert reloaded_prov["dataset_production_ready"] is False
    assert reloaded_prov["non_production_override"] is True
    assert reloaded_prov["promotion_eligible"] is False
    assert "limit_files=5" in reloaded_prov["dataset_non_ready_reasons"]
    assert "coverage_floor not met" in reloaded_prov["dataset_non_ready_reasons"]


def test_trainer_saved_npz_round_trip_clean_run_promotion_eligible(trainer, tmp_path):
    """Codex 2026-08-15 P1: clean production_ready dataset yields
    promotion_eligible=True and the plain gap_net_v3_candidate label."""
    model = trainer.GapNetV3()

    provenance = {
        "model":                     "gap_net_v3_candidate",
        "dataset_production_ready":  True,
        "dataset_non_ready_reasons": [],
        "non_production_override":   False,
        "promotion_eligible":        True,
    }
    out = tmp_path / "prod.npz"
    trainer._save(model, out, provenance)

    z = np.load(str(out), allow_pickle=True)
    reloaded_prov = json.loads(str(z["provenance"]))
    assert reloaded_prov["model"] == "gap_net_v3_candidate"
    assert reloaded_prov["dataset_production_ready"] is True
    assert reloaded_prov["non_production_override"] is False
    assert reloaded_prov["promotion_eligible"] is True
    assert reloaded_prov["dataset_non_ready_reasons"] == []


# ── P2(1) emitted_by_band_phase counter ─────────────────────────────────────

def test_emitted_by_band_phase_key_present_in_provenance(extractor):
    """Self-review 2026-08-14: n_emitted-per-(band, phase) surfaces via
    counters.emitted_by_band_phase in provenance so a coverage-floor halt
    also reports actual emitted counts."""
    prov = extractor._make_provenance(
        counters={"n_emitted": 0, "n_hybrid_rows": 0, "n_model_only_rows": 0,
                  "emitted_by_band_phase": {"lower|place": 100}},
        disposition={},
        ledger_info={"provenance": {"files_manifest_sha256": "b" * 64}},
        pass1_stats={},
        teacher_net=Path("/dev/null"), malom_db_dir="",
        games_dir=Path("/dev/null"), out_dir=Path("/dev/null"),
        min_empirical_support=25, temperature=1.0,
        coverage_floor_rows=1_000_000, limit_files=None,
        gate_status="halt_coverage_floor", elapsed_wall=1.0,
    )
    assert "emitted_by_band_phase" in prov["extract_counters"]
    assert prov["extract_counters"]["emitted_by_band_phase"]["lower|place"] == 100


def test_pass2_disposition_records_state_counts_by_band_phase(extractor, tmp_path):
    """Codex P2(1): disposition includes both event and state counts per (b, p)."""
    # Build session_meta with three tiers touching the same state_key
    ids = ["ss_a", "ss_b", "ss_c"]
    hashes = {sid: hashlib.sha256(sid.encode()).hexdigest() for sid in ids}
    min_id = min(ids, key=lambda s: hashes[s])
    others = [s for s in ids if s != min_id]
    session_meta = {
        min_id:    {"tier": "train", "session_hash": hashes[min_id],   "source_file": "a.jsonl"},
        others[0]: {"tier": "val",   "session_hash": hashes[others[0]], "source_file": "b.jsonl"},
        others[1]: {"tier": "test",  "session_hash": hashes[others[1]], "source_file": "c.jsonl"},
    }
    games_dir = tmp_path / "games"
    games_dir.mkdir()

    def _write(sid: str):
        rec = {"session_id": sid, "white_elo": 1200, "black_elo": 1200,
               "moves": [{"board_fen_before": _initial_fen(),
                          "to": "a7", "color": "white"}]}
        (games_dir / f"{sid}.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

    for sid in ids:
        _write(sid)

    owning, _ = extractor._scan_pass1_owning_tier(games_dir, session_meta)
    _, disp = extractor._scan_pass2_aggregate(games_dir, session_meta, owning)

    # Every event in this synthetic corpus is placement phase (initial board),
    # middle band (Elo 1200 → 1200/50*50=1200 → middle).  Owning tier is train.
    # Expected: kept 1 state, discarded 1 state (only val + test's separate sessions
    # contributed events, and state_keys land at the same key from initial board).
    assert "states_kept_by_band_phase" in disp
    assert "states_discarded_other_tier_by_band_phase" in disp
    assert "events_discarded_other_tier_by_band_phase" in disp
    # At least the placement/middle bucket must appear
    keys_kept = list(disp["states_kept_by_band_phase"].keys())
    keys_disc = list(disp["states_discarded_other_tier_by_band_phase"].keys())
    assert any("place" in k for k in keys_kept), f"no place bucket in {keys_kept}"
    assert any("place" in k for k in keys_disc), f"no place bucket in {keys_disc}"

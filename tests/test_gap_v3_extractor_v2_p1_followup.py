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


def _ready_prov(**overrides) -> dict:
    """Default valid ready provenance (matching extractor contract).

    Codex 2026-08-15 (2nd pass) P1: `_validate_dataset_provenance_contract`
    requires `production_ready`, `non_ready_reasons`, and `gate_status`
    with strict invariants.  Test helpers must produce contract-conformant
    provenance."""
    base = {
        "production_ready":  True,
        "non_ready_reasons": [],
        "gate_status":       "ok",
    }
    base.update(overrides)
    return base


def _nonready_prov(reason: str = "limit_files=5", gate_status: str = "ok",
                   **overrides) -> dict:
    """Default valid non-ready provenance (matching extractor contract)."""
    base = {
        "production_ready":  False,
        "non_ready_reasons": [reason],
        "gate_status":       gate_status,
    }
    base.update(overrides)
    return base


def test_trainer_refuses_non_production_dataset(trainer, tmp_path):
    """Codex P1(4): Stage E hard-rejects non-production datasets by default."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, _nonready_prov("limit_files=5"))
    with pytest.raises(SystemExit, match="NOT production_ready"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_trainer_allow_non_production_overrides(trainer, tmp_path):
    """--allow-non-production-dataset lets the run proceed (with warning).

    Codex 2026-08-15 P1: return dict carries structured taint fields.
    2nd pass: promotion_eligible is no longer computed here — it moved to
    `_build_saved_provenance` (must factor Stage E gate + frozen thresholds)."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, _nonready_prov("limit_files=5"))
    taint = trainer._verify_dataset_production_ready(
        dataset_dir, allow_non_production=True,
    )
    assert taint["dataset_production_ready"] is False
    assert taint["non_production_override"] is True
    assert "limit_files=5" in taint["dataset_non_ready_reasons"]
    # Contract change: promotion_eligible is no longer in the taint dict.
    assert "promotion_eligible" not in taint


def test_trainer_accepts_production_ready_dataset(trainer, tmp_path):
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, _ready_prov())
    taint = trainer._verify_dataset_production_ready(dataset_dir)
    assert taint["dataset_production_ready"] is True
    assert taint["non_production_override"] is False
    assert taint["dataset_non_ready_reasons"] == []
    assert "promotion_eligible" not in taint


# ── Codex 2026-08-15 (2nd pass) P1: strict provenance contract ──────────────

def test_strict_contract_rejects_string_true_as_production_ready(trainer, tmp_path):
    """`production_ready` must be an exact JSON boolean.  A string "true"
    (silently truthy in Python) must not be accepted — Codex 2026-08-15
    called this out explicitly."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready":  "true",           # string, not bool
        "non_ready_reasons": [],
        "gate_status":       "ok",
    })
    with pytest.raises(SystemExit, match="exact JSON boolean"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_strict_contract_rejects_int_1_as_production_ready(trainer, tmp_path):
    """`production_ready`=1 (int) is truthy in Python but not a bool — reject."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready":  1,                # int, not bool
        "non_ready_reasons": [],
        "gate_status":       "ok",
    })
    with pytest.raises(SystemExit, match="exact JSON boolean"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_strict_contract_rejects_string_false_as_production_ready(trainer, tmp_path):
    """A string "false" is a truthy non-empty string — must not be treated as
    non-ready.  It's a malformed field, not a signal.  Fail closed."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready":  "false",          # string, not bool
        "non_ready_reasons": [],
        "gate_status":       "ok",
    })
    with pytest.raises(SystemExit, match="exact JSON boolean"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_strict_contract_rejects_missing_production_ready(trainer, tmp_path):
    """`production_ready` is a required field — missing → malformed."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        # production_ready omitted
        "non_ready_reasons": [],
        "gate_status":       "ok",
    })
    with pytest.raises(SystemExit, match="missing `production_ready`"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_strict_contract_rejects_missing_gate_status(trainer, tmp_path):
    """`gate_status` is a required field — missing → malformed."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready":  True,
        "non_ready_reasons": [],
        # gate_status omitted
    })
    with pytest.raises(SystemExit, match="missing `gate_status`"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_strict_contract_rejects_ready_with_non_empty_reasons(trainer, tmp_path):
    """Invariant A: production_ready=True must have non_ready_reasons=[].
    A ready flag paired with any reason is self-inconsistent — reject even
    though the ready flag looks "good"."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready":  True,
        "non_ready_reasons": ["limit_files=5"],   # contradicts ready=True
        "gate_status":       "ok",
    })
    with pytest.raises(SystemExit, match="invariant A"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_strict_contract_rejects_ready_with_bad_gate_status(trainer, tmp_path):
    """Invariant B: production_ready=True implies gate_status='ok'.
    A ready flag paired with any halt gate is self-inconsistent."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready":  True,
        "non_ready_reasons": [],
        "gate_status":       "halt_coverage_floor",
    })
    with pytest.raises(SystemExit, match="invariant B"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_strict_contract_rejects_nonready_with_empty_reasons(trainer, tmp_path):
    """Invariant C: production_ready=False must have at least one reason.
    A non-ready flag with no explanation is malformed."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready":  False,
        "non_ready_reasons": [],                  # empty
        "gate_status":       "ok",
    })
    with pytest.raises(SystemExit, match="invariant C"):
        trainer._verify_dataset_production_ready(dataset_dir)


def test_strict_contract_reject_cannot_be_bypassed_by_override(trainer, tmp_path):
    """Malformed provenance is NOT bypassable by --allow-non-production-dataset.
    The override flag is for legitimate non-ready datasets, not broken metadata."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready":  "false",   # malformed
        "non_ready_reasons": [],
        "gate_status":       "ok",
    })
    with pytest.raises(SystemExit, match="exact JSON boolean"):
        trainer._verify_dataset_production_ready(
            dataset_dir, allow_non_production=True,
        )


def test_strict_contract_reasons_must_be_list(trainer, tmp_path):
    """non_ready_reasons must be a list, not a string."""
    dataset_dir = tmp_path / "ds"
    _write_synth_dataset_metadata(dataset_dir, {
        "production_ready":  False,
        "non_ready_reasons": "limit_files=5",   # string, not list
        "gate_status":       "ok",
    })
    with pytest.raises(SystemExit, match="`non_ready_reasons` must be a list"):
        trainer._verify_dataset_production_ready(dataset_dir)


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
    assert "no_dataset_provenance" in taint["dataset_non_ready_reasons"]
    # Contract change (2nd pass): promotion_eligible moved to the builder.
    assert "promotion_eligible" not in taint


# ── Codex 2026-08-15 (2nd pass) P1 + P2: _build_saved_provenance builder ────

class _StubArgs:
    """Minimal args namespace for _build_saved_provenance."""
    def __init__(self, **kw):
        self.lr           = kw.get("lr", 3e-4)
        self.batch_size   = kw.get("batch_size", 4096)
        self.weight_decay = kw.get("weight_decay", 1e-5)
        self.seed         = kw.get("seed", 42)
        self.d4_augmentation = kw.get("d4_augmentation", "off")


def _make_dataset_dir_with_sha_files(tmp_path: Path) -> Path:
    """Create the four binaries + metadata.npz that _build_saved_provenance
    sha256s.  Content is stub — we only need reproducible bytes."""
    d = tmp_path / "ds"
    d.mkdir()
    for name in ("parent_feats.f32.bin", "targets.f32.bin",
                 "targets_uniform.f32.bin", "targets_empirical.f32.bin"):
        (d / name).write_bytes(b"stub")
    np.savez(
        str(d / "metadata.npz"),
        state_keys=np.array(["sk"], dtype=object),
        provenance=np.array(json.dumps(_ready_prov()), dtype=object),
    )
    return d


def _clean_taint() -> dict:
    return {
        "dataset_production_ready":  True,
        "dataset_non_ready_reasons": [],
        "non_production_override":   False,
        "dataset_provenance":        _ready_prov(),
    }


def _nonready_taint() -> dict:
    return {
        "dataset_production_ready":  False,
        "dataset_non_ready_reasons": ["limit_files=5", "coverage_floor not met"],
        "non_production_override":   True,
        "dataset_provenance":        _nonready_prov("limit_files=5"),
    }


def _gate_summary(overall: str) -> dict:
    return {
        "overall_verdict":              overall,
        "n_cells_total":                9,
        "n_cells_pass":                 9 if overall == "PASS" else 0,
        "n_cells_fail":                 9 if overall == "FAIL" else 0,
        "n_cells_skip_insufficient":    0,
        "n_cells_skip_degenerate":      0,
        "n_cells_skip_non_finite":      0,
    }


def test_builder_promotion_eligible_only_when_all_three_conditions_met(
    trainer, tmp_path,
):
    """Codex 2026-08-15 (2nd pass) P1: promotion_eligible requires
    dataset_eligibility AND stage_e_gate == PASS AND frozen thresholds id."""
    ds = _make_dataset_dir_with_sha_files(tmp_path)
    prov, label = trainer._build_saved_provenance(
        dataset_taint=_clean_taint(),
        dataset_dir=ds,
        gate_results=[],
        gate_summary=_gate_summary("PASS"),
        gate_thresholds_frozen_id="stage_e_thresholds_v1_2026-08-15",
        args=_StubArgs(),
        n_train=100, n_val=20,
        best_val_loss=0.1, epochs_trained=1,
        tr_means=[0.0, 0.0, 0.0],
    )
    assert prov["promotion_eligible"] is True
    assert prov["promotion_ineligible_reasons"] == []
    assert prov["dataset_eligibility"] is True
    assert prov["stage_e_gate_verdict"] == "PASS"
    assert prov["gate_thresholds_frozen"] is True
    assert prov["gate_thresholds_frozen_id"] == "stage_e_thresholds_v1_2026-08-15"
    assert label == "gap_net_v3_candidate"
    assert prov["model"] == "gap_net_v3_candidate"


def test_builder_gate_fail_blocks_promotion(trainer, tmp_path):
    """Codex 2026-08-15 (2nd pass) P1: a clean-dataset run whose Stage E
    gate FAILs must NOT be promotable."""
    ds = _make_dataset_dir_with_sha_files(tmp_path)
    prov, label = trainer._build_saved_provenance(
        dataset_taint=_clean_taint(),
        dataset_dir=ds,
        gate_results=[],
        gate_summary=_gate_summary("FAIL"),
        gate_thresholds_frozen_id="stage_e_thresholds_v1",
        args=_StubArgs(),
        n_train=100, n_val=20,
        best_val_loss=0.1, epochs_trained=1,
        tr_means=[0.0, 0.0, 0.0],
    )
    assert prov["promotion_eligible"] is False
    assert prov["dataset_eligibility"] is True
    assert prov["stage_e_gate_verdict"] == "FAIL"
    assert any("stage_e_gate_verdict" in r for r in prov["promotion_ineligible_reasons"])
    assert label == "gap_net_v3_candidate_nonpromotion"
    assert prov["model"] == "gap_net_v3_candidate_nonpromotion"


def test_builder_gate_skip_blocks_promotion(trainer, tmp_path):
    """A run whose Stage E overall_verdict is FAIL_INSUFFICIENT_COVERAGE
    (all cells SKIP) must NOT be promotable — this was the specific
    scenario Codex flagged as still labelled promotable in da596ef."""
    ds = _make_dataset_dir_with_sha_files(tmp_path)
    prov, _ = trainer._build_saved_provenance(
        dataset_taint=_clean_taint(),
        dataset_dir=ds,
        gate_results=[],
        gate_summary=_gate_summary("FAIL_INSUFFICIENT_COVERAGE"),
        gate_thresholds_frozen_id="stage_e_thresholds_v1",
        args=_StubArgs(),
        n_train=100, n_val=20,
        best_val_loss=0.1, epochs_trained=1,
        tr_means=[0.0, 0.0, 0.0],
    )
    assert prov["promotion_eligible"] is False
    assert prov["stage_e_gate_verdict"] == "FAIL_INSUFFICIENT_COVERAGE"


def test_builder_missing_frozen_id_blocks_promotion_even_with_pass(
    trainer, tmp_path,
):
    """Even a clean-dataset PASS run without --gate-thresholds-frozen-id
    is not promotable — draft thresholds carry no promotion authority."""
    ds = _make_dataset_dir_with_sha_files(tmp_path)
    prov, label = trainer._build_saved_provenance(
        dataset_taint=_clean_taint(),
        dataset_dir=ds,
        gate_results=[],
        gate_summary=_gate_summary("PASS"),
        gate_thresholds_frozen_id=None,          # not frozen
        args=_StubArgs(),
        n_train=100, n_val=20,
        best_val_loss=0.1, epochs_trained=1,
        tr_means=[0.0, 0.0, 0.0],
    )
    assert prov["promotion_eligible"] is False
    assert prov["gate_thresholds_frozen"] is False
    assert prov["gate_thresholds_frozen_id"] is None
    assert any("gate_thresholds_frozen_id" in r
               for r in prov["promotion_ineligible_reasons"])
    assert label == "gap_net_v3_candidate_nonpromotion"


def test_builder_empty_frozen_id_blocks_promotion(trainer, tmp_path):
    """Empty string / whitespace-only id is not a valid frozen identifier."""
    ds = _make_dataset_dir_with_sha_files(tmp_path)
    prov, _ = trainer._build_saved_provenance(
        dataset_taint=_clean_taint(),
        dataset_dir=ds,
        gate_results=[],
        gate_summary=_gate_summary("PASS"),
        gate_thresholds_frozen_id="   ",         # whitespace only
        args=_StubArgs(),
        n_train=100, n_val=20,
        best_val_loss=0.1, epochs_trained=1,
        tr_means=[0.0, 0.0, 0.0],
    )
    assert prov["promotion_eligible"] is False
    assert prov["gate_thresholds_frozen"] is False


def test_builder_nonready_dataset_blocks_promotion_and_lists_all_reasons(
    trainer, tmp_path,
):
    """A non-ready dataset (override set) with gate FAIL and no frozen id
    must record ALL three ineligibility reasons in the saved provenance."""
    ds = _make_dataset_dir_with_sha_files(tmp_path)
    prov, label = trainer._build_saved_provenance(
        dataset_taint=_nonready_taint(),
        dataset_dir=ds,
        gate_results=[],
        gate_summary=_gate_summary("FAIL"),
        gate_thresholds_frozen_id=None,
        args=_StubArgs(),
        n_train=100, n_val=20,
        best_val_loss=0.1, epochs_trained=1,
        tr_means=[0.0, 0.0, 0.0],
    )
    reasons_blob = " | ".join(prov["promotion_ineligible_reasons"])
    assert prov["promotion_eligible"] is False
    assert "dataset_not_production_ready" in reasons_blob
    assert "non_production_override_set" in reasons_blob
    assert "stage_e_gate_verdict" in reasons_blob
    assert "gate_thresholds_frozen_id" in reasons_blob
    assert label == "gap_net_v3_candidate_nonpromotion"


def test_builder_missing_overall_verdict_treated_as_missing(trainer, tmp_path):
    """A gate_summary without overall_verdict (unexpected schema drift)
    must be treated as non-PASS, not silently pass."""
    ds = _make_dataset_dir_with_sha_files(tmp_path)
    prov, _ = trainer._build_saved_provenance(
        dataset_taint=_clean_taint(),
        dataset_dir=ds,
        gate_results=[],
        gate_summary={},                          # no overall_verdict
        gate_thresholds_frozen_id="stage_e_thresholds_v1",
        args=_StubArgs(),
        n_train=100, n_val=20,
        best_val_loss=0.1, epochs_trained=1,
        tr_means=[0.0, 0.0, 0.0],
    )
    assert prov["promotion_eligible"] is False
    assert prov["stage_e_gate_verdict"] == "MISSING"


# ── Codex 2026-08-15 (2nd pass) P2: save/load round-trip through builder ────

def test_builder_saved_npz_round_trip_all_promotion_fields_preserved(
    trainer, tmp_path,
):
    """End-to-end round-trip through the exact seam `main()` uses:
    builder → _save → np.load.  Every promotion-related field must survive."""
    ds = _make_dataset_dir_with_sha_files(tmp_path)
    provenance, model_label = trainer._build_saved_provenance(
        dataset_taint=_clean_taint(),
        dataset_dir=ds,
        gate_results=[],
        gate_summary=_gate_summary("PASS"),
        gate_thresholds_frozen_id="stage_e_thresholds_v1_2026-08-15",
        args=_StubArgs(),
        n_train=100, n_val=20,
        best_val_loss=0.1, epochs_trained=1,
        tr_means=[0.0, 0.0, 0.0],
    )
    model = trainer.GapNetV3()
    out = tmp_path / "candidate.npz"
    trainer._save(model, out, provenance)

    z = np.load(str(out), allow_pickle=True)
    reloaded = json.loads(str(z["provenance"]))
    for k in ("model", "dataset_production_ready", "dataset_non_ready_reasons",
              "non_production_override", "dataset_eligibility",
              "stage_e_gate_verdict", "gate_thresholds_frozen",
              "gate_thresholds_frozen_id", "promotion_eligible",
              "promotion_ineligible_reasons"):
        assert k in reloaded, f"promotion field {k!r} missing from saved NPZ"
    assert reloaded["model"] == model_label == "gap_net_v3_candidate"
    assert reloaded["promotion_eligible"] is True
    assert reloaded["gate_thresholds_frozen_id"] == "stage_e_thresholds_v1_2026-08-15"


def test_builder_saved_npz_round_trip_nonpromotion_carries_reasons(
    trainer, tmp_path,
):
    """A non-promotable run's saved NPZ must record ineligibility reasons
    that a downstream consumer can enumerate without running the trainer
    again."""
    ds = _make_dataset_dir_with_sha_files(tmp_path)
    provenance, model_label = trainer._build_saved_provenance(
        dataset_taint=_nonready_taint(),
        dataset_dir=ds,
        gate_results=[],
        gate_summary=_gate_summary("FAIL"),
        gate_thresholds_frozen_id=None,
        args=_StubArgs(),
        n_train=100, n_val=20,
        best_val_loss=0.1, epochs_trained=1,
        tr_means=[0.0, 0.0, 0.0],
    )
    model = trainer.GapNetV3()
    out = tmp_path / "nonpromo.npz"
    trainer._save(model, out, provenance)

    z = np.load(str(out), allow_pickle=True)
    reloaded = json.loads(str(z["provenance"]))
    assert reloaded["model"] == model_label == "gap_net_v3_candidate_nonpromotion"
    assert reloaded["promotion_eligible"] is False
    assert reloaded["promotion_ineligible_reasons"]   # non-empty
    assert reloaded["gate_thresholds_frozen_id"] is None


# ── Codex 2026-08-15 (2nd pass) P2: bounded integration test through main() ─

def test_main_integration_records_all_promotion_fields(trainer, tmp_path, monkeypatch):
    """Bounded integration test: run trainer.main() end-to-end on a tiny
    synthetic dataset and assert every promotion field that the builder
    emits also appears in the saved NPZ.

    This catches the specific regression Codex 2026-08-15 flagged as
    uncaught: main() dropping a field between builder output and _save
    call.  Bounded because we run only 1 epoch on a 60-row synthetic
    dataset (~1s).

    The synthetic empirical target is all-NaN, so the Stage E gate SKIPs
    every cell → overall_verdict is FAIL_INSUFFICIENT_COVERAGE → the
    saved NPZ is NOT promotable.  That is intentional: the test asserts
    the promotion-block reasons are recorded (not that promotion succeeds)."""
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()

    # Build a synthetic dataset with 60 rows: 40 train, 20 val.
    n_total  = 60
    n_train  = 40
    n_val    = 20
    n_bands  = 3
    board_dim = 79
    heads    = 3

    rng = np.random.default_rng(0)
    # Feats: random floats.
    feats = rng.standard_normal((n_total, board_dim)).astype(np.float32)
    (dataset_dir / "parent_feats.f32.bin").write_bytes(feats.tobytes())
    # Targets: constant per row to make model easy to fit.
    tgt = np.ones((n_total, heads), dtype=np.float32) * 0.1
    (dataset_dir / "targets.f32.bin").write_bytes(tgt.tobytes())
    unif = np.ones((n_total, heads), dtype=np.float32) * 0.5
    (dataset_dir / "targets_uniform.f32.bin").write_bytes(unif.tobytes())
    # Empirical: NaN → all cells SKIP → gate FAIL_INSUFFICIENT_COVERAGE.
    # That's what we want here — this test verifies promotion is CORRECTLY
    # BLOCKED when the gate SKIPs, exercising the full main() → builder path.
    emp = np.full((n_total, heads), np.nan, dtype=np.float32)
    (dataset_dir / "targets_empirical.f32.bin").write_bytes(emp.tobytes())

    split      = np.array([0] * n_train + [1] * n_val, dtype=np.int64)
    band_idx   = rng.integers(0, n_bands, n_total).astype(np.int64)
    np.savez(
        str(dataset_dir / "metadata.npz"),
        split=split,
        band_idx=band_idx,
        state_keys=np.array([f"sk_{i}" for i in range(n_total)], dtype=object),
        provenance=np.array(json.dumps(_ready_prov()), dtype=object),
    )

    out_path = tmp_path / "candidate.npz"
    monkeypatch.setattr(sys, "argv", [
        "train_gap_net_v3.py",
        "--dataset-dir",  str(dataset_dir),
        "--out",          str(out_path),
        "--epochs",       "1",
        "--patience",     "0",
        "--batch-size",   "16",
        "--gate-thresholds-frozen-id", "stage_e_thresholds_v1_test",
    ])
    trainer.main()

    assert out_path.exists(), "main() must produce the output NPZ"
    z = np.load(str(out_path), allow_pickle=True)
    prov = json.loads(str(z["provenance"]))

    # These are the builder's promotion fields — main() must not drop any.
    for k in ("model", "dataset_production_ready", "dataset_non_ready_reasons",
              "non_production_override", "dataset_eligibility",
              "stage_e_gate_verdict", "gate_thresholds_frozen",
              "gate_thresholds_frozen_id", "promotion_eligible",
              "promotion_ineligible_reasons",
              "gate_results", "gate_summary",
              "dataset_feats_sha256", "dataset_metadata_sha256",
              "best_val_loss", "epochs_trained", "seed"):
        assert k in prov, f"required field {k!r} missing from main() output"

    # This synthetic dataset produces all-SKIP gate → NOT promotable.
    assert prov["dataset_production_ready"] is True
    assert prov["dataset_eligibility"] is True
    assert prov["gate_thresholds_frozen"] is True
    assert prov["gate_thresholds_frozen_id"] == "stage_e_thresholds_v1_test"
    assert prov["stage_e_gate_verdict"] != "PASS"   # empirical all-NaN → SKIP
    assert prov["promotion_eligible"] is False
    assert any("stage_e_gate_verdict" in r
               for r in prov["promotion_ineligible_reasons"])
    assert prov["model"] == "gap_net_v3_candidate_nonpromotion"


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

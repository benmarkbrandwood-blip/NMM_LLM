from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import report_target_refresh_mature_fork_diagnostic as report
from scripts import train_s_gen_v2 as trainer


def _row(index: int) -> dict:
    outcomes = (trainer.WIN_REWARD, trainer.DRAW_LONG, trainer.LOSS_REWARD)
    return {
        "game": 100 + index,
        "outcome": outcomes[index % 3],
        "game_type": "vs_frozen" if index % 2 == 0 else "vs_sanmill",
        "learner_color": "W" if index % 2 == 0 else "B",
        "termination_reason": (
            "win_fewer_than_three" if index % 3 == 0 else "draw_threefold"
        ),
        "ply": 24 + index,
        "entropy_mean": 1.5,
        "chosen_prob_mean": 0.3,
        "malom_preserving_move_rate": 0.75,
    }


def test_training_summary_keeps_raw_strata_and_partial_final_block() -> None:
    summary = report._training_summary([_row(index) for index in range(51)])

    assert summary["overall"]["games"] == 51
    assert summary["by_opponent"]["vs_frozen"]["games"] == 26
    assert summary["by_opponent"]["vs_sanmill"]["games"] == 25
    assert summary["by_learner_colour"]["W"]["games"] == 26
    assert len(summary["by_termination_reason"]) == 2
    blocks = summary["fixed_blocks_up_to_50_games"]
    assert [block["games"] for block in blocks] == [50, 1]
    assert [block["complete_50_game_window"] for block in blocks] == [True, False]


def test_strict_jsonl_accepts_uniform_windows_crlf(tmp_path: Path) -> None:
    path = tmp_path / "windows.jsonl"
    path.write_bytes(b'{"value": 1}\r\n{"value": 2}\r\n')

    assert report._strict_jsonl(path) == [{"value": 1}, {"value": 2}]


def test_strict_jsonl_rejects_mixed_line_framing(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_bytes(b'{"value": 1}\r\n{"value": 2}\n')

    with pytest.raises(
        report.MatureTargetRefreshReportError,
        match="JSONL framing differs",
    ):
        report._strict_jsonl(path)


def test_strict_jsonl_rejects_missing_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "unterminated.jsonl"
    path.write_bytes(b'{"value": 1}')

    with pytest.raises(
        report.MatureTargetRefreshReportError,
        match="JSONL framing differs",
    ):
        report._strict_jsonl(path)


def test_strict_jsonl_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_bytes(b'{"value": 1, "value": 2}\n')

    with pytest.raises(
        report.MatureTargetRefreshReportError,
        match="duplicate JSON key",
    ):
        report._strict_jsonl(path)


def test_strict_jsonl_rejects_non_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "non-finite.jsonl"
    path.write_bytes(b'{"value": NaN}\n')

    with pytest.raises(
        report.MatureTargetRefreshReportError,
        match="non-finite JSON value",
    ):
        report._strict_jsonl(path)


def test_frozen_hash_bound_reference_inputs_accept_their_tracked_format() -> None:
    contract = report.load_contract(report.DEFAULT_CONTRACT)
    policy_contract = contract["measurement_contract"]["policy_distribution"]
    direct_contract = contract["measurement_contract"]["direct_crossplay"]

    policy_corpus = report._read_hash_bound_json_object(
        report.DEFAULT_POLICY_CORPUS,
        expected_sha256=policy_contract["fixed_corpus_sha256"],
        label="policy corpus",
    )
    report.validate_phase_corpus(policy_corpus)

    replay_corpus = report._read_hash_bound_json_object(
        report.DEFAULT_REPLAY_CORPUS,
        expected_sha256=direct_contract["replay_corpus_sha256"],
        label="replay corpus",
    )
    report.validate_phase_replay_development_corpus(replay_corpus)

    replay_audit = report._read_hash_bound_json_object(
        report.DEFAULT_REPLAY_AUDIT,
        expected_sha256=direct_contract["replay_audit_sha256"],
        label="replay audit",
    )
    report.validate_phase_replay_sanmill_audit(
        replay_audit,
        corpus=replay_corpus,
    )


def test_hash_bound_reference_input_rejects_byte_drift(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text('{"value": 1}\n', encoding="utf-8")

    with pytest.raises(
        report.MatureTargetRefreshReportError,
        match="policy corpus identity differs",
    ):
        report._read_hash_bound_json_object(
            path,
            expected_sha256="0" * 64,
            label="policy corpus",
        )


def test_generated_authority_json_remains_canonical_only(tmp_path: Path) -> None:
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps({"value": 1}, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        report.MatureTargetRefreshReportError,
        match="JSON is not canonical",
    ):
        report._strict_json(path)


def test_analysis_source_accepts_reporter_only_descendant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "a" * 40
    analysis_head = "b" * 40
    monkeypatch.setattr(
        report,
        "_git_identity",
        lambda commit: {
            "training_head": commit,
            "analysis_head": analysis_head,
            "origin_dev": analysis_head,
        },
    )
    monkeypatch.setattr(
        report,
        "_git_output",
        lambda *arguments: (
            "scripts/report_target_refresh_mature_fork_diagnostic.py\n"
            "tests/test_target_refresh_mature_fork_report.py"
        ),
    )

    source = report._inspect_analysis_source(expected)

    assert source["post_training_analysis_paths"] == [
        "scripts/report_target_refresh_mature_fork_diagnostic.py",
        "tests/test_target_refresh_mature_fork_report.py",
    ]


def test_analysis_source_rejects_training_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "a" * 40
    analysis_head = "b" * 40
    monkeypatch.setattr(
        report,
        "_git_identity",
        lambda commit: {
            "training_head": commit,
            "analysis_head": analysis_head,
            "origin_dev": analysis_head,
        },
    )
    monkeypatch.setattr(
        report,
        "_git_output",
        lambda *arguments: "scripts/train_s_gen_v2.py",
    )

    with pytest.raises(
        report.MatureTargetRefreshReportError,
        match="post-training source changes are not analysis-only",
    ):
        report._inspect_analysis_source(expected)

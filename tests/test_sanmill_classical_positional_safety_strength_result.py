from __future__ import annotations

import json
from pathlib import Path

from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "docs/evidence/"
    "sanmill-classical-positional-safety-strength-v1-manifest-2026-08-20.json"
)


def _load_verified_result() -> dict:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    body = dict(result)
    identity = body.pop("result_identity")
    assert canonical_sha256(body) == identity
    return result


def test_completed_result_is_bound_and_within_authorized_envelope() -> None:
    result = _load_verified_result()
    assert result["result_identity"] == (
        "0e8112776c9fe626c2a84a4710fa9fc09a27cf02789339bfc56fe4ae15665d61"
    )
    assert result["plan_identity"] == (
        "87aebf86fbaaede73560fc4706dcf54778f2d9bb80ecbd8c89be1ed9060882e5"
    )
    assert result["authorization_identity"] == (
        "bbc4be1594f238fed030d1929bb5e85db708feca57404b13e768a6005bab6a6e"
    )
    assert result["known_answer_gate"]["passed"] is True
    assert result["known_answer_gate"]["differing_game_ids"] == []
    assert result["resources"]["complete_games"] == 480
    assert result["resources"]["within_all_limits"] is True
    assert result["resources"]["filtered_games"] == 192
    assert result["resources"]["unfiltered_games"] == 192
    assert result["resources"]["known_answer_games"] == 96


def test_completed_result_records_exact_zero_final_gate_increment() -> None:
    result = _load_verified_result()
    assert result["gate_final_status"]["requests"] == 3739
    assert result["gate_final_status"]["applied_requests"] == 3739
    assert result["gate_final_status"]["interventions"] == 0
    assert result["gate_final_status"]["runtime_failures"] == 0
    assert result["gate_final_status"]["selection_failures"] == 0
    assert result["gate_final_status"]["unavailable_requests"] == 0

    for difficulty in (9, 10):
        primary = result["analysis"]["primary"][
            f"difficulty_{difficulty}_filtered_minus_unfiltered"
        ]
        assert primary["support"] == 48
        assert primary["mean"] == 0.0
        assert primary["lower"] == 0.0
        assert primary["upper"] == 0.0
        assert primary["half_width"] == 0.0
        assert primary["difference_distribution"] == {"0.0": 48}
        assert primary["precision_adequate"] is True

        filtered = result["analysis"]["by_arm"][
            f"classical-d{difficulty}-a-pos"
        ]
        unfiltered = result["analysis"]["by_arm"][
            f"classical-d{difficulty}-unfiltered"
        ]
        assert filtered["strict_wdl"] == unfiltered["strict_wdl"]
        assert filtered["terminal_reasons"] == unfiltered["terminal_reasons"]
        assert filtered["original_self_downgrade"]["events"] == 0
        assert unfiltered["original_self_downgrade"]["events"] == 0


def test_completed_result_preserves_fail_closed_access_and_branch_findings() -> None:
    result = _load_verified_result()
    assert result["database_read_only_audit"]["unchanged"] is True
    assert result["database_read_only_audit"]["database_writes"] == 0
    assert all(value == 0 for value in result["access_audit"].values())
    assert result["unfiltered_v2_comparison"]["exact_match"] is False
    assert result["unfiltered_v2_comparison"]["reference_games"] == 192
    assert result["unfiltered_v2_comparison"]["observed_games"] == 192
    assert result["unfiltered_v2_comparison"]["differing_games"] == 166
    assert result["unfiltered_comparison_continuation"][
        "filter_disabled_side_effect_excluded_by_canary"
    ] is True
    assert len(result["machine_records"]["candidate_compact"]) == 384
    assert len(result["machine_records"]["reproduction"]) == 96

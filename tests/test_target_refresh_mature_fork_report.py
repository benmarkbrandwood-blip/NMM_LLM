from __future__ import annotations

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

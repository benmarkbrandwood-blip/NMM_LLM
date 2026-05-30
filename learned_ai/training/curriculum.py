"""Staged curriculum controller.

Stages (configurable lengths per config YAML):
    1: encoding sanity (1-2 self-play games, no learning)
    2: train vs random opponent — graduates when rolling win rate >= stage2_win_threshold
    3: train vs heuristic, difficulty 1 → difficulty_max — each level gated at
       stage3_difficulty_threshold; graduates to stage 4 when the threshold is
       held at the maximum difficulty
    4: self-play with checkpoint opponent pool
    5: human-data fine-tuning (no-op stub when no human data present)

Stage 2 and 3 thresholds use a rolling evaluation window (deque).  The budget
keys (stage2_episodes etc.) act as *hard safety caps* so a plateauing model
cannot stay stuck forever.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


STAGE_NAMES = [
    "stage1_sanity",
    "stage2_vs_random",
    "stage3_vs_heuristic",
    "stage4_self_play",
    "stage5_human_finetune",
]


@dataclass
class CurriculumState:
    current_stage: int = 1
    episodes_in_stage: int = 0
    stage_budgets: Dict[int, int] = None  # type: ignore[assignment]
    heuristic_difficulty: int = 1
    last_event: Optional[str] = None  # set by step(); read by trainer for display

    def episodes_left(self) -> int:
        budget = self.stage_budgets.get(self.current_stage, 0)
        return max(0, budget - self.episodes_in_stage)

    def stage_name(self) -> str:
        idx = max(1, min(len(STAGE_NAMES), self.current_stage))
        return STAGE_NAMES[idx - 1]


class Curriculum:
    """Track which stage we are in and advance when the conditions are met."""

    def __init__(
        self,
        stage_budgets: Dict[int, int],
        start_stage: int = 1,
        stage2_win_threshold: float = 0.60,
        stage3_difficulty_start: int = 1,
        stage3_difficulty_max: int = 10,
        stage3_difficulty_threshold: float = 0.55,
        eval_window: int = 200,
    ) -> None:
        if start_stage not in stage_budgets:
            raise ValueError(
                f"start_stage {start_stage} not in budgets {list(stage_budgets)}"
            )
        self.state = CurriculumState(
            current_stage=start_stage,
            episodes_in_stage=0,
            stage_budgets=dict(stage_budgets),
            heuristic_difficulty=stage3_difficulty_start,
        )
        self._stage2_win_threshold = float(stage2_win_threshold)
        self._stage3_difficulty_max = max(1, int(stage3_difficulty_max))
        self._stage3_difficulty_threshold = float(stage3_difficulty_threshold)
        self._eval_window = int(eval_window)
        self._recent_results: deque = deque(maxlen=self._eval_window)

    # ------------------------------------------------------------------
    # Outcome tracking

    def record_outcome(self, won: bool) -> None:
        """Call once per episode with the learned agent's result."""
        self._recent_results.append(1.0 if won else 0.0)

    def rolling_win_rate(self) -> float:
        if not self._recent_results:
            return 0.0
        return sum(self._recent_results) / len(self._recent_results)

    def window_full(self) -> bool:
        return len(self._recent_results) >= self._eval_window

    # ------------------------------------------------------------------
    # Core advance logic

    def step(self) -> None:
        self.state.last_event = None
        self.state.episodes_in_stage += 1
        stage = self.state.current_stage
        max_stage = max(self.state.stage_budgets)

        if stage >= max_stage:
            return  # final stage; just count

        budget_exhausted = self.state.episodes_left() <= 0

        if stage <= 1:
            # Stage 1: pure budget
            if budget_exhausted:
                self._advance_stage()
            return

        if stage == 2:
            win_rate = self.rolling_win_rate()
            threshold_met = self.window_full() and win_rate >= self._stage2_win_threshold
            if threshold_met or budget_exhausted:
                self._advance_stage()
                reason = "threshold" if threshold_met else "budget"
                self.state.last_event = f"stage_advance:{reason}:{win_rate:.3f}"
            return

        if stage == 3:
            win_rate = self.rolling_win_rate()
            threshold_met = self.window_full() and win_rate >= self._stage3_difficulty_threshold
            if threshold_met:
                if self.state.heuristic_difficulty < self._stage3_difficulty_max:
                    self.state.heuristic_difficulty += 1
                    self._recent_results.clear()
                    self.state.last_event = (
                        f"difficulty_bump:{self.state.heuristic_difficulty}:{win_rate:.3f}"
                    )
                else:
                    # At max difficulty and threshold held — graduate to stage 4.
                    self._advance_stage()
                    self.state.last_event = f"stage_advance:threshold:{win_rate:.3f}"
            elif budget_exhausted:
                # Safety cap: force-advance without meeting threshold.
                self._advance_stage()
                self.state.last_event = f"stage_advance:budget:{win_rate:.3f}"
            return

        # Stage 4+: pure budget
        if budget_exhausted:
            self._advance_stage()

    def _advance_stage(self) -> None:
        self.state.current_stage += 1
        self.state.episodes_in_stage = 0
        self._recent_results.clear()

    # ------------------------------------------------------------------

    def opponent_kind(self) -> str:
        """Map stage -> opponent type label used by the trainer."""
        stage = self.state.current_stage
        if stage <= 2:
            return "random"
        if stage == 3:
            return "heuristic"
        return "self"

    def heuristic_difficulty(self) -> int:
        """Current difficulty level for stage 3 heuristic opponent."""
        return self.state.heuristic_difficulty

    def finished(self) -> bool:
        return (
            self.state.current_stage == max(self.state.stage_budgets)
            and self.state.episodes_left() == 0
        )

    @classmethod
    def from_config(cls, cfg: dict, start_stage: int = 1) -> "Curriculum":
        budgets = {
            1: int(cfg.get("stage1_episodes", 10)),
            2: int(cfg.get("stage2_episodes", 30000)),
            3: int(cfg.get("stage3_episodes", 60000)),
            4: int(cfg.get("stage4_episodes", 70000)),
            5: int(cfg.get("stage5_episodes", 0)),
        }
        return cls(
            stage_budgets=budgets,
            start_stage=start_stage,
            stage2_win_threshold=float(cfg.get("stage2_win_threshold", 0.60)),
            stage3_difficulty_start=int(cfg.get("stage3_difficulty_start", 1)),
            stage3_difficulty_max=int(cfg.get("stage3_difficulty_max", 10)),
            stage3_difficulty_threshold=float(cfg.get("stage3_difficulty_threshold", 0.55)),
            eval_window=int(cfg.get("eval_window", 200)),
        )

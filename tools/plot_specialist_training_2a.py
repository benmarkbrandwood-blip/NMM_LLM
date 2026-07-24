"""tools/plot_specialist_training_2a.py — v2a training dashboard for specialist(s).

Adds a 7th row rendering the 50-game rolling termination-reason mix (win by
material / blocking, rules-based draws, max-ply truncation) as a stacked area,
plus infrastructure-failure counts as separate scatter markers.  Distinguishes
truncation draws from rules-based draws so a high "draw" rate is not confused
with genuine drawn play.  See docs/gen_2b_plan.md for the design rationale.

Original doc:

Reads train_log.jsonl from each specified checkpoint folder and plots:
  Row 0: entropy_mean + chosen_prob_mean  (policy exploration / confidence)
  Row 1: malom_win_move_rate + heuristic_top1_rate + policy_top1_rate
  Row 2: best_win_rate + win_rate_200 + draw rate  /  ply on rhs axis
  Row 3: sentinel_chosen_mean vs sentinel_mean + gap shaded
  Row 4: reward breakdown (sentinel/heuristic)  /  LR on rhs axis
  Row 5: retro reward (outcome signal)
  Row 6: termination-reason mix (50-game rolling %) + infra failures

Recovery event markers on win-rate panel:
  black  dashed — hot-explore triggered (Stage 1)
  green  dashed — checkpoint restored   (Stage 2)
  green  solid  — post-grace resurrection

One column per specialist.  Defaults to open/mid/end/generalist when no folders given.
Refreshes every 20 minutes.

Usage:
    .venv/bin/python tools/plot_specialist_training.py
    .venv/bin/python tools/plot_specialist_training.py learned_ai/checkpoints/scaffolded/s_gen_v2a
    .venv/bin/python tools/plot_specialist_training.py s_open_v2 s_mid_v2
    .venv/bin/python tools/plot_specialist_training.py --interval 5   # minutes
    .venv/bin/python tools/plot_specialist_training.py --no-loop      # single render

Folder arguments can be absolute paths or relative to the repo root or to
learned_ai/checkpoints/scaffolded/ — whichever exists first.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT      = Path(__file__).resolve().parent.parent
CKPT_BASE = ROOT / "learned_ai" / "checkpoints" / "scaffolded"

DEFAULT_SPECIALISTS = [
    ("Opening",    CKPT_BASE / "s_open_v2"),
    ("Midgame",    CKPT_BASE / "s_mid_v2"),
    ("Endgame",    CKPT_BASE / "s_end_v2"),
    ("Generalist", CKPT_BASE / "s_gen_v2"),
]

SMOOTH = 50   # rolling-average window


# ── Data helpers ──────────────────────────────────────────────────────────────

def _resolve_folder(arg: str) -> Path:
    p = Path(arg)
    if p.is_absolute() and p.is_dir():
        return p
    bases = (Path.cwd(), ROOT, CKPT_BASE)
    # First pass: prefer a candidate that actually contains train_log.jsonl,
    # so an empty stub directory can't mask the real run.
    for base in bases:
        candidate = base / p
        if (candidate / "train_log.jsonl").is_file():
            return candidate.resolve()
    # Second pass: fall back to any matching directory.
    for base in bases:
        candidate = base / p
        if candidate.is_dir():
            return candidate.resolve()
    return (CKPT_BASE / p).resolve()


def _load(path: Path) -> list[dict]:
    log = path / "train_log.jsonl"
    if not log.exists():
        return []
    rows = []
    with open(log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    # Discard data from old/smoke-test runs by detecting fresh-run starts:
    # a game number that drops to near-zero (< 10) after a real run has begun.
    # Intra-run restarts (e.g. 1500→1302 from checkpoint mismatch) are NOT
    # stripped — only fresh starts are. Event rows are always preserved so
    # recovery markers are never lost even if they predate last_reset.
    last_reset = 0
    max_game   = -1
    for i, r in enumerate(rows):
        if "event" in r:          # event rows have high game numbers; exclude from scan
            continue
        g = r.get("game")
        if isinstance(g, int):
            if g < 10 and max_game > 50:   # fresh-run start, not an intra-run restart
                last_reset = i
            if g > max_game:
                max_game = g
    result = []
    for i, r in enumerate(rows):
        if i >= last_reset or "event" in r:
            result.append(r)
    return result


def _smooth(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return arr
    kernel = np.ones(window) / window
    out = np.convolve(arr, kernel, mode="full")[: len(arr)]
    for i in range(min(window - 1, len(arr))):
        out[i] = arr[: i + 1].mean()
    return out


def _get(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is not None:
            xs.append(r.get("game", len(xs)))
            ys.append(float(v))
    return xs, ys


_WIN_OUTCOME  =  1.5
_LOSS_OUTCOME = -1.0


def _get_draw_rate(rows: list[dict]) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in rows:
        outcome = r.get("outcome")
        if outcome is not None:
            v = float(outcome)
            xs.append(r.get("game", len(xs)))
            ys.append(1.0 if (v != _WIN_OUTCOME and v != _LOSS_OUTCOME) else 0.0)
    return xs, ys


def _get_advances(rows: list[dict]) -> list[tuple[int, int]]:
    advances, prev = [], None
    for r in rows:
        d, g = r.get("difficulty"), r.get("game")
        if d is None or g is None:
            continue
        if prev is not None and d > prev:
            advances.append((g, d))
        prev = d
    return advances


def _get_recovery_events(rows: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    events: dict[str, list[tuple[int, dict]]] = {
        "recovery_stage1": [],
        "recovery_stage2": [],
        "resurrection":    [],
    }
    # Collect explicit event rows (stage2 + resurrection are reliably logged this way)
    for r in rows:
        ev = r.get("event")
        if ev in events:
            events[ev].append((r.get("game", 0), r))

    # Detect hot-explore start from per-game hot_explore_remaining field (0 → N transition).
    # This catches stage1 even when _log_event missed the log_every boundary.
    # seen_zero_her guard: after a checkpoint restore hot_explore_remaining may start >0,
    # so we only fire the transition once we've seen at least one her==0 row first.
    # DEDUP_WINDOW: an event row from _log_event is written at the log boundary, but the
    # first batched game row with her>0 lands a few games later. Both sources point at
    # the same Stage 1 trigger — treat any transition within DEDUP_WINDOW of an existing
    # stage1 event as a duplicate.
    DEDUP_WINDOW  = 100
    prev_her      = 0
    seen_zero_her = False
    stage1_games  = [g for g, _ in events["recovery_stage1"]]
    def _near_existing(g: int) -> bool:
        return any(abs(g - eg) <= DEDUP_WINDOW for eg in stage1_games)
    for r in rows:
        if "event" in r:
            continue
        her = r.get("hot_explore_remaining") or 0
        g   = r.get("game", 0)
        if her > 0 and prev_her == 0 and seen_zero_her and not _near_existing(g):
            events["recovery_stage1"].append((g, r))
            stage1_games.append(g)
        if her == 0:
            seen_zero_her = True
        prev_her = her

    events["recovery_stage1"].sort(key=lambda t: t[0])
    return events


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _plot_series(ax, xs, ys, label, color, window=SMOOTH, alpha_raw=0.15, linestyle="-"):
    if not ys:
        return
    smoothed = _smooth(ys, window)
    ax.plot(xs, ys, color=color, alpha=alpha_raw, linewidth=0.6)
    ax.plot(xs, smoothed, color=color, linewidth=1.6, label=label, linestyle=linestyle)


_ADVANCE_COLOR = "#448AFF"   # blue — was bright green

def _draw_advances(axes_col: list, advances: list[tuple[int, int]], label_ax=None) -> None:
    if not advances:
        return
    for ax in axes_col:
        first_on_ax = (ax is label_ax)
        for game, _ in advances:
            lbl = "diff advance" if first_on_ax else None
            ax.axvline(game, color=_ADVANCE_COLOR, linewidth=0.9, linestyle="--", alpha=0.7, zorder=3, label=lbl)
            first_on_ax = False
    ax_top = axes_col[0]
    for game, level in advances:
        ax_top.text(game, 1.0, f"L{level}", fontsize=6, color=_ADVANCE_COLOR,
                    ha="left", va="top", transform=ax_top.get_xaxis_transform(),
                    zorder=4, bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, lw=0))


def _draw_recovery_events(ax, events: dict[str, list], add_labels: bool = False) -> None:
    _labeled: set[str] = set()

    def _vline(game, color, ls, lw, label):
        if add_labels and label not in _labeled:
            lbl = label
            _labeled.add(label)
        else:
            lbl = None
        ax.axvline(game, color=color, linewidth=lw, linestyle=ls, alpha=0.85, zorder=3, label=lbl)

    for game, _ in events["recovery_stage1"]:
        _vline(game, "black",    "--", 1.0, "hot-explore")
    for game, _ in events["recovery_stage2"]:
        _vline(game, "#4CAF50", "--", 1.0, "restore")
    for game, _ in events["resurrection"]:
        _vline(game, "#4CAF50", "-",  1.4, "resurrect")


def _twin_rhs(ax, color: str, ylabel: str):
    """Create a right-hand secondary axis with matching tick colour."""
    ax2 = ax.twinx()
    ax2.tick_params(axis="y", labelcolor=color, labelsize=6)
    ax2.set_ylabel(ylabel, fontsize=6, color=color)
    return ax2


def _caption(ax, text: str) -> None:
    """Add a small italic grey caption below the x-axis."""
    ax.set_xlabel(text, fontsize=5.5, color="#909090", style="italic", labelpad=4)


# ── Main draw ─────────────────────────────────────────────────────────────────

def draw(fig, _axes_unused, specialists):
    """Rebuild all panels from scratch on each refresh (handles twinx cleanly)."""
    n_cols = len(specialists)
    fig.clf()
    raw = fig.subplots(7, n_cols, sharex=False)
    # Normalise to list-of-lists regardless of shape
    if n_cols == 1:
        axes = [[ax] for ax in raw]
    else:
        axes = [list(row) for row in raw]

    for col, (name, ckpt_dir) in enumerate(specialists):
        rows  = _load(ckpt_dir)
        n_games = sum(1 for r in rows if "event" not in r)

        ax_ent  = axes[0][col]   # entropy / confidence
        ax_top1 = axes[1][col]   # top-1 + malom
        ax_wr   = axes[2][col]   # win rates  (ply rhs)
        ax_sent = axes[3][col]   # sentinel signal
        ax_rew  = axes[4][col]   # reward breakdown  (LR rhs)
        ax_ret  = axes[5][col]   # retro reward
        ax_term = axes[6][col]   # termination-reason mix

        subtitle = f"{name}  (n={n_games})"

        # ── Row 0: entropy + chosen_prob ─────────────────────────────────
        ax_ent.set_title(subtitle, fontsize=9, pad=3)
        xs_ent, ys_ent = _get(rows, "entropy_mean")
        xs_cp,  ys_cp  = _get(rows, "chosen_prob_mean")
        _plot_series(ax_ent, xs_ent, ys_ent, "entropy",     "#2196F3")
        _plot_series(ax_ent, xs_cp,  ys_cp,  "chosen prob", "#4CAF50")
        ax_ent.set_ylim(bottom=0)
        ax_ent.legend(fontsize=6, loc="upper right")
        _caption(ax_ent, "entropy→0 = policy collapsed / stuck;  chosen prob↑ = model more decisive")

        # ── Row 1: malom + heuristic_top1 + policy_top1 ──────────────────
        xs_m, ys_m = _get(rows, "malom_win_move_rate")
        xs_h, ys_h = _get(rows, "heuristic_top1_rate")
        xs_p, ys_p = _get(rows, "policy_top1_rate")
        _plot_series(ax_top1, xs_m, ys_m, "malom win-move",  "#2196F3")
        _plot_series(ax_top1, xs_h, ys_h, "heuristic top-1", "#FF9800")
        _plot_series(ax_top1, xs_p, ys_p, "policy top-1",    "#4CAF50")
        ax_top1.set_ylim(0, 1.05)
        ax_top1.legend(fontsize=6, loc="lower right")
        ax_top1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        _caption(ax_top1, "malom↑ = Malom-optimal moves;  policy≈heuristic = copying;  gap widening = diverging")

        # ── Row 2: win / draw rates  +  ply (rhs) ────────────────────────
        rec_events = _get_recovery_events(rows)
        reset_games = {g for g, _ in rec_events["recovery_stage2"] + rec_events["resurrection"]}

        def _plot_wr(ax, xs, ys, label, color):
            """Like _plot_series but inserts NaN gaps in smoothed line at recovery resets."""
            if not ys:
                return
            sm = _smooth(ys, SMOOTH).copy()
            if reset_games:
                for i, x in enumerate(xs):
                    if any(rg <= x <= rg + SMOOTH for rg in reset_games):
                        sm[i] = float("nan")
            ax.plot(xs, ys,  color=color, alpha=0.15, linewidth=0.6)
            ax.plot(xs, sm,  color=color, linewidth=1.6, label=label)

        xs_b, ys_b = _get(rows, "best_win_rate")
        xs_w, ys_w = _get(rows, "win_rate_200")
        xs_d, ys_d = _get_draw_rate(rows)
        _plot_wr(ax_wr, xs_b, ys_b, "best win rate", "#E91E63")
        _plot_wr(ax_wr, xs_w, ys_w, "win rate 200",  "#9C27B0")
        _plot_wr(ax_wr, xs_d, ys_d, "draw rate",     "#FF9800")
        ax_wr.set_ylim(0, 1.05)
        ax_wr.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        ax_wr2 = _twin_rhs(ax_wr, "#00BCD4", "ply")
        xs_ply, ys_ply = _get(rows, "ply")
        if ys_ply:
            ax_wr2.plot(xs_ply, ys_ply, color="#00BCD4", alpha=0.10, linewidth=0.6)
            ax_wr2.plot(xs_ply, _smooth(ys_ply, SMOOTH),
                        color="#00BCD4", linewidth=1.2, label="ply")

        h1, l1 = ax_wr.get_legend_handles_labels()
        h2, l2 = ax_wr2.get_legend_handles_labels()
        ax_wr.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower right")
        _caption(ax_wr, "win↑ good;  draw↑ = passive play;  ply↑ = longer / more drawn games")

        # ── Row 3: sentinel chosen vs mean + gap ──────────────────────────
        xs_sc, ys_sc = _get(rows, "sentinel_chosen_mean")
        xs_sm, ys_sm = _get(rows, "sentinel_mean")
        _plot_series(ax_sent, xs_sc, ys_sc, "chosen sentinel", "#00BCD4", alpha_raw=0.12)
        _plot_series(ax_sent, xs_sm, ys_sm, "mean sentinel",   "#607D8B", alpha_raw=0.12)
        if ys_sc and ys_sm:
            clen = min(len(xs_sc), len(xs_sm))
            xs_c = xs_sc[:clen]
            sm_s = _smooth(ys_sm[:clen], SMOOTH)
            sc_s = _smooth(ys_sc[:clen], SMOOTH)
            ax_sent.fill_between(xs_c, sm_s, sc_s, where=(sc_s >= sm_s),
                                 alpha=0.25, color="#00BCD4", label="gap (chosen > mean)")
            ax_sent.fill_between(xs_c, sm_s, sc_s, where=(sc_s < sm_s),
                                 alpha=0.25, color="#FF5722")
        ax_sent.set_ylim(0, 1.05)
        ax_sent.legend(fontsize=6, loc="lower right")
        _caption(ax_sent, "teal gap (chosen>mean) = model using sentinel signal;  flat gap = ignoring it")

        # ── Row 4: sentinel/heuristic rewards  +  LR (rhs) ────────
        xs_rs, ys_rs = _get(rows, "reward_sentinel_mean")
        xs_rh, ys_rh = _get(rows, "reward_heuristic_mean")
        _plot_series(ax_rew, xs_rs, ys_rs, "sentinel",  "#00BCD4", alpha_raw=0.20)
        _plot_series(ax_rew, xs_rh, ys_rh, "heuristic", "#FF9800", alpha_raw=0.20)
        ax_rew.axhline(0, color="white", alpha=0.20, linewidth=0.7, linestyle="--")

        ax_rew2 = _twin_rhs(ax_rew, "#F44336", "LR ×10⁻⁵")
        xs_lr, ys_lr = _get(rows, "lr")
        if ys_lr:
            ys_lr_scaled = [v * 1e5 for v in ys_lr]
            ax_rew2.plot(xs_lr, ys_lr_scaled, color="#F44336", linewidth=0.9, alpha=0.85, label="LR")
            ax_rew2.set_ylim(bottom=0)

        h1, l1 = ax_rew.get_legend_handles_labels()
        h2, l2 = ax_rew2.get_legend_handles_labels()
        ax_rew.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower right")
        _caption(ax_rew, "sentinel/heur near 0 = reward mostly from retro (outcome);  LR at min (0.5) = model losing")

        # ── Row 5: retro reward ────────────────────────────────────────────
        xs_rr, ys_rr = _get(rows, "reward_retro_mean")
        _plot_series(ax_ret, xs_rr, ys_rr, "retro", "#4CAF50", alpha_raw=0.20)
        ax_ret.axhline(0, color="white", alpha=0.20, linewidth=0.7, linestyle="--")
        ax_ret.legend(fontsize=6, loc="lower right")
        _caption(ax_ret, "retro↑ = winning outcome reward;  retro↓ = losing;  near 0 = draws / mixed")

        # ── Row 6: termination-reason mix (50-game rolling %) ─────────────
        # Stacked area for the five reasons that produce a valid outcome,
        # colour-coded green/red/grey.  Infra counts drawn as a separate
        # marker series so a burst of infrastructure failure is visible.
        _term_series = [
            ("term_win_lt3_pct",      "win <3",     "#2E7D32"),  # dark green
            ("term_win_blocked_pct",  "win blocked", "#66BB6A"),  # light green
            ("term_draw_trunc_pct",   "draw trunc", "#9E9E9E"),   # grey
            ("term_draw_rep_pct",     "draw rep",   "#BDBDBD"),
            ("term_draw_50_pct",      "draw 50",    "#E0E0E0"),
            ("term_loss_blocked_pct", "loss blocked", "#EF5350"),
            ("term_loss_lt3_pct",     "loss <3",    "#B71C1C"),
        ]
        _xs = [r.get("game", 0) for r in rows if "term_win_lt3_pct" in r]
        if _xs:
            _stack = []
            for key, _lbl, _c in _term_series:
                _stack.append([float(r.get(key, 0.0)) for r in rows if "term_win_lt3_pct" in r])
            ax_term.stackplot(
                _xs, *_stack,
                labels=[s[1] for s in _term_series],
                colors=[s[2] for s in _term_series],
                alpha=0.85,
            )
            ax_term.set_ylim(0, 100)

            # Infra counts drawn as red x's at their game position
            xs_il = [r.get("game", 0) for r in rows if r.get("term_infra_learner_count", 0) > 0]
            ys_il = [50 for _ in xs_il]  # placed mid-panel for visibility
            xs_io = [r.get("game", 0) for r in rows if r.get("term_infra_opponent_count", 0) > 0]
            ys_io = [50 for _ in xs_io]
            if xs_il:
                ax_term.scatter(xs_il, ys_il, marker="x", s=25, color="#FF5252",
                                label="infra learner", zorder=5)
            if xs_io:
                ax_term.scatter(xs_io, ys_io, marker="+", s=35, color="#D500F9",
                                label="infra opponent", zorder=5)

            ax_term.legend(fontsize=5, loc="lower right", ncol=2)
        _caption(ax_term, "50-game rolling termination-reason mix (excl. infra); "
                          "high draw_trunc = games hitting max_ply, not rules-based draws")

        # ── Advancement + recovery markers on all panels ──────────────────
        # Labels are only produced for the top panel so its legend collects all
        # four vertical-line entries (diff advance, hot-explore, restore, resurrect).
        advances = _get_advances(rows)
        _all_axes = [ax_ent, ax_top1, ax_wr, ax_sent, ax_rew, ax_ret, ax_term]
        _draw_advances(_all_axes, advances, label_ax=ax_ent)
        for _ax in _all_axes:
            _draw_recovery_events(_ax, rec_events, add_labels=(_ax is ax_ent))
        # Re-render top panel legend so new vline labels appear alongside entropy/chosen prob.
        ax_ent.legend(fontsize=6, loc="upper right", ncol=2)

        for ax in _all_axes:
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3, linewidth=0.4)

    row_labels = [
        "Entropy / confidence",
        "Top-1 + Malom %",
        "Win rates",
        "Sentinel signal",
        "Rewards / LR",
        "Retro reward",
        "Termination mix %",
    ]
    for row, label in enumerate(row_labels):
        axes[row][0].set_ylabel(label, fontsize=7)

    fig.suptitle(
        f"Specialist training  ·  {SMOOTH}-game smoothed  ·  {time.strftime('%H:%M:%S')}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5)
    fig.canvas.draw()
    fig.canvas.flush_events()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="*",
                        help="Checkpoint folder(s) containing train_log.jsonl. "
                             "Can be absolute, relative to repo root, or just the "
                             "folder name under learned_ai/checkpoints/scaffolded/. "
                             "Defaults to open/mid/end/generalist.")
    parser.add_argument("--interval", type=float, default=20.0,
                        help="Refresh interval in minutes (default 20)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Render once and exit")
    args = parser.parse_args()

    if args.folders:
        specialists = [(Path(f).name, _resolve_folder(f)) for f in args.folders]
    else:
        specialists = DEFAULT_SPECIALISTS

    n_cols = len(specialists)
    fig_w  = max(6, 4.5 * n_cols)
    fig    = plt.figure(figsize=(fig_w, 21))

    plt.ion()
    draw(fig, None, specialists)

    if args.no_loop:
        plt.ioff()
        plt.show()
        return

    interval_s = args.interval * 60
    try:
        while True:
            deadline = time.time() + interval_s
            while time.time() < deadline:
                plt.pause(1.0)
            draw(fig, None, specialists)
    except KeyboardInterrupt:
        pass

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

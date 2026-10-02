"""tools/plot_gen5_progress.py — Gen5 training progress from gen4 resume point.

Shows only entries from game 14263 onwards (the point where gen4 BWDB best.pt
was loaded into the gen5 script).  The old gen5 scratch run (games 1–~10721,
stuck at difficulty 7) is excluded.

Usage:
    .venv/bin/python tools/plot_gen5_progress.py
    .venv/bin/python tools/plot_gen5_progress.py --start-game 14263
    .venv/bin/python tools/plot_gen5_progress.py --interval 5
    .venv/bin/python tools/plot_gen5_progress.py --no-loop
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

DEFAULT_FOLDER = CKPT_BASE / "s_gen_v5" / "from_v4"
DEFAULT_START  = 14263
SMOOTH         = 50


def _load(path: Path, start_game: int = DEFAULT_START) -> list[dict]:
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

    # Keep only rows from the gen4-seeded run onwards.
    # Event rows that fall within range are kept; those before start_game dropped.
    result = []
    for r in rows:
        g = r.get("game", 0)
        if g >= start_game:
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
    for r in rows:
        ev = r.get("event")
        if ev in events:
            events[ev].append((r.get("game", 0), r))

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


def _plot_series(ax, xs, ys, label, color, window=SMOOTH, alpha_raw=0.15, linestyle="-"):
    if not ys:
        return
    smoothed = _smooth(ys, window)
    ax.plot(xs, ys, color=color, alpha=alpha_raw, linewidth=0.6)
    ax.plot(xs, smoothed, color=color, linewidth=1.6, label=label, linestyle=linestyle)


_ADVANCE_COLOR = "#448AFF"

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
    ax2 = ax.twinx()
    ax2.tick_params(axis="y", labelcolor=color, labelsize=6)
    ax2.set_ylabel(ylabel, fontsize=6, color=color)
    return ax2


def _caption(ax, text: str) -> None:
    ax.set_xlabel(text, fontsize=5.5, color="#909090", style="italic", labelpad=4)


def draw(fig, folder: Path, start_game: int):
    rows    = _load(folder, start_game)
    n_games = sum(1 for r in rows if "event" not in r)

    fig.clf()
    raw  = fig.subplots(6, 1, sharex=False)
    axes = list(raw)

    ax_ent, ax_top1, ax_wr, ax_sent, ax_rew, ax_ret = axes

    subtitle = f"Gen5 diff_DB2  (from game {start_game}, n={n_games})"

    # ── Row 0: entropy + chosen_prob ─────────────────────────────────────────
    ax_ent.set_title(subtitle, fontsize=9, pad=3)
    xs_ent, ys_ent = _get(rows, "entropy_mean")
    xs_cp,  ys_cp  = _get(rows, "chosen_prob_mean")
    _plot_series(ax_ent, xs_ent, ys_ent, "entropy",     "#2196F3")
    _plot_series(ax_ent, xs_cp,  ys_cp,  "chosen prob", "#4CAF50")
    ax_ent.set_ylim(bottom=0)
    ax_ent.legend(fontsize=6, loc="upper right")
    _caption(ax_ent, "entropy→0 = policy collapsed / stuck;  chosen prob↑ = model more decisive")

    # ── Row 1: malom + heuristic_top1 + policy_top1 ──────────────────────────
    xs_m, ys_m = _get(rows, "malom_win_move_rate")
    xs_h, ys_h = _get(rows, "heuristic_top1_rate")
    xs_p, ys_p = _get(rows, "policy_top1_rate")
    _plot_series(ax_top1, xs_m, ys_m, "malom win-move",  "#2196F3")
    _plot_series(ax_top1, xs_h, ys_h, "heuristic top-1", "#FF9800")
    _plot_series(ax_top1, xs_p, ys_p, "policy top-1",    "#4CAF50")
    ax_top1.set_ylim(0, 1.05)
    ax_top1.legend(fontsize=6, loc="lower right")
    ax_top1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    _caption(ax_top1, "malom↑ = Malom-optimal;  policy≈heuristic = copying;  gap widening = diverging")

    # ── Row 2: win / draw rates  +  ply (rhs) ────────────────────────────────
    rec_events  = _get_recovery_events(rows)
    reset_games = {g for g, _ in rec_events["recovery_stage2"] + rec_events["resurrection"]}

    def _plot_wr(ax, xs, ys, label, color):
        if not ys:
            return
        sm = _smooth(ys, SMOOTH).copy()
        if reset_games:
            for i, x in enumerate(xs):
                if any(rg <= x <= rg + SMOOTH for rg in reset_games):
                    sm[i] = float("nan")
        ax.plot(xs, ys, color=color, alpha=0.15, linewidth=0.6)
        ax.plot(xs, sm, color=color, linewidth=1.6, label=label)

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
        ax_wr2.plot(xs_ply, _smooth(ys_ply, SMOOTH), color="#00BCD4", linewidth=1.2, label="ply")

    h1, l1 = ax_wr.get_legend_handles_labels()
    h2, l2 = ax_wr2.get_legend_handles_labels()
    ax_wr.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower right")
    _caption(ax_wr, "win↑ good;  draw↑ = passive play;  ply↑ = longer / more drawn games")

    # ── Row 3: sentinel chosen vs mean + gap ──────────────────────────────────
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

    # ── Row 4: sentinel/heuristic rewards  +  LR (rhs) ───────────────────────
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
    _caption(ax_rew, "sentinel/heur near 0 = reward mostly from retro (outcome);  LR at min = model losing")

    # ── Row 5: retro reward ───────────────────────────────────────────────────
    xs_rr, ys_rr = _get(rows, "reward_retro_mean")
    _plot_series(ax_ret, xs_rr, ys_rr, "retro", "#4CAF50", alpha_raw=0.20)
    ax_ret.axhline(0, color="white", alpha=0.20, linewidth=0.7, linestyle="--")
    ax_ret.legend(fontsize=6, loc="lower right")
    _caption(ax_ret, "retro↑ = winning outcome reward;  retro↓ = losing;  near 0 = draws / mixed")

    # ── Advancement + recovery markers on all panels ──────────────────────────
    advances = _get_advances(rows)
    _all_axes = [ax_ent, ax_top1, ax_wr, ax_sent, ax_rew, ax_ret]
    _draw_advances(_all_axes, advances, label_ax=ax_ent)
    for _ax in _all_axes:
        _draw_recovery_events(_ax, rec_events, add_labels=(_ax is ax_ent))
    ax_ent.legend(fontsize=6, loc="upper right", ncol=2)

    row_labels = [
        "Entropy / confidence",
        "Top-1 + Malom %",
        "Win rates",
        "Sentinel signal",
        "Rewards / LR",
        "Retro reward",
    ]
    for ax, label in zip(axes, row_labels):
        ax.set_ylabel(label, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3, linewidth=0.4)

    if not n_games:
        ax_ent.text(0.5, 0.5, f"No data yet (waiting for games ≥ {start_game})",
                    ha="center", va="center", transform=ax_ent.transAxes,
                    fontsize=11, color="#aaaaaa")

    fig.suptitle(
        f"Gen5 diff_DB2 (from gen4 BWDB, game {start_game}+)  ·  "
        f"{SMOOTH}-game smoothed  ·  {time.strftime('%H:%M:%S')}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.5)
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=str(DEFAULT_FOLDER),
                        help="Checkpoint folder containing train_log.jsonl "
                             f"(default: {DEFAULT_FOLDER})")
    parser.add_argument("--start-game", type=int, default=DEFAULT_START,
                        help=f"First game to include (default {DEFAULT_START})")
    parser.add_argument("--interval", type=float, default=20.0,
                        help="Refresh interval in minutes (default 20)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Render once and exit")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_absolute():
        for base in (Path.cwd(), ROOT, CKPT_BASE):
            candidate = base / folder
            if (candidate / "train_log.jsonl").is_file():
                folder = candidate
                break
        else:
            folder = CKPT_BASE / folder

    fig = plt.figure(figsize=(10, 18))
    plt.ion()
    draw(fig, folder, args.start_game)

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
            draw(fig, folder, args.start_game)
    except KeyboardInterrupt:
        pass

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

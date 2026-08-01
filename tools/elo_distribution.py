"""tools/elo_distribution.py — Plot Elo distribution across human game files.

Shows four views so you can decide which Elo threshold to use for value-net retraining:
  1. Per-game minimum Elo  (quality of the weaker player — the binding constraint)
  2. Per-game average Elo  (typical game quality)
  3. Per-player peak Elo   (best each player ever achieved)
  4. Per-player mean Elo   (typical strength of each player)

Usage:
    .venv/bin/python tools/elo_distribution.py
    .venv/bin/python tools/elo_distribution.py --games-dir data/human_games --bins 40
    .venv/bin/python tools/elo_distribution.py --save elo_dist.png
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_elo_data(games_dir: Path) -> tuple[list, list, dict]:
    """Return (game_min_elos, game_avg_elos, player_elos_dict)."""
    game_min: list[float] = []
    game_avg: list[float] = []
    player_elos: dict[str, list[float]] = defaultdict(list)

    files = sorted(games_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files found in {games_dir}")

    print(f"Scanning {len(files):,} game files...", flush=True)
    errors = 0
    for i, f in enumerate(files):
        if i % 10000 == 0 and i > 0:
            print(f"  {i:,} / {len(files):,}", flush=True)
        try:
            d = json.loads(f.read_text())
        except Exception:
            errors += 1
            continue

        we = d.get("white_elo")
        be = d.get("black_elo")
        wp = d.get("white_player")
        bp = d.get("black_player")

        if we is not None and be is not None:
            game_min.append(min(we, be))
            game_avg.append((we + be) / 2.0)
        if we is not None and wp:
            player_elos[wp].append(float(we))
        if be is not None and bp:
            player_elos[bp].append(float(be))

    print(f"Done. {len(game_min):,} games with Elo data; {errors} parse errors.")
    return game_min, game_avg, player_elos


def percentile_table(label: str, data: list[float], thresholds: list[int]) -> None:
    arr = np.array(data)
    print(f"\n{'─'*56}")
    print(f"  {label}")
    print(f"{'─'*56}")
    print(f"  n={len(arr):,}   min={arr.min():.0f}   max={arr.max():.0f}"
          f"   mean={arr.mean():.0f}   median={np.median(arr):.0f}")
    print(f"{'─'*56}")
    print(f"  {'Percentile':>12}  {'Elo cutoff':>10}  {'Games kept':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}")
    for pct in thresholds:
        cutoff = np.percentile(arr, pct)
        kept = int((arr >= cutoff).sum())
        print(f"  {f'top {100-pct}%':>12}  {cutoff:>10.0f}  {kept:>10,}")
    print(f"{'─'*56}")


def plot(game_min, game_avg, player_elos, bins: int, save: str | None) -> None:
    player_peak = [max(v) for v in player_elos.values()]
    player_mean = [np.mean(v) for v in player_elos.values()]

    datasets = [
        (game_min,    "Per-game minimum Elo\n(weaker player)", "#e07b54"),
        (game_avg,    "Per-game average Elo",                  "#5b8fd4"),
        (player_peak, "Per-player peak Elo",                   "#6abf6a"),
        (player_mean, "Per-player mean Elo",                   "#b07fd4"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Elo Distribution — Human Game Files", fontsize=14, fontweight="bold")

    for ax, (data, title, color) in zip(axes.flat, datasets):
        arr = np.array(data)
        lo = int(arr.min() // 50) * 50
        hi = int(arr.max() // 50 + 1) * 50
        bin_edges = np.arange(lo, hi + 1, (hi - lo) / bins)

        counts, edges = np.histogram(arr, bins=bin_edges)
        ax.bar(edges[:-1], counts, width=np.diff(edges), color=color,
               alpha=0.82, edgecolor="white", linewidth=0.4, align="edge")

        # percentile lines
        for pct, ls, lw in [(50, "--", 1.2), (75, "-.", 1.2), (90, "-", 1.5), (95, "-", 1.8)]:
            val = np.percentile(arr, pct)
            ax.axvline(val, color="black", linestyle=ls, linewidth=lw, alpha=0.7,
                       label=f"p{pct} = {val:.0f}")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Elo rating", fontsize=9)
        ax.set_ylabel("Number of " + ("games" if "game" in title.lower() else "players"),
                      fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=7.5, framealpha=0.6)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        # annotate n
        ax.text(0.98, 0.97, f"n={len(arr):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#333")

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {save}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Elo distribution of human game files")
    ap.add_argument("--games-dir", default="data/human_games",
                    help="Directory of JSONL game files (default: data/human_games)")
    ap.add_argument("--bins", type=int, default=35,
                    help="Number of histogram bins (default: 35)")
    ap.add_argument("--save", default=None,
                    help="Save plot to file instead of displaying (e.g. elo_dist.png)")
    args = ap.parse_args()

    games_dir = Path(args.games_dir)
    if not games_dir.is_dir():
        sys.exit(f"Directory not found: {games_dir}")

    game_min, game_avg, player_elos = load_elo_data(games_dir)

    # Print percentile tables
    thresholds = [50, 60, 70, 75, 80, 85, 90, 95]
    percentile_table("Per-game minimum Elo (weaker player)", game_min, thresholds)
    percentile_table("Per-game average Elo",                 game_avg, thresholds)

    player_peak = [max(v) for v in player_elos.values()]
    player_mean = [float(np.mean(v)) for v in player_elos.values()]
    percentile_table(f"Per-player peak Elo  ({len(player_peak):,} unique players)",
                     player_peak, thresholds)
    percentile_table(f"Per-player mean Elo  ({len(player_mean):,} unique players)",
                     player_mean, thresholds)

    plot(game_min, game_avg, player_elos, args.bins, args.save)


if __name__ == "__main__":
    main()

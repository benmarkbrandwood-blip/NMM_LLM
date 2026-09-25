# Dual Formation Guide

## What it does

The Formation Guide currently shows a single target arrangement for the stronger
player (6 or 7 pieces vs 4) — yellow arrows pointing each piece toward its
destination in a known winning pattern.

The **Dual Formation Guide** extends this to show a second target arrangement
simultaneously (brown arrows), so the weaker player cannot resist both threats
at the same time.

---

## Why two formations create an unblockable fork

The opponent has 4 pieces.  Each opponent piece can "block" at most the squares
it currently occupies or can reach in a few moves.  If the stronger player aims
at formation A (target squares T_A) and formation B (target squares T_B):

- A square in **T_A ∩ T_B** (shared by both) costs the opponent one piece to
  block both formations simultaneously — efficient for the opponent.
- A square in **T_A only** costs one piece to block just formation A.
- A square in **T_B only** costs one piece to block just formation B.

Minimum opponent pieces needed to block both = |T_A ∩ T_B| + |T_A \ T_B| +
|T_B \ T_A|, capped at how many squares they can actually cover in time.

For a 6v4 endgame with 2–3 shared squares between the two formations:

    shared  = 2-3
    A-only  = 3-4
    B-only  = 3-4
    min pieces to block both ≈ 2-3 + 3-4 = ~7-8

The opponent only has 4.  The fork is mathematically unblockable.

---

## The sweet spot: 2–3 shared squares

| Shared squares | Properties |
|---|---|
| 5–6 (nearly identical) | Easy for opponent to cover both; not a real fork |
| 3–4 | Good fork; player can transition between A and B in 1–2 moves |
| **2–3 (recommended)** | **Fork unblockable with 4 opponent pieces; transition in 2–4 moves** |
| 0–1 (completely different) | Maximum pressure but long transition; paths may interfere |

2–3 shared squares is the sweet spot: the fork is provably unblockable with 4
opponent pieces, yet the player doesn't need to choose one formation completely
before starting — both paths share a common core (the anchor squares).

---

## Anchor squares

Squares in T_A ∩ T_B — pieces the player should never voluntarily leave, since
they serve double duty in both formations.  Shown as **gold rings** (distinct
from the orange target rings of each individual formation).

---

## Reachability constraint

The secondary formation must be achievable in a reasonable number of moves,
otherwise it is not a credible simultaneous threat.  A secondary formation whose
BFS total travel cost exceeds `primary_cost + cost_slack` is discarded.

- **6v4 cost_slack**: 6 steps (pieces are fewer; transitions are shorter)
- **7v4 cost_slack**: 8 steps (more pieces; slightly more slack)

If no candidate meets the sweet-spot overlap (2–3 shared) within the cost
slack, the window expands outward (4 shared, then 1 shared, then 0 or 5) until
a valid secondary is found or the search is exhausted.

---

## Algorithm

### 6v4 (`best_formation`)

1. All 336 candidates (42 formations × 8 D4 symmetry transforms) are already
   evaluated with exact BFS assignment costs.
2. Pick the primary as before (lowest cost, Malom-verified if requested).
3. For the secondary: iterate the sorted candidate list, skipping the primary.
   For each candidate compute `overlap = |target ∩ primary_target|`.
   Keep candidates with `cost ≤ primary_cost + 6`.
   Select the one whose overlap is closest to the sweet spot midpoint (2–3),
   tie-broken by lowest cost.

### 7v4 (`best_formation_7v4`)

1. The current algorithm pre-screens 41,408 candidates (5,176 formations × 8
   D4) by piece-overlap to a pool of ≤ 100, then does exact BFS on those.
2. For the secondary, extend the pool to ≤ 200 and apply the same sweet-spot
   selection after BFS assignment.

---

## Return shape

Both `best_formation` and `best_formation_7v4` gain a `secondary` key:

```json
{
  "formation_id": "7",
  "target_squares": ["a7","g4","b6","d6","f6","f2"],
  "arrows": [{"from":"a4","to":"a7"}, ...],
  "stay": ["d6"],
  "total_distance": 4,
  "mode_used": "naive",
  "secondary": {
    "formation_id": "22",
    "target_squares": ["a4","g7","b6","d6","f6","c4"],
    "arrows": [{"from":"a7","to":"a4"}, ...],
    "stay": ["b6","d6","f6"],
    "total_distance": 6,
    "shared_squares": ["b6","d6","f6"],
    "anchor_squares": ["b6","d6","f6"]
  }
}
```

`secondary` is `null` when no suitable second formation is found.

---

## UI

| Element | Primary | Secondary |
|---|---|---|
| Target rings | Orange dashed circle | Brown dashed circle |
| Movement arrows | Orange (#ff8c00) | Brown (#a0522d) |
| Origin dot | Orange | Brown |
| Anchor squares | — | Gold ring (#ffd700) overlaid on both |

The gold anchor rings are drawn on top of both sets of coloured rings, giving
a clear visual hierarchy: "always keep your pieces here; direct the others
toward either orange or brown."

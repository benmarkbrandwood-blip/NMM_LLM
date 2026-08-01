# Nine Men's Morris — Phase Strategy Guide

Concise, phase-segmented playing guide. Feed the relevant section(s) to the LLM
at each stage of the game. Side-specific guidance is separated where it differs.

---

## Phase A — Early Placement (own pieces placed: 1–6)

### General rules (both sides)

- **Scatter widely.** No two adjacent own pieces in the first 4–5 placements.
- **Prefer cross-ring positions** — squares that participate in two different mill lines
  (e.g. b4, d6, f4 connect outer and middle rings; e4 connects middle and inner).
- **Avoid outer-ring side mills** (a7-d7-g7, g7-g4-g1, g1-d1-a1, a1-a4-a7) in placements
  1–6 without a capture. They trap two pieces in corner squares (only 2 connections each)
  and severely hurt move-phase mobility.
- **Avoid inner-ring mills** (c5-d5-e5, e5-e4-e3, e3-d3-c3, c3-c4-c5) for the same
  reason — they confine pieces to the smallest ring.
- Closing a mill on placements 1–5 **without a capture** is almost always wrong unless
  the mill has 3 or more empty exit squares and the opponent has no immediate 2-config.
- A **two-for-one placement** — blocks an opponent 2-config while simultaneously creating
  an own 2-config on a different line — maintains initiative even when defending.
  Prefer these over pure blocking or pure setup moves.

### White's priorities (placements 1–4)

- Claim **at least 2 of the 4 cardinal nodes** (b4, d2, d6, f4).
  These have 4 connections each, participate in 2 mills each, and provide the best
  long-term mobility and mill-formation flexibility.
- Build **cross-ring mill skeletons** that span outer and middle rings, or middle and
  inner rings — not same-ring formations.
- If a cardinal node is taken by Black, contest the adjacent cross-ring node
  (e.g. if Black takes d6, control d7 or d5 to limit that line).

### Black's priorities (placements 1–4)

- **React to White's structural plan.** If White is building a cross-ring skeleton,
  block the convergence point rather than developing your own structure passively.
- A placement that **forces White to react** (creates a 2-config White must block)
  is better than a passive positional move, even if the forced square is less obviously
  "good" by position alone.
- Placing on a cardinal node is valuable for Black, but the intent is
  *denying White's plan* — not claiming the node for its own sake.
- Black's goal in placements 1–4: create independent opportunity structures while
  limiting White's ability to build unchallenged cross-ring mills.

---

## Phase B — Late Placement (own pieces placed: 7–9)

### The shift in intent

Stop building abstract opportunity. Start forcing, blocking, and converting.

A placement that only creates isolated structure — no nearby feeder piece, no forcing
continuation, no opponent threat blocked — is a wasted move. This is especially true
on placements 8 and 9.

### White's end-of-placement requirement

White needs **two independent potential mills** (2-configs with no shared pieces) by the
end of placement. Because Black places last, White's 8th placement must leave two live
independent threats that Black's remaining placements cannot both cover.

A single potential mill is not enough: Black can block it with the 9th placement.

### Black's end-of-placement requirement

Black needs only **one potential mill** at the end of placement. Black's 9th placement
can close a mill directly — this is Black's last-move structural advantage.

Placements 7–8 should set up a forced 9th placement: a closing square that White
cannot contest in time, or two simultaneous threats that White cannot both cover.

### Dual-purpose rule (placements 8–9, both sides)

Prefer a square that simultaneously:
1. Blocks or contests an opponent active mill line or mobile mill pivot, **AND**
2. Creates a new own 2-config or advances an existing one.

A pure block with no follow-up is acceptable only when the threat is urgent and
no dual-purpose square exists. A pure setup that ignores active opponent threats
is increasingly wrong on placements 8–9.

### Busy-chain priority

A **forcing chain** — where every own placement compels a predictable opponent response,
ending with a mill closure — outranks a simple immediate mill closure, unless the
immediate mill also delivers a capture.

Quality of a chain depends partly on the **value of the opponent's forced blocking
square**: forcing the opponent onto a corner node (2 connections, limited reuse) is
high-quality forcing; forcing them onto a cardinal node (4 connections) is weaker.

---

## Phase C — Move Phase (midgame)

### Tactical priority ladder (strict order)

1. **Close own mill with a capture that also removes the opponent's immediate mill
   threat** — this dual-purpose conversion is the top priority.
2. **Close own mill + capture** (even if the captured piece is not the most critical).
3. **Block the opponent's immediate mill threat** — mandatory regardless of positional cost.
4. **Execute a forcing chain** toward a fork or cycling mill within 2–3 moves.
5. **Build own 2-config or cycling-mill setup.**
6. **Herd or squeeze opponent pieces** — reduce their legal moves toward a blockade.
7. **General positional improvement.**

### Capture selection

When making a mill, choose which opponent piece to remove:
1. The piece **blocking your best 2-config** (removes the blocker, enables next closure).
2. The **feeder for the opponent's cycling mill** (disrupts their repeating capture plan).
3. A piece on a **high-mobility cardinal or cross node**.
4. A low-value isolated piece (last resort when nothing else applies).

### Dual mill cycling

A closed mill with at least one free exit square can be opened and re-closed every 2
turns to force a capture per cycle. This is the primary winning mechanism in midgame.

Once this structure exists, **preserve it** — do not break a cycling mill without a
clear tactical reason.

**Dual mill oscillation** — two cycling mills sharing a pivot piece — is the strongest
midgame structure. The opponent cannot simultaneously block both closure squares while
you hold the pivot mobile. Maintain and harvest this structure until the endgame.

### Locked mills

A closed mill where all exit squares are opponent-occupied contributes nothing
(no cycling value, no threat). Escape it: move a piece out toward a new 2-config.
Prefer moving the mill piece whose departure still leaves the cleanest re-closure
route for the remaining two pieces.

### Fork (double threat)

Two simultaneous 2-configs that the opponent cannot both block in one move guarantee
a mill closure next turn. Prefer captures and moves that create or advance toward a
fork over those that only maintain piece count.

---

## Phase D — Endgame (≤4 pieces per side; fly phase approaching or active)

### 4v4 — Fly transition caution

Neither side is in fly phase yet. Capturing the opponent's 4th piece gives them
fly-phase freedom — they can jump to any empty square.

Do not make this capture unless your own remaining 3-piece structure (closed mill
or strong 2-config) will be strong in the resulting 4v3 position. If your cycling
mill can force captures without a direct 4v4 capture, prefer that path.

### 4v3 — You have 4 pieces; opponent flies

- Build **two independent 2-configs** (no shared pieces between them). If the
  opponent captures one of your pieces, the remaining 3 still contain a mill threat.
- Keep pieces spread across different rings where possible.
- Do not rush to reduce the opponent to 2 pieces if your own structure would collapse.

### 3v4 — You fly with 3 pieces; opponent has 4

- **Fly-pin rule:** do not move a piece that is the sole blocker of an opponent 2-config.
  Moving it gives the opponent an immediate mill closure and capture.
- Priority: close own mill → build fork → exploit opponent piece separation.
- If the opponent's 4 pieces form two disconnected groups, threaten one group — isolated
  pieces cannot defend each other when you can fly to any square.

### 3v3 — Both players in fly phase

- The winner is whoever achieves a **fork** first: two simultaneous 2-configs that the
  opponent cannot both block with a single move.
- Every move should either build toward your own fork or prevent the opponent's fork.
- Fly-pin rule still applies: never vacate the sole blocker of an opponent 2-config.
- With structurally equal positions, the side to move has a small advantage and can often
  force a win with accurate play.

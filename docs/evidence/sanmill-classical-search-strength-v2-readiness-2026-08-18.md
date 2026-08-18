# Sanmill Classical-Search Strength v2 Readiness

Date: 2026-08-18

Verdict: `ready_for_smoke`

This verdict authorizes no work beyond the product owner's bounded request.
The first formal stage is the frozen known-answer reproduction gate.  The
classical-search runtime is not loaded unless that gate matches every selected
reference game exactly.

## Question and claim boundary

The measurement asks how the product's `origin/main` difficulty 9 and 10
classical coordinators score against the already pinned 100,000-node Sanmill
runtime.  It is an internal directional measurement on one frozen start
subset.  It is not an equivalence, user-population, human-opponent, promotion,
deployment, release, or training claim.

The candidate moves are not constrained by Malom.  Malom is used read-only to
label positional W/D/L self-downgrades as a secondary diagnostic.  This is
`A_pos`-level evidence only and not full-history `A_allow` safety.

## Repository and source state

- Repository: `I:/Mill_Training/NMM_LLM`
- Branch: `dev`
- Measurement implementation commit: `3ed9eb6860ad6cc1818755cb8b88413a847c2000`
- Product source commit: `4e4a7241e9d5427100b46dfe34f5ae384ff9f613`
- Product source tree: `6e9ce5f74fa9feee35014a97b796c7550ba3c3dd`
- Sanmill commit: `a6623f88959f7453594df274fbe1f128af7ff55e`
- Sanmill binary SHA-256: `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`
- Malom trust label: `sector-corrected-v1`

The exported `origin/main` `ai/game_ai.py` and `ai/heuristics.py` files were
compared with `git show origin/main:<path>` by Git blob hash; both pairs were
identical.  The exact main Rust extension was built into the ignored runtime
snapshot.  Its wheel, native extension, Python sources, resolved heuristic
weights, and all read-only product resources are hash-bound by the plan.

The measured product route is the canonical balanced AI-vs-AI `GameAI` route.
It uses the tracked product maximum depth of 14, two threads for wall-clock
calibration, extended quiescence, the evolved heuristic weights, FullgameDB,
EndgameSolvedDB, PhaseValueNet, and GapNet.  The main route's stale Malom path
does not resolve on this machine, so Malom is absent from the candidate search.
The session-level human opening and trajectory state is intentionally absent:
it cannot be reconstructed for frozen midgame starts.  This is recorded as a
measurement-to-interactive-product deviation.

## Timing calibration

Calibration plan identity:
`71dbfcbe3a160211c14e868c0ff4492679cb2ea2b9c175ee3dfbcd0f14beb1f8`.

Calibration result identity:
`84a4c363233dfb3eaaddf510fe357f7e4f9ef6cedc160fdc322cb0a1e0f57c4d`.

Twelve result-blind states were frozen before product search observation: four
per source phase, with very early placement states excluded.  Difficulty 9
used the product's 30-second setting and difficulty 10 used its 60-second
setting.  There were zero complete games and zero Sanmill processes.

The mapping rule was fixed before calibration: take the median positive node
count and round down to 1,000 nodes.  It produced:

- difficulty 9: 13,887,000 nodes;
- difficulty 10: 18,367,000 nodes.

Each difficulty had 12 observations, nine of which entered the ordinary search
path.  Three were handled by product bypass paths.  The positive-node ranges
were 42 to 121,251,840 for difficulty 9 and 42 to 234,518,528 for difficulty
10.  This spread means that a single node budget is a calibrated deterministic
proxy, not a claim that every product move consumes the same wall time.

For each mapped budget, two fresh single-thread instances were run on one
blind state per phase.  All six fresh-instance pairs chose identical moves and
reported identical node counts and completed depths.  The flying canary
finished naturally at 15,106 nodes rather than consuming the full ceiling.

The same-instance canary also chose the same move before and after warming the
Rust transposition table, while depth and elapsed work changed.  The formal
contract therefore creates a fresh AI per game, retains its Rust table within
that game as the product does, and uses one search thread for deterministic
fixed-node work.  Product wall-clock play uses two threads, so the proxy removes
thread scheduling noise at the cost of this documented runtime difference.

Calibration consumed 433.487563600007 seconds, no games, no Sanmill process,
and no database writes.

## Precision correction and frozen v2

The first post-calibration plan, identity `5e6768a629281343a1a462bcfb857037a809c296ac0dd46af703c3cc2a3a642c`,
was never executed.  Its authorization identity is
`6aba86fffb545b8789e7eb374efd1b4132b40a787659b72eab4ee0f67fa9a9f2`.
No measurement marker was created and no known-answer or candidate outcome was
read under that plan.

That v1 selected 32 starts under a 75% time-reserve rule but projected a
9.97-percentage-point half-width while retaining a 7.5-point precision gate.
The contradiction was detected from timing-only inputs.  The immutable v1
records are preserved, and v2 supersedes them before any outcome observation.

Frozen v2 plan identity:
`0bbe5145b83e29ba48617d1f1ec32c0e35e2921457557ab98aa52acb0974fa39`.

Frozen v2 authorization identity:
`a5e3941a5deaa3502fc5d0b81db63b11050b86a8279622e00a11a91677d33460`.

V2 raises the result-blind resource design ceiling from 75% to 85% and freezes
48 phase-balanced starts, 16 per source phase.  The projected primary
half-width is 8.14 percentage points and the frozen maximum is 8.5 points.
This is below the smallest 11-point gap in the motivating 30%/45%/56% scale.
It is not designed to resolve one-point differences or establish equivalence.

The planned work is:

- 96 random-safe known-answer reproduction games;
- 96 difficulty-9 games, two colors per start;
- 96 difficulty-10 games, two colors per start;
- 288 complete games in total;
- 52,614.67 conservatively projected formal seconds;
- 433.49 already consumed calibration seconds;
- 64,800 authorized active seconds and 1,600 authorized games.

The output namespace is new:
`out/evaluation/sanmill-classical-search-strength-v2-20260818-001`.
Automatic retry or resume is not authorized.

## Known-answer and comparison contracts

Before importing the product runtime, the runner recreates the random-safe arm
on the exact 48-start subset.  For both colors it compares the complete move
sequence, terminal reason, strict history digest, no-progress clock, and
repetition clock with the existing attempt-002 record.  Any mismatch is the
task's hard stop.

Sanmill commit, tree, binary, strict options, referee profile, 100,000-node
budget, seed, Malom identity, and source-pool identity must all match the old
measurement.  Existing v4, constrained-specialist, free-specialist, and
random-safe scores are recomputed only on the same 48 starts.

The independent unit is one start after averaging the candidate's two colors.
Each classical-minus-prior contrast uses a 95% normal interval over those 48
paired start-level differences.  Direction is established only if the interval
excludes zero.  A half-width over 8.5 points is explicitly
`precision_inadequate`; it cannot be softened after seeing results.

## Paths and write isolation

Machine-local Sanmill and Malom locations remain resolved through ignored
`data/training_paths.local.json`.  FullgameDB, EndgameSolvedDB, ValueNet,
GapNet, and Malom are opened read-only.  Their file hashes, sizes, timestamps,
and journal/WAL/SHM absence are captured before and after formal work; a change
fails closed.  No checkpoint is loaded or modified by this measurement.

Official selection, confirmation, final-test, research-confirmation, and the
remaining source pool are outside the plan.  Their frozen read counts are zero.

## Verification

Before calibration:

- 41 focused classical-search, strict-referee, safe-guidance, and lightweight
  reproduction tests passed;
- 103 required Malom, DB-teacher, and label-provenance tests passed, plus 498
  parameterized subtests;
- task-scope Ruff passed;
- importing the runner did not load the host `nmm_core` extension.

After v2 freeze, the focused plan tests were rerun and passed.  Formal launch
still requires a clean tracked worktree, no existing Sanmill process, exact
implementation hashes, a fresh output namespace, and all runtime identities.

Exact launch command after the verified commit is clean:

```powershell
.\.venv\Scripts\python.exe `
  scripts/run_sanmill_classical_search_strength.py measure `
  --plan docs/experiments/sanmill-classical-search-strength-v2.json `
  --authorization `
  docs/experiments/sanmill-classical-search-strength-v2/authorization.json
```

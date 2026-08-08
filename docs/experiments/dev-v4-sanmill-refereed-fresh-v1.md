# Fresh Sanmill-Refereed Generalist v1

## Status and authority

Status: `implementation_ready_for_clean-commit_preflight`

Experiment ID: `dev-v4-sanmill-refereed-fresh-v1`

The product owner selected a new fresh training lineage on 8 August 2026.
This document records that lineage and the required architecture. It is not a
managed plan, a smoke authorization, a long-run authorization, or strength
evidence.

## Lineage decision

The model starts from random weights. It must not load, import, or resume:

- the completed 5,000-game local-GameAI baseline;
- the rules-corrected successor-v2 run;
- any maintainer `main` checkpoint; or
- any historical Generalist, specialist, Sentinel, ValueNet, or GapNet
  checkpoint.

The first process omits both `--resume` and `--auto-resume-best` and uses an
empty dedicated output directory and a separate empty SpecialistDB carrying
`malom_label_version=sector-corrected-v1`. Later process segments may use only
verified `exact-resume` from the immediately preceding checkpoint in this same
lineage.

The completed local-GameAI run remains immutable historical comparison
evidence. It is not continuation material for this experiment.

## Required runtime architecture

Sanmill participates in training in two distinct roles:

1. **Authoritative referee for every rollout.** Sanmill owns the complete
   action history, repetition observations, no-progress count, legal actions,
   compulsory removal, terminal status, winner, and terminal reason. This
   applies to frozen-model self-play as well as Sanmill-opponent games,
   confirmations, retries, and any future branch rollout.
2. **Search opponent.** Sanmill replaces the in-repository `GameAI` on the
   non-self-play opponent stratum. A retained run must never instantiate local
   `HeuristicAgent` or `GameAI` as its opponent.

NMM_LLM retains `BoardState` as the policy and feature-encoding mirror. At
every stable logical-turn boundary it must match Sanmill's legal primary
actions, projected board state, side to move, phase, action/logical counts,
and terminal state. Any divergence is a fail-closed segment quarantine, not a
fallback to local adjudication.

One logical ply is one primary placement/movement action plus any compulsory
removal. Sanmill action tokens may therefore increase by two while the logical
ply count increases by one.

## Pinned Sanmill source contract

The training integration targets Sanmill commit:

```text
a6623f88959f7453594df274fbe1f128af7ff55e
```

with Git tree:

```text
17b9b0fd51ee8dac54c0454a6935978a47d19e0c
```

The required process contract is:

- `StrictFailurePolicy=true`;
- `StrictRefereeProfile=mif-stable-moving-v1`;
- `Threads=1` and `UseLazySmp=false`;
- `Shuffling=false` and fixed `SearchShuffleSeed`;
- `MoveTimeMs=0`;
- `go logical nodes N` with an aggregate primary-plus-removal budget;
- `statejson` after every applied complete logical turn;
- no Perfect DB, HumanDB move substitution, patch, trap, shallow-search, or
  random failure fallback; and
- one isolated clean release binary whose size and SHA-256 are frozen before
  a smoke plan is published.

The historical strict-v2 and prefix-replay installations retain their own
older commits and binary identities. This experiment adds a new installation
contract rather than relabelling either historical runtime.

The Windows host's isolated release binary is 5,641,216 bytes with SHA-256:

```text
5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619
```

Its strict-referee identity is:

```text
sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94
```

## Initial technical boundary

The first bounded implementation smoke retains the corrected v4-style
learning path:

- A2C; PPO disabled;
- Sentinel, ValueNet, GapNet, S1A warm-start, imitation mixing, S1B refresher,
  and trainer-side opening forcing disabled;
- HumanDB frequencies and outcomes available while unversioned historical
  Malom columns remain masked;
- corrected Malom and a fresh `sector-corrected-v1` SpecialistDB;
- fixed seed 42 and `batch_games=1`;
- one fixed-work Sanmill level with curriculum advancement disabled; the old
  local-GameAI 55% gate is not reused;
- frozen-target self-play retained as a separately logged stratum;
- Sanmill search used for every non-self-play opponent turn; and
- rule draws reported separately from experiment truncation.

`DrawOnHumanExperience=true` remains part of the Sanmill search contract. It
is a phase-aware search policy, not an opening-book activation. Opening Book,
HumanDB, and Perfect DB prefix corpora remain separate frozen evaluation/data
assets and are not silently injected into this fresh baseline.

## Unfrozen calibration items

The following values require implementation evidence before an immutable
managed plan can be generated:

- the Sanmill node/depth curriculum and its number of levels;
- advancement and cooling behaviour under that curriculum;
- the final frozen-target/Sanmill-opponent ratio;
- smoke and retained-run truncation ceilings;
- process reuse and timeout bounds;
- games and active wall-time budgets; and
- segment and checkpoint cadence after measured throughput.

The prior local `GameAI` difficulty ladder and its 55% advancement gate are
not inherited by implication. Sanmill work levels must be calibrated and
logged in Sanmill-native node/depth terms.

## Required evidence before any smoke

Implementation must prove all of the following before a smoke command can be
reviewed:

1. exact Sanmill source, tree, release binary, licence, rule, strict-referee,
   and option identities;
2. complete logical-turn conversion in both directions, including removal;
3. legal-action and post-turn state parity for learner, frozen-target, and
   Sanmill turns;
4. Sanmill-authoritative repetition, no-progress, material-loss, blocked-loss,
   and truncation separation;
5. hard failure on protocol errors, timeouts, illegal actions, history drift,
   or any attempted fallback;
6. byte-equivalent fixed-seed fresh-process results;
7. exact-resume equivalence across a process boundary; and
8. bounded throughput and resource measurements on the intended Windows host.

No training command is currently authorized. A later smoke must use a new
disposable output directory and SpecialistDB, and its one-run authorization
must be recorded separately.

The implementation-focused baseline currently records:

- 137 trainer, contract, resume, manifest, and Sanmill-referee tests passed;
- 103 Malom/label-provenance tests and 498 parameterized subtests passed;
- two fresh Sanmill processes produced the same semantic first-turn record;
- a reviewed expert line replayed nine logical plies and ten action tokens,
  including one compulsory removal;
- an eight-ply mixed learner/Sanmill rollout recorded four Sanmill searches
  and eight matching authoritative logical plies; and
- changed-scope Ruff checks and `git diff --check` passed.

The 1,246-test full suite did not complete within a 15-minute command bound;
it reached approximately 16% without a failure. Therefore only the focused
results above are claimed for this implementation change.

## Provisional clean-commit preflight

The ignored disposable SpecialistDB
`data/specialist_db.sanmill_refereed_fresh_v1.smoke.sqlite` has been created
empty with `sector-corrected-v1` metadata and no lineage binding. The output
path below does not yet exist. After the implementation is committed on
`dev`, the reviewed read-only command is:

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --preflight smoke `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 `
  --start-mode fresh `
  --out-dir out\sanmill-refereed-fresh-v1-smoke-001 `
  --specialist-db data\specialist_db.sanmill_refereed_fresh_v1.smoke.sqlite `
  --referee-engine sanmill `
  --opponent-engine sanmill `
  --sanmill-node-ladder 1000 `
  --curriculum-advance-policy disabled `
  --diff-start 1 `
  --diff-max 1 `
  --self-play-ratio 0.60 `
  --seed 42 `
  --max-games 2 `
  --segment-games 2 `
  --max-ply 120 `
  --batch-games 1 `
  --sim-ply-depth 5 `
  --temp-start 0.90 `
  --log-every 1 `
  --max-branches-per-game 0 `
  --minimal-rollouts `
  --no-recovery `
  --no-sentinel `
  --no-value-net `
  --no-gap-net `
  --no-s1a-warmstart `
  --no-s1b-refresher `
  --no-imitation-mix `
  --no-opening-forcing
```

The 1,000-node level is only an integration-smoke workload. It is not a
strength baseline or a proposed retained-run curriculum. With seed 42 and a
0.60 frozen-target ratio, the two scheduled primary games deterministically
cover one Sanmill-search opponent game and one frozen-target game, while
Sanmill referees both. Replacing `--preflight smoke` with `--launch smoke`
and adding a run ID remains prohibited until the owner explicitly authorizes
that exact one-run smoke.

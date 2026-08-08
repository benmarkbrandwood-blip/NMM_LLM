# Fresh Sanmill-Refereed Generalist v1

## Status and authority

Status: `long_launch_authorized_pending_managed_plan_and_preflight`

Experiment ID: `dev-v4-sanmill-refereed-fresh-v1`

The product owner selected a new fresh training lineage on 8 August 2026.
This document records that lineage and the required architecture. It is not a
managed plan or strength evidence. The separate
[managed long-run contract](sanmill-refereed-managed-long-v1.md) records the
subsequent bounded launch authorization and its remaining local gates.

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

## Historical calibration items

The following values were initially left open pending implementation evidence:

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

The fixed-resource curriculum decision and its passing schedule smoke now
freeze these items for the first retained run. This section remains as the
historical reason that the values were not copied from the local-GameAI run.

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

At the initial publication of this architecture, no further training command
was authorized and every later smoke required a new disposable output and
SpecialistDB. The completed evidence chain and subsequent bounded long-run
authorization are recorded below and in the managed long-run contract.

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

## Smoke-001 failure

The owner-authorised `sanmill-refereed-fresh-v1-smoke-001` launch failed
closed during its first primary game at clean published commit `aeac29c`. A
valid Sanmill `game_over` FEN used raw action `?`, while NMM_LLM incorrectly
routed it through a projector limited to placing and moving actions. The run
produced no completed game, optimiser update, checkpoint, training log, or
update log. Its output and SpecialistDB are quarantined, and the one-run
authorisation is consumed.

The complete identities, database audit, diagnostic side effect, repair, and
claim boundary are recorded in the
[smoke-001 failure evidence](../evidence/sanmill-refereed-fresh-v1-smoke-001-failure-2026-08-08.md).
Repair commit `4e734e4a3105b1a590fbb11ab13c3197cb6a9fce` makes terminal
projection explicit while retaining Sanmill's structured state as outcome
authority. It does not change gameplay, search, or training objectives.

## Smoke-002 launch contract

The ignored retry SpecialistDB
`data/specialist_db.sanmill_refereed_fresh_v1.smoke-002.sqlite` is empty,
passes `quick_check`, carries `sector-corrected-v1`, and has no lineage
binding. Its initial SHA-256 is
`5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`.
The proposed retry output did not exist. After the repair and readiness
evidence were committed, the read-only command reviewed was:

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --preflight smoke `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 `
  --start-mode fresh `
  --out-dir out\sanmill-refereed-fresh-v1-smoke-002 `
  --specialist-db data\specialist_db.sanmill_refereed_fresh_v1.smoke-002.sqlite `
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
Sanmill referees both. The owner later authorised exactly one replacement of
`--preflight smoke` with `--launch smoke` plus run ID
`sanmill-refereed-fresh-v1-smoke-002`. That one-run authority is now consumed.

## Readiness result

The historical smoke-001 command returned `ready_for_smoke` immediately before
its launch, but the launch then exposed the terminal-projection defect above.
That historical verdict is superseded and cannot authorize a retry.

The smoke-002 command returned `ready_for_smoke` with no errors or unresolved
decisions at clean commit `d7d6e4dbc22f95c79280ef05c93c8eb8e0a03167`. The
[readiness record](../evidence/sanmill-refereed-fresh-v1-smoke-002-readiness-2026-08-08.md)
owns its identities and claim boundary.

Because that evidence record changed the Git identity, the command was run
once more from clean published commit
`894360d11fc309e5aa58e3289d1c6817831a553c`. It again returned
`ready_for_smoke` with no errors or unresolved decisions. The separately
authorised launch then completed both games, one finite optimiser update, a
verified final checkpoint, and a completed lifecycle chain. The
[smoke-002 result](../evidence/sanmill-refereed-fresh-v1-smoke-002-result-2026-08-08.md)
owns all raw identities and the claim boundary.

The smoke passes only the integration gate. Its single Sanmill-opponent game
and 1,000-node ceiling are not representative throughput evidence. No further
smoke, exact resume, long run, node ladder, or advancement rule is authorised.

The later separately authorised
[Sanmill node-throughput calibration v1](sanmill-node-throughput-calibration-v1.md)
completed all 720 engine-only searches over eight replayed placement,
movement, flying, and compound-capable roots, five node ceilings, nine
repetitions, and cold-process and warm-sequence modes. The
[result record](../evidence/sanmill-node-throughput-calibration-v1-result-2026-08-08.md)
binds its ignored raw output. That calibration did not load a model or
trainer and did not select a node ladder automatically. Its authority is
consumed. A node-ladder decision and any no-update integrated-route probe
remain separate, unauthorised gates.

The subsequent
[node-ladder decision brief](sanmill-node-ladder-v1-decision-brief.md)
recommends retaining the five measured ceilings
`1,000 -> 5,000 -> 25,000 -> 100,000 -> 500,000` as a probe-only resource
matrix. It does not identify five proven strength levels, select an
advancement rule, change the trainer's current one-level preflight, or
authorize execution.

The separate
[no-update integrated-route probe proposal](sanmill-no-update-integrated-route-probe-v1.md)
was subsequently implemented and completed over all 36 immutable scheduled
indices. The route covered all five node ceilings, normal depth-5 and
occasional depth-12 learner routes, both learner colors, and Sanmill-refereed
frozen-target controls. Index 31 exposed a terminal-turn projection mismatch;
the red-first repair and exact reproduction passed before the remaining
controls were run. The final
[closure record](../evidence/sanmill-no-update-frozen-target-controls-continuation-v1-result-2026-08-08.md)
binds the four last controls and the complete evidence chain. This remains
no-update integration evidence, not training or strength evidence.

The first
[Sanmill exact-resume parity smoke](sanmill-refereed-exact-resume-parity-smoke-v1.md)
compared an uninterrupted two-game route with the same schedule split over a
fresh and exact-resume process, including finite periodic A2C updates and an
exact semantic comparison of checkpoint, logs, and SpecialistDB rows. It
stopped before its second segment because an over-broad
preflight policy rejected exact resume within an experiment whose identity
described a fresh lineage. Its
[failure record](../evidence/sanmill-refereed-exact-resume-parity-smoke-v1-failure-2026-08-08.md)
quarantines both completed v1 outputs. Repair `6e820c1` retains the prohibition
on weights-only imports while permitting checkpoint-verified continuation
inside the same lineage. A fully isolated
[v2 parity smoke](sanmill-refereed-exact-resume-parity-smoke-v2.md) is now
isolated with new run IDs, outputs, and databases. It completed successfully
at `b3d049b`: its
[result record](../evidence/sanmill-refereed-exact-resume-parity-smoke-v2-result-2026-08-08.md)
proves exact model, optimiser, RNG, trainer/data-state, log, and SpecialistDB
semantic equality after a real process-boundary resume. No retained training
plan or long-run command was authorized by that parity result alone.

The subsequent
[fixed-resource curriculum decision](sanmill-fixed-resource-curriculum-v1.md)
freezes a deterministic five-level, game-indexed Sanmill node schedule and the
remaining retained-run product envelope. Implementation, focused tests, and
one isolated five-game schedule smoke passed under that decision.
The schedule is resource exposure, not a score-based advancement or strength
claim. The subsequent
[managed long-run contract](sanmill-refereed-managed-long-v1.md) authorizes
creation of the immutable local plan and authorization, final preflight, and
background launch only after every recorded gate passes.

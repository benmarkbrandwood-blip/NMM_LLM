# Sanmill-Preserving Retained Long Run v3

## Status and authority

Status: `frozen_unlaunched_needs_product_authorization`

Experiment ID: `dev-v4-sanmill-preserving-retained-v3-seed58`

Plan ID: `managed-sanmill-preserving-retained-v3-seed58`

This document freezes the next retained Generalist baseline. It authorizes
read-only preparation, creation of one ignored managed plan and one fresh
isolated SpecialistDB, and final readiness checks. It does not itself authorize
training, retry, recovery outside the managed exact-resume contract,
evaluation, promotion, publication, or a larger resource envelope.

## Evidence-based decision

The first Sanmill-refereed retained candidate completed 5,000 games but lost
its frozen held-out comparison: 3 wins, 102 draws, and 23 losses, with a
42.1875% score and a paired interval wholly below zero. A read-only audit found
19 first candidate WDL downgrades. Sixteen formed a mill, and the legacy trainer
awarded those contradictory turns a total immediate bonus of `+4.0` even
though a value-preserving primary action existed in every state.

The `malom-preserving-only` route deterministically removes that bonus from a
value-downgrading complete logical turn while retaining it for a
value-preserving mill. The subsequent three-seed, 500-game learning ablation
was safe and favoured the corrected arm in two of three pairs, but its median
reduction was only 0.396 percentage points and missed the frozen five-point
gate. This run therefore adopts the mode as a correction of contradictory
reward semantics, not as a proven strength improvement.

Two denser alternatives were tested and rejected for this baseline. The fixed
downgrade penalty produced only a 0.614-point median reduction against a
two-point gate. Fixed and normalized policy auxiliaries did not produce a
material, cross-seed functional response; the final no-game audit returned
`stop_gradient_ratio_escalation`. The v3 baseline therefore has no generic
Malom penalty and no Malom policy auxiliary.

Seed 58 was selected before this run and has not appeared in the completed
seed-42-to-57 training experiments. It avoids choosing seed 42 after observing
that seed's unusually favourable short preserving-only result. Because the
seed differs from retained-v2, v3 is not a strict one-factor causal comparison.

## Frozen training configuration

- fresh A2C initialization, seed 58, learning rate `0.0001`, gamma `0.99`,
  entropy coefficient `0.01`, update interval 64, and `batch_games=1`;
- `--mill-bonus-mode malom-preserving-only`;
- `--malom-policy-aux-mode fixed` and
  `--malom-policy-aux-coef 0.0`;
- no generic exact-WDL downgrade penalty;
- Sanmill as complete-history referee for every turn and fixed-node search
  opponent for 40% of games;
- frozen-target opponents for 60% of games, using deterministic argmax over
  the same depth-5 lookahead and SpecialistDB feature route as the learner;
- fixed Sanmill node ceilings `1,000, 5,000, 25,000, 100,000, 500,000` for
  global game intervals `500, 500, 500, 1,000, 2,500`;
- temperature `0.90`, linearly reaching `0.20` at 80% of 5,000 games;
- `max_ply=120`, minimal rollouts, no branches, no recovery, and no wall-clock
  search limit;
- target refresh and checkpoint/log cadence every 50 games;
- no PPO, Sentinel, ValueNet, GapNet, S1A warm-start, imitation mixing, S1B
  refresher, or opening forcing; and
- HumanDB frequencies and outcomes available while its unversioned historical
  Malom fields remain masked.

The 120-logical-ply ceiling is a truncation guard, not a draw rule. The pinned
Sanmill referee separately applies threefold repetition and the 100-moving-ply
no-progress rule. Termination reasons must remain disaggregated.

## Runtime and data boundary

Use the isolated strict Sanmill runtime derived from commit
`a6623f88959f7453594df274fbe1f128af7ff55e`, with
`StrictFailurePolicy=true`,
`StrictRefereeProfile=mif-stable-moving-v1`, one thread, fixed seed, shuffling
off, and no Perfect DB, patch, shallow-search, or random fallback.

The runtime tree is
`17b9b0fd51ee8dac54c0454a6935978a47d19e0c`; its release binary SHA-256
is `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`;
and the repository-defined runtime identity is
`705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea`.
The portable strict-referee semantic digest is
`sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`.

The protocol and rules identities are:

- MIF tag `mif-suite-1.0`, release commit
  `a0a0f21cff5d6fbde045cd1482e416b92e0dc45a`, Suite JCS SHA-256
  `81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f`,
  final-evidence SHA-256
  `2c23983281858386bc66e3adfce52f365c712d9e63a31c53f6a68bd6b2de08e1`,
  and release-manifest SHA-256
  `dde89416bf5251cdc445ebdb9b92a899f58ec3930d1d8077ae26f1cb1a084499`;
- training ruleset `data/rulesets/nmm-training-core@2.json`, semantic digest
  `52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`,
  and document digest
  `1dfdf5777f36866a53a942c1addd21857d3b72eede8ea2bf4fe1beedfbe878f2`;
- corrected Malom manifest identity
  `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747`;
  and
- HumanDB identity
  `8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31`.

HumanDB contributes only empirical frequencies and outcomes. Its unversioned
historical Malom columns remain masked. Any identity drift stops preparation;
it is not permission to rebuild or substitute a database.

The run must bind the final `mif-suite-1.0` release, the current training
ruleset semantic digest, the corrected Malom manifest, the HumanDB identity,
the strict Sanmill runtime identity, and an independent experiment digest in
its run manifest.

Control and output directory:

```text
learned_ai/checkpoints/scaffolded/s_gen_v2_sanmill_refereed/
managed-sanmill-preserving-retained-v3-seed58
```

SpecialistDB:

```text
data/specialist_db.sanmill_preserving_retained_v3.seed58.sqlite
```

Before plan generation, both targets must be absent. The database is copied
once from
`data/specialist_db.mill_bonus_ablation_v1.template.sqlite`. The template and
the fresh copy must be 45,056 bytes with SHA-256
`5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`,
`malom_label_version=sector-corrected-v1`, `quick_check=ok`, zero positions,
zero winning lines, zero preferred plays, and no WAL, SHM, or rollback-journal
sidecar. No prior checkpoint or database may seed this lineage. The first
segment starts at game zero without `--resume` or `--auto-resume-best`.

## Resource envelope and segment gate

- exactly one fresh lineage;
- at most 5,000 completed games;
- at most 12 active supervisor hours;
- 250 games per process segment; and
- one local machine and one CUDA device.

The existing fixed 29-state policy-health audit runs before each completed
segment becomes an accepted resume parent. It must retain direct-lookahead
value preservation `1.0`, candidate preservation at least `0.50`, and mean
best-preserving minus best-downgrading logit margin at least `-0.10`.
The corpus is `docs/experiments/dev-v4-phase-covered-corpus-v1.json` with
SHA-256
`cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e`.

The gate detects catastrophic direction failure only. It is inspected
development evidence, not validation or strength evidence. A failed gate,
non-finite value, checkpoint/database mismatch, referee failure, rules or MIF
identity drift, broken event chain, CUDA failure, or resource limit stops the
run and quarantines the affected segment.

## Managed-plan preparation

Run this only from the final clean published source commit:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py prepare `
  --control-dir learned_ai\checkpoints\scaffolded\s_gen_v2_sanmill_refereed\managed-sanmill-preserving-retained-v3-seed58 `
  --plan-id managed-sanmill-preserving-retained-v3-seed58 `
  --max-wall-hours 12 `
  --objective "fresh Sanmill-refereed retained baseline with value-preserving mill shaping and no policy auxiliary" `
  --experiment-id dev-v4-sanmill-preserving-retained-v3-seed58 `
  --seed 58 `
  --max-games 5000 --segment-games 250 --max-ply 120 `
  --engine-profile sanmill-fixed-resource --self-play-ratio 0.60 `
  --sanmill-node-ladder 1000,5000,25000,100000,500000 `
  --sanmill-stage-games 500,500,500,1000,2500 `
  --mill-bonus-mode malom-preserving-only `
  --malom-policy-aux-mode fixed --malom-policy-aux-coef 0.0 `
  --specialist-db data\specialist_db.sanmill_preserving_retained_v3.seed58.sqlite `
  --policy-health-gate --policy-health-device auto
```

The generated plan and authorization must remain ignored. The authorization
must bind the exact plan SHA-256, objective, 5,000-game and 12-hour ceilings,
fresh lineage, isolated output and database, and the prohibition on automatic
retry, expansion, promotion, publication, or held-out execution.

## Final launch gates

Before `run-authorized`:

1. `HEAD == origin/dev == plan.git_commit`, branch `dev`, clean tracked
   worktree, and reviewed `origin/main` exactly
   `bc46b51e69724e12a8e5f17e3ff696b9f88456d9`;
2. plan, path registry, experiment document, database template, policy-health
   corpus and audit-script identities match;
3. the fresh output path is absent and the new SpecialistDB retains its empty
   identity with no sidecars;
4. the mandatory Malom, DB-teacher, and label-provenance tests plus focused
   manager, trainer, checkpoint, exact-resume, Sanmill-referee, reward-mode,
   and policy-health tests pass;
5. the exact first-segment preflight returns `ready_for_long_run`, no errors,
   no unresolved decisions, and the managed plan's resume-config identity;
6. the MIF Suite, ruleset, corrected Malom, HumanDB and strict Sanmill runtime
   match their frozen identities; and
7. no trainer, supervisor, stale lock, or competing process owns the output,
   database, runtime, or CUDA resource.

Any mismatch stops before training. A missing dependency is never interpreted
as an implicitly disabled component.

## Evaluation and claim boundary

Training curves, W/D/L, node-level outcomes, the 29-state gate, and the last
200 games are diagnostics. They cannot promote the model or trigger a
configuration change.

The prior 64-start held-out corpus was used to diagnose and select this reward
correction. The 29-state phase corpus was also used throughout mechanism
development. Neither may become v3's independent promotion test. Before any
candidate result is inspected for strength, freeze a new candidate-blind,
source-only held-out corpus with complete histories, phase and colour
coverage, a pinned 500,000-node Sanmill baseline, fixed work, rules-compliant
termination, and a preregistered paired decision rule.

This run may establish a clean retained research baseline and training-health
evidence only. It makes no advance claim that the correction improves strength
and grants no publication or promotion authority.

## Diagnostic interpretation contract

Substantive analysis must keep separate:

1. observed facts from raw artefacts and exact identities;
2. falsifiable hypotheses;
3. evidence supporting each hypothesis;
4. counterevidence, missing train/validation evidence, confounders, and
   alternative explanations; and
5. the smallest next validation experiment.

Report raw and rolling curves, seed and hyperparameters, data and ruleset
versions, baseline identity, available ablations, and per-phase, per-opponent,
per-colour, per-node-level, and per-termination metrics. Ordinary RL has no
supervised validation curve; do not relabel a training metric as validation.

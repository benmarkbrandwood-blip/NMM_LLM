# Target-refresh common-anchor policy distribution analysis

Status: `complete_read_only_development_evidence`  
Training-readiness verdict: `needs_decision`

This is a no-training follow-up to the completed attempt-003 target-refresh
diagnostic. It compares every legal action produced by the refresh and
no-refresh candidate checkpoints on the same 64 fixed positions and the same
game-50 feature anchor. It is mechanism evidence, not held-out strength,
promotion, publication, or long-run evidence.

## Immutable evidence

- Analysis implementation commits: `0d117bb4fe3f229d26c0fde822016ca96b356c5e`
  and `9939c4b09a86f556e9a622a82e4e477f261af4ac`.
- Source branch at execution: local `dev` at `9939c4b09a86f556e9a622a82e4e477f261af4ac`.
- Published `origin/dev` at execution: `0320f440b2cb0c3aa4283e99666b08eb92bd5840`.
- Attempt-003 plan identity:
  `8cc192f5152bb15957f5bc7860bce12d6db0200bc51c2d6766752ed4fc54c634`.
- Fixed 64-position corpus SHA-256:
  `cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e`.
- Raw local report:
  `out/target-refresh-common-anchor-policy-analysis-v1/result.json`.
- Raw report SHA-256:
  `02f847658e9f5ecfedb7e585e9e5803d19427dafcaa72904a985e946d30e47ac`.
- Analysis identity:
  `a885711afb6963dfcbeb2d17e85fed065e7379b347b8896029255a2f13ff4023`.

The raw report is ignored local evidence. It contains every legal action's
normalized identity, Malom quality, refresh and no-refresh logit, rank, and
probability at temperatures 1.0 and 0.2. Its internal analysis identity was
independently recomputed successfully after generation.

## Frozen method

The comparison used seeds 64 and 65 and post-anchor optimizer-update deltas
4, 8, 12, and 16. Each seed used its own game-50 anchor. Within each seed, the
refresh and no-refresh anchor model states were byte-identical. The same
feature matrix was then passed to both candidate models at every checkpoint.

The feature route retained the attempt-003 measurement configuration:

- 12-ply lookahead shape with five simulated plies;
- HumanDB frequencies and outcomes only;
- historical unversioned HumanDB Malom fields masked;
- trusted `sector-corrected-v1` Malom tablebase;
- SpecialistDB, Sentinel, ValueNet, and GapNet disabled;
- CPU inference and fail-closed dependency handling;
- no games, optimizer updates, checkpoint writes, or database writes.

HumanDB was opened with SQLite `mode=ro&immutable=1`. The main database,
WAL/SHM observations, and the Malom `std.secval` observation were identical
before and after the successful run. An earlier attempt stopped before
writing a result when a non-immutable read-only connection participated in
normal SQLite SHM lock coordination.

The predeclared primary distribution temperature was 0.2. KL is reported in
both directions, Jensen-Shannon distance is reported in natural-log nats, and
each fixed position has equal weight. Top-1 changes were explicitly excluded
as a standalone material-divergence gate because near-tied logits can change
rank without materially changing the sampling distribution.

## Observed facts

### Complete-corpus trajectory

| Seed | Update delta | Top-1 agreement | Mean JS, temp 0.2 | Mean total variation |
|---:|---:|---:|---:|---:|
| 64 | 4 | 98.44% | 6.61e-12 | 2.65e-6 |
| 64 | 8 | 92.19% | 3.06e-10 | 1.49e-5 |
| 64 | 12 | 87.50% | 8.20e-10 | 2.67e-5 |
| 64 | 16 | 89.06% | 1.97e-9 | 4.60e-5 |
| 65 | 4 | 87.50% | 7.34e-10 | 2.11e-5 |
| 65 | 8 | 75.00% | 5.24e-9 | 5.57e-5 |
| 65 | 12 | 67.19% | 8.93e-9 | 7.51e-5 |
| 65 | 16 | 76.56% | 6.45e-9 | 7.09e-5 |

The distances increase from the first checkpoint, but remain several orders
of magnitude below the predeclared near-identical ceilings. Seed 65 is not
monotonic at the final checkpoint, so the data do not support a claim of
steadily growing behavioural separation.

### Final checkpoint by phase

| Seed | Phase | States | Top-1 agree | Mean JS | Mean TV | Mean absolute Malom-preserving mass delta |
|---:|---|---:|---:|---:|---:|---:|
| 64 | placement | 22 | 90.91% | 2.77e-9 | 5.65e-5 | 2.47e-5 |
| 64 | movement | 21 | 90.48% | 1.93e-9 | 4.78e-5 | 1.87e-5 |
| 64 | flying | 21 | 85.71% | 1.17e-9 | 3.31e-5 | 2.64e-6 |
| 65 | placement | 22 | 90.91% | 8.86e-9 | 8.89e-5 | 6.24e-5 |
| 65 | movement | 21 | 80.95% | 6.20e-9 | 7.28e-5 | 3.50e-5 |
| 65 | flying | 21 | 57.14% | 4.16e-9 | 5.02e-5 | 1.72e-5 |

Seed 65 flying has the largest top-1 instability, but not the largest
distribution distance. Every one of its nine changed flying states, and all
changed states in both seeds, had a top-1 margin below `1e-4` in both models.
The rank changes therefore occur inside an extremely flat action surface.

### Malom direction and policy concentration

All 1,583 legal actions were known to Malom: 1,168 preserved the current WDL
value and 415 downgraded it. The corpus contains 29 critical states with both
preserving and downgrading choices. Both conditions in both seeds selected a
preserving top-1 action in all 64 states.

That apparently perfect top-1 result does not mean the policies are focused.
At the final checkpoint, the all-phase Malom-preserving probability masses
were:

| Seed | Refresh | No refresh | Equal-action uniform reference |
|---:|---:|---:|---:|
| 64 | 0.713531 | 0.713523 | 0.713436 |
| 65 | 0.713591 | 0.713629 | 0.713436 |

The learned mass is only about `0.00009` to `0.00019` above the state-weighted
uniform reference. Mean entropy deficits from the maximum possible entropy
were at most `4.39e-7` nats across the complete final corpus. The policy is
therefore almost exactly uniform even at temperature 0.2; the preserving
top-1 action wins by a microscopic margin.

### Parameter change is real but does not transfer to this action surface

The final refresh/no-refresh policy-head relative L2 differences were 0.262
for seed 64 and 0.289 for seed 65. The corresponding value-head differences
were 1.042 and 1.617. Thus the models did not remain weight-identical. The
counterevidence is important: substantial parameter-space separation and
visible rank changes exist, but they do not produce material probability
separation on this fixed action corpus within 16 post-anchor updates.

The two seeds had different game-50 anchor model identities, yet produced the
same 64-state feature-corpus identity:
`caf9c573c44f83d6b5243677b33ad1a8c116568b22ac42f58d480bf069836cd5`.
All 1,583 actions have exact successor Malom values. The current lookahead
implementation checks Malom immediately after the candidate action and ends
that simulated trajectory when a value is available. Therefore the frozen
anchor model is not consulted deeper in these particular feature rows. This
does not invalidate the common-input comparison, but it means this corpus
tests candidate-policy response to fixed Malom-complete features rather than
anchor-sensitive feature construction.

## Hypotheses and evidence

### Primary hypothesis

Sixteen post-anchor updates are too short for the target-refresh treatment to
produce a material action-distribution effect on the fixed corpus, even though
the two optimization trajectories and value heads have separated.

Supporting evidence:

- JS and total variation remain extremely small in both seeds and all phases;
- Malom-preserving mass differs by at most `6.24e-5` in a phase mean;
- entropy remains effectively maximal;
- all top-1 changes occur at microscopic margins;
- the earlier outcome measurement was floor-limited against Sanmill and the
  fixed anchor, so it could not expose a small policy effect.

Counterevidence:

- policy-head weights have separated materially in parameter space;
- rank displacement and top-1 disagreement increase, especially for seed 65
  movement and flying;
- the fixed corpus is Malom-complete and does not exercise anchor-dependent
  deeper simulation, so a visited-state distribution could behave differently.

### Alternative hypothesis

The prior endogenous frozen-opponent score contrast primarily reflected a
moving opponent/denominator and different visited-state distributions, rather
than a broad change in the candidate's policy probabilities.

The common-input result supports this explanation, but does not prove it.
Only a longer paired run with identical optimizer-consumed transition counts
can distinguish delayed policy separation from a denominator-only effect.

## Gate audit

| Gate | Expected | Observed | Result |
|---|---|---|---|
| Attempt lineage | exact attempt-003 plan identity | exact match | pass |
| Fixed corpus | 64 states; 22/21/21 phases; pinned SHA | exact match | pass |
| Pair anchor | same game-50 model state within seed | exact match | pass |
| Action support | same finite complete legal action set | 1,583/1,583 | pass |
| Malom provenance | trusted manifest; historical HumanDB labels masked | exact match | pass |
| Mutation safety | no checkpoint or database writes | before/after observations equal | pass |
| Near-identical JS | max phase mean <= 5e-4 | max 8.86e-9 | pass |
| Near-identical TV | max phase mean <= 0.02 | max 8.89e-5 | pass |
| Near-identical Malom mass delta | max phase mean <= 0.02 | max 6.24e-5 | pass |
| Strength or promotion evidence | forbidden | none produced | pass |

## Interpretation and next validation

The result classification is `near_identical`. It does not select refresh or
no-refresh and it does not clear long training. The next experiment is a
longer paired mechanism diagnostic whose candidate checkpoints consume
exactly the same number of learner transitions in exact 64-transition update
batches. Game counts and generated-but-unconsumed pending transitions must be
reported separately and must not determine the comparison boundary.

The successor design is recorded in
[`sanmill-target-refresh-equal-transition-diagnostic-v1.md`](../experiments/sanmill-target-refresh-equal-transition-diagnostic-v1.md).

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_common_anchor_policy_distribution.py `
  tests\test_generalist_policy_health.py `
  tests\test_target_refresh_common_anchor_diagnostic.py `
  -q -p no:cacheprovider `
  --basetemp out\pytest-common-anchor-policy-focused

.\.venv\Scripts\python.exe -m pytest `
  tests\test_malom_db.py `
  tests\test_sentinel_db_teacher.py `
  tests\test_malom_label_provenance.py `
  -q -p no:cacheprovider `
  --basetemp out\pytest-common-anchor-policy-malom

ruff check `
  learned_ai\evaluation\common_anchor_policy_distribution.py `
  scripts\analyze_common_anchor_policy_distribution.py `
  tests\test_common_anchor_policy_distribution.py

.\.venv\Scripts\python.exe -B `
  scripts\analyze_common_anchor_policy_distribution.py
```

Results: 19 focused tests passed; 103 Malom/provenance tests and 498 subtests
passed; Ruff and `git diff --check` passed. The successful analysis completed
in 34.2 seconds on CPU.

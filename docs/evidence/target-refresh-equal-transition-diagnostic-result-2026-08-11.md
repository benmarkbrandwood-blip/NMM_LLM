# Target-refresh equal-transition diagnostic result

Status: `complete_inconclusive_late_onset`  
Training-readiness verdict: `needs_successor_design`

This is the immutable result of the three-seed equal-transition mechanism
diagnostic. It does not select `refresh-once` or `no-refresh`, does not measure
held-out playing strength, and does not authorize another diagnostic, held-out
evaluation, model promotion, publication, or long training.

## Evidence identity

- Training source: clean published `dev` commit
  `33a98696994cddf8be0b1ab516a879f52483ef02`.
- Analysis source: clean published `dev` commit
  `0f8e9eb04e9fe046f72fbe47ed0551eeeafc22d4`.
- Frozen plan identity:
  `b14d69db9a33b005c0a19fbb97e7f5b9a16364f1f74390ae85ff3e9d4edabb97`.
- Readiness identity:
  `3b869d706f64332ff917c5ade5ebfbf70c6d9827526f4d4f05070ecebd547cf2`.
- Contract SHA-256:
  `fd39f20cc363b6c34f9b5df81ebc32f642cd1a6561e2f52346d89f01ebb5502f`.
- Readiness SHA-256:
  `29a18b16993b855b9be03b7313730e3b451457509555baee6b792ba47da5aec1`.
- Fixed 64-position phase-corpus SHA-256:
  `cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e`.
- Raw ignored result:
  `out/target-refresh-equal-transition-diagnostic-v1/result.json`.
- Raw result SHA-256:
  `b518849fa4ca3339bf1b3e4842cf5c10f20002088e9b4f4074da3891cb2d2ca3`.
- Result identity:
  `8c6be27feb96d0e50662e299b594140c96b14ec57cf447ecf572fe07757a95dd`.

The first two result-publication attempts stopped before creating a result.
The first compared a valid branch-local resume path with the original prefix
path. The second expected branch-rebind-only descriptor fields to persist in
later runtime checkpoints, although the trainer correctly rebuilds those
descriptors. Commits `e958a3a` and `0f8e9eb` correct only the fail-closed
lineage validation. The successful publisher validates the branch-entry
checkpoint against the shared checkpoint and payload, and then validates each
candidate's configuration, experiment digest, source path, treatment, and
transition counts. It does not change weights, optimizer state, training
data, results, thresholds, or gameplay.

## Frozen design and provenance

Seeds 64, 65, and 66 each ran one shared 50-game prefix. Within each seed,
the two branch-entry envelopes have the same payload SHA-256 and independent,
byte-identical SpecialistDB clones. The only treatment is:

- `refresh-once`: copy the game-50 candidate into the frozen target once;
- `no-refresh`: retain the pre-existing frozen target.

No later target refresh occurs. Every arm consumes exactly 8,192 post-fork
learner transitions in 128 ordered batches of 64. Generated overflow remains
in the pending queue and is not trained or flushed. The comparison checkpoints
are at 1,024, 2,048, 4,096, and 8,192 consumed post-fork transitions.

The common training contract is A2C, fixed learning rate `0.0001`, 60% frozen
target and 40% Sanmill opponents, `max_ply=120`, `sim_ply_depth=5`, no branch
rollouts, and no Sentinel, ValueNet, GapNet, imitation, opening forcing,
recovery, or Malom policy auxiliary. SpecialistDB reads are
`theoretical-only`. Historical unversioned HumanDB Malom columns are masked.

The bound external identities are:

- MIF Suite release `a0a0f21cff5d6fbde045cd1482e416b92e0dc45a`;
- ruleset `nmm-training-core@2`, semantic digest
  `sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`;
- Sanmill commit `a6623f88959f7453594df274fbe1f128af7ff55e`,
  binary SHA-256
  `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`;
- strict-referee semantic digest
  `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`;
- corrected Malom manifest identity
  `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747`;
- HumanDB identity
  `8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31`;
  and
- initially empty, isolated `sector-corrected-v1` SpecialistDB lineage.

The sequence executed 150 prefix games plus 2,233 post-fork arm games, for
2,383 actual games. It performed 65 prefix updates once and 768 post-fork
updates, consumed 49,152 post-fork transitions, and used 1,724.757 process
seconds (`0.4791` active hours). All nine managed plans completed once. There
was no retry, extension, resume, held-out run, promotion, or model publication.

## Predeclared result

The immutable classifier returned `inconclusive_late_onset`.

At 1,024, 2,048, and 4,096 transitions every seed remained inside all three
`near_identical` ceilings. At 8,192 transitions seeds 64 and 65 first crossed
at least one material threshold, but no seed had a material trigger at both
4,096 and 8,192. The contract explicitly classifies final-boundary-only onset
as inconclusive and forbids automatic extension.

| Seed | Boundary | Mean JS | Mean TV | Top-1 disagreement | Max phase mean absolute Malom-mass delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1,024 | 2.31e-9 | 4.58e-5 | 7.81% | 1.65e-5 |
| 64 | 2,048 | 5.29e-7 | 7.05e-4 | 12.50% | 4.85e-4 |
| 64 | 4,096 | 3.29e-5 | 0.00552 | 12.50% | 0.00184 |
| 64 | 8,192 | 0.00833 | 0.09242 | 14.06% | 0.06128 |
| 65 | 1,024 | 8.12e-8 | 2.19e-4 | 26.56% | 2.81e-4 |
| 65 | 2,048 | 3.00e-6 | 0.00131 | 51.56% | 0.00170 |
| 65 | 4,096 | 8.60e-5 | 0.00738 | 60.94% | 0.00928 |
| 65 | 8,192 | 0.00238 | 0.04280 | 68.75% | 0.05226 |
| 66 | 1,024 | 1.84e-8 | 1.35e-4 | 0.00% | 4.99e-5 |
| 66 | 2,048 | 8.01e-7 | 9.00e-4 | 1.56% | 5.56e-4 |
| 66 | 4,096 | 3.59e-5 | 0.00566 | 1.56% | 0.00268 |
| 66 | 8,192 | 0.00131 | 0.03415 | 4.69% | 0.01047 |

All distribution values above use the predeclared primary evaluation
temperature `0.2`. Top-1 disagreement is interpretive only: seed 65 changes
many argmax choices while its full probability distribution remains below the
all-state JS and TV material thresholds.

## Phase-stratified final result

The signed Malom column is `no-refresh - refresh`. Its direction is not
consistent across seeds. Seed 64 favours refresh, seed 65 favours no refresh,
and seed 66 has only small mixed changes.

| Seed | Phase | Top-1 agreement | Mean JS | Mean TV | Mean absolute Malom-mass delta | Signed Malom-mass delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | placement | 77.27% | 0.01179 | 0.11089 | 0.06128 | -0.06128 |
| 64 | movement | 90.48% | 0.00758 | 0.09577 | 0.03096 | -0.03096 |
| 64 | flying | 90.48% | 0.00545 | 0.06972 | 0.00798 | -0.00385 |
| 65 | placement | 36.36% | 0.00396 | 0.06029 | 0.05226 | +0.05226 |
| 65 | movement | 23.81% | 0.00196 | 0.04058 | 0.02416 | +0.02416 |
| 65 | flying | 33.33% | 0.00116 | 0.02671 | 0.01197 | +0.01197 |
| 66 | placement | 86.36% | 0.00136 | 0.03374 | 0.01047 | -0.00146 |
| 66 | movement | 100.00% | 0.00124 | 0.03575 | 0.00443 | -0.00229 |
| 66 | flying | 100.00% | 0.00132 | 0.03299 | 0.00337 | +0.00303 |

The all-phase Malom-preserving probability masses at temperature `0.2` are:

| Seed | Refresh | No refresh | Equal-action reference |
| ---: | ---: | ---: | ---: |
| 64 | 0.76965 | 0.73716 | 0.71344 |
| 65 | 0.73242 | 0.76224 | 0.71344 |
| 66 | 0.78551 | 0.78525 | 0.71344 |

Thus all six final candidates put more mass on Malom-preserving actions than
the equal-action reference on this development corpus. This is evidence of a
non-uniform policy signal, not held-out strength or a consistent advantage for
either target treatment.

## Training curves and outcome strata

All 128 post-fork updates in every arm have finite policy loss, value loss,
entropy, and learning rate. The final 4,096-transition window has:

| Seed | Condition | Mean policy loss | Mean value loss | Mean update entropy |
| ---: | --- | ---: | ---: | ---: |
| 64 | refresh once | 0.3407 | 0.3898 | 2.2142 |
| 64 | no refresh | 0.1784 | 0.4978 | 2.3620 |
| 65 | refresh once | 0.0392 | 0.2122 | 2.0414 |
| 65 | no refresh | 0.2862 | 0.5450 | 2.4892 |
| 66 | refresh once | -0.1524 | 0.0319 | 2.2802 |
| 66 | no refresh | 0.0386 | 0.0881 | 1.9636 |

The directions differ by seed and metric; there is no shared non-finite or
simple loss-divergence failure. These are non-stationary online RL training
losses, not supervised train/validation curves. No supervised validation
split or validation-loss curve exists. The fixed 64-state action analysis is
development evidence and is not a substitute for held-out evaluation.

Training W/D/L/T below separates rules draws from `max-ply` truncations. The
frozen opponent differs by treatment, so these numbers are environment
diagnostics, not a common-baseline strength comparison.

| Seed | Condition | Frozen W/D/L/T | Sanmill W/D/L/T |
| ---: | --- | ---: | ---: |
| 64 | refresh once | 0/5/294/0 | 0/0/218/2 |
| 64 | no refresh | 106/18/4/31 | 0/0/113/2 |
| 65 | refresh once | 0/66/109/25 | 0/0/104/1 |
| 65 | no refresh | 211/5/0/2 | 0/0/114/2 |
| 66 | refresh once | 0/1/336/0 | 0/0/196/0 |
| 66 | no refresh | 7/87/32/49 | 0/0/93/0 |

No arm won or achieved a rules draw against Sanmill. The large no-refresh
scores occur against the older stale frozen target, while refresh arms face
the stronger game-50 candidate copy. They do not show that no refresh is a
stronger policy. Both learner colours show the same qualitative within-seed
pattern, so a simple colour imbalance does not explain it.

## Game-count schedule coupling

Exact transition exposure did not equalize game count. Game length is a
post-treatment mediator, and both temperature and fixed-resource curriculum
are currently indexed by game count. The final checkpoints therefore differ
on additional runtime schedules:

| Seed | Condition | Final game | Temperature | Sanmill level | Sanmill node counts observed |
| ---: | --- | ---: | ---: | ---: | --- |
| 64 | refresh once | 569 | 0.8006 | 2 | 192 at 1k; 28 at 5k |
| 64 | no refresh | 324 | 0.8435 | 1 | 115 at 1k |
| 65 | refresh once | 355 | 0.8381 | 1 | 105 at 1k |
| 65 | no refresh | 384 | 0.8330 | 1 | 116 at 1k |
| 66 | refresh once | 583 | 0.7982 | 2 | 165 at 1k; 31 at 5k |
| 66 | no refresh | 318 | 0.8445 | 1 | 93 at 1k |

All arms were still at level 1 at 4,096 transitions, where every seed was
near-identical. By 8,192 transitions, refresh seeds 64 and 66 had crossed the
500-game boundary and trained against some 5,000-node Sanmill games, while
their controls had not. Their training temperatures also differed by about
0.043 to 0.046. This can amplify or mask a late target-refresh effect.

The counterevidence is seed 65: neither arm changed level, yet its placement
Malom-mass delta crossed the final material threshold. Conversely seed 66 did
change level but crossed no material threshold. Schedule coupling is therefore
a plausible contributor, not a complete explanation.

## Hypotheses, support, and counterevidence

### Hypothesis 1: target refresh has a delayed total effect

Support:

- branch payloads and same-seed prefixes are exact;
- optimizer-consumed transition exposure is equal;
- two seeds cross a material threshold at the final boundary; and
- candidate action distributions are no longer almost uniform at temperature
  `0.2` on the fixed corpus.

Counterevidence:

- no seed has a persistent trigger from 4,096 through 8,192 transitions;
- the Malom direction reverses between seeds 64 and 65;
- seed 66 remains below every material threshold; and
- the external Sanmill training stratum is at a complete outcome floor.

### Hypothesis 2: game-count schedules mediate the late separation

Support:

- the treatment changes game length and therefore final game count;
- temperature and Sanmill node level are both functions of game count;
- two refresh arms reach level 2 while their controls remain at level 1; and
- material separation is absent before those late schedule differences grow.

Counterevidence:

- seed 65 has a final placement trigger without a level difference; and
- seed 66 reaches level 2 without a material trigger.

### Hypothesis 3: no-refresh training wins are denominator effects

Support:

- no-refresh is measured against the deliberately retained stale target;
- refresh is measured against the game-50 candidate copy;
- every arm remains winless against the external Sanmill opponent; and
- prior common-anchor gameplay was floor-limited.

Counterevidence:

- the final candidate distributions do separate for two seeds; and
- all candidates improve Malom-preserving mass over the equal-action reference,
  so the runs contain some real policy learning.

## Decision and next validation

Do not select no refresh merely because it beats the stale target. Do not
select refresh merely because seed 64 assigns more Malom-preserving mass. Do
not extend these consumed authorizations or run held-out evaluation.

The next experiment should be a new, separately frozen schedule-isolation
diagnostic. It should retain exact transition batches and the one-time target
treatment, but index training temperature by consumed post-fork transitions
and hold the Sanmill node level fixed within the paired horizon. It should use
fresh seeds and add a no-update, multi-start development measurement against a
common fixed opponent at the 4,096 and 8,192 boundaries. Training-target,
Sanmill, phase, colour, and termination strata must remain separate. Only that
result can determine whether the late divergence is attributable enough to
the target policy to choose a long-training configuration.

This successor needs a new immutable contract, fresh databases, plans,
preflights, and one bounded parent resource authorization. It must not reuse
the consumed equal-transition grants.

## Verification

Before the successful result publication:

- the new branch-entry positive and tampering regression passed;
- 106 target-refresh, branch, trainer, manager, checkpoint, and exact-update
  tests passed;
- all 12 real seed/boundary candidate pairs passed direct lineage validation;
- Ruff and `git diff --check` passed; and
- the publisher observed identical HumanDB, Malom, and SQLite sidecar state
  before and after analysis.

The expected HumanDB warning states that unversioned historical Malom columns
are disabled. Human frequencies and outcomes remain enabled. It is a trust
gate, not an analysis failure.

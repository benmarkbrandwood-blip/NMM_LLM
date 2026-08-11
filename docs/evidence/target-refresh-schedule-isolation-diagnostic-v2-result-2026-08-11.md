# Target-refresh schedule-isolation v2 result

Status: `complete_no_selection`
Training-readiness verdict: `needs_successor_design`

The six training arms completed once in the original bounded sequence. Its
result publisher then failed closed on uniformly CRLF-framed Windows JSONL
before any development game ran. The separately authorised recovery loaded
the completed checkpoints read-only and completed the frozen 288-game CPU
development analysis once. It did not retrain an arm or write an optimizer,
database, or checkpoint.

The predeclared outcome classifier returned
`no_material_paired_outcome_effect`. The policy-distribution classifier
returned `inconclusive_late_onset`. Neither `refresh-once` nor `no-refresh` is
selected for long training.

## Evidence identity

- Training source: clean published `dev` commit
  `49defb8a79d07a19c035ca6c1f23f266ae5ed2b2`.
- Analysis source: clean published `dev` commit
  `6be069da0bed11664f7f4037d72cd5ba887b6cab`.
- Frozen schedule-isolation contract identity:
  `0580389b3d696df9859ac9e7aea6c4b478bf6e791b7e27bf780d2a6e02db5b0b`.
- Analysis-recovery plan identity:
  `41981a6648afb807c0d30c593670e04b264ef3736d3617def01ca9b11f1bf527`.
- Analysis-recovery plan SHA-256:
  `8bc4e607e68bcc5f730ebffaa6aa98b7718fe55591872f01c4364ca29ba0e2e1`.
- Readiness identity:
  `034ed820d299f4af7c8ff5140568d6db678ec92626576354078fa575081600e9`.
- Readiness SHA-256:
  `d8705e740e33c634d57b0012740e490b33fd7ba211608be89f390ae3dac6d854`.
- Authorisation identity:
  `6e9969257e9ff2b3379466dbcd19698896b047a6b9a3bc5ee6b28b32e85ddf59`.
- Authorisation SHA-256:
  `e0c63023948293aea815caae9032010973b2caac30cfd5fbee891cabec12e26c`.
- Run ID:
  `schedule-isolation-analysis-recovery-v1-2026-08-11-attempt-001`.
- Launch identity:
  `e3091070fb9b8baf0787bb81d1eeb0b6a18bc9775ac580f19bc14bcb6aebec03`.
- Completion identity:
  `9c8235bb5f9f163006799b8ed340653e06d6671b77684783c08b55c5fac683d9`.
- Ignored raw result:
  `out/target-refresh-schedule-isolation-diagnostic-v2/result.json`.
- Raw result SHA-256:
  `34fcf6f2f1354e734b4a58eccaf041c036a5c96518a8a2430d033f05d713edbf`.
- Result identity:
  `a438148928abd8bc1fe6f410c9aebd8e310b5c112d514f2df4d452e26800b65c`.
- Ignored development ledger:
  `out/target-refresh-schedule-isolation-diagnostic-v2/development-outcome-ledger.jsonl`.
- Ledger rows and SHA-256: 288 and
  `cb46cdc6269f33237f1c66c44b9b4bd5b224f0f111a8d8fe7def927bbcc8f90d`.

The executed publisher SHA-256 is
`8ecb62e7f97bd1730471db32486e19d135cbbabbeea4ca7ca393561e1c7f13f6`.
The contract-frozen publisher identity remains separately recorded as
`7e8bd6300dbe0bb45c5c644c63598dd8c7853745469a0416378f36605099c257`.
The intervening allowlisted commits repaired only evidence framing, lineage,
one-shot recovery control, and publisher identity recording.

## Scope and no-write audit

The recovery completed 288 no-update games in 0.04977 active hours. It loaded
candidate checkpoints but performed:

- zero training games;
- zero optimizer updates;
- zero database writes; and
- zero checkpoint writes.

Recorded HumanDB, HumanDB WAL/SHM, and Malom file size and modification-time
observations are identical before and after the run. There is no recovery
failure record. The stderr log contains only the expected warning that
unversioned historical HumanDB Malom columns are masked while human frequency
and outcome data remain enabled.

The recovery authorization is consumed. It forbids an automatic retry,
extension, held-out evaluation, promotion, publication, or long-training
launch.

## Frozen training design and data identity

Seeds 67, 68, and 69 each used one fresh 50-game shared prefix followed by
paired `refresh-once` and `no-refresh` branches. Every branch consumed exactly
8,192 post-fork learner transitions as 128 ordered batches of 64. Paired arms
used byte-identical behavior-temperature exposure and fixed 1,000-node
Sanmill work.

The common configuration was A2C, fixed learning rate `0.0001`, 60% frozen
target and 40% Sanmill training opponents, `max_ply=120`,
`sim_ply_depth=5`, no branch rollouts, and no Sentinel, ValueNet, GapNet,
imitation, opening forcing, recovery, or Malom policy auxiliary.
SpecialistDB reads were `theoretical-only`. Historical HumanDB Malom columns
were masked.

The bound evidence includes:

- fixed phase corpus SHA-256
  `cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e`;
- fixed replay corpus identity
  `ca4b410dd2913933d3ecbd8672fe274ea4a2f8ad42db3f039dabfa52af196aa4`;
- replay audit identity
  `9d4c54270c6e66dd9e16b4dae5af9291b1fea6d1385856650e71119dc4c0dbbf`;
- corrected Malom manifest identity
  `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747`;
  and
- HumanDB identity
  `8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31`.

The original sequence ran 150 shared-prefix games and 2,682 branch games,
for 2,832 distinct training games. All six branches completed once in 0.4824
active hours. No branch was rerun during analysis recovery.

## Observed policy facts

The primary analysis temperature is `0.2`. Mean Jensen-Shannon divergence,
mean total variation, and top-1 disagreement below compare `no-refresh` with
`refresh-once` over the same 64-state placement/movement/flying corpus.

| Seed | Transitions | Mean JS | Mean TV | Top-1 disagreement | Max phase absolute Malom-mass delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 67 | 1,024 | 6.38e-8 | 0.00021 | 3.12% | 0.00023 |
| 67 | 2,048 | 1.06e-6 | 0.00096 | 1.56% | 0.00079 |
| 67 | 4,096 | 0.00040 | 0.01651 | 9.38% | 0.01923 |
| 67 | 8,192 | 0.02288 | 0.11534 | 3.12% | 0.19354 |
| 68 | 1,024 | 1.30e-7 | 0.00027 | 20.31% | 0.00033 |
| 68 | 2,048 | 1.01e-5 | 0.00238 | 42.19% | 0.00304 |
| 68 | 4,096 | 0.00025 | 0.01149 | 29.69% | 0.01567 |
| 68 | 8,192 | 0.01838 | 0.10144 | 21.88% | 0.16258 |
| 69 | 1,024 | 1.10e-8 | 0.00011 | 14.06% | 0.00007 |
| 69 | 2,048 | 1.39e-6 | 0.00107 | 15.62% | 0.00110 |
| 69 | 4,096 | 1.39e-6 | 0.00121 | 6.25% | 0.00056 |
| 69 | 8,192 | 0.00032 | 0.01838 | 6.25% | 0.01489 |

Seeds 67 and 68 cross material JS, TV, and Malom-mass thresholds only at the
final boundary. Seed 69 crosses none. No seed has a material trigger at both
4,096 and 8,192 transitions, so the predeclared policy result is
`inconclusive_late_onset`, not persistent separation.

At 8,192 transitions, the signed Malom probability-mass direction is positive
for `no-refresh` in every seed and phase:

| Seed | Phase | Top-1 agreement | Refresh mass | No-refresh mass | Signed delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 67 | placement | 95.45% | 0.56945 | 0.76299 | +0.19354 |
| 67 | movement | 100.00% | 0.84507 | 0.92551 | +0.08044 |
| 67 | flying | 95.24% | 0.81458 | 0.86064 | +0.04607 |
| 68 | placement | 63.64% | 0.53080 | 0.69337 | +0.16258 |
| 68 | movement | 90.48% | 0.82280 | 0.90136 | +0.07856 |
| 68 | flying | 80.95% | 0.80841 | 0.84082 | +0.03241 |
| 69 | placement | 81.82% | 0.62187 | 0.63676 | +0.01489 |
| 69 | movement | 100.00% | 0.87165 | 0.87877 | +0.00712 |
| 69 | flying | 100.00% | 0.82269 | 0.82635 | +0.00366 |

This is a real late policy signal in seeds 67 and 68, but it is neither
persistent nor a strength result. It cannot by itself select the stale-target
condition.

## Development outcomes

The common 1,000-node Sanmill anchor is independent of the two training
targets. W/D/L below is from the candidate's perspective and treats rules
draws as one half point. `T` is max-ply truncation and is reported separately.

| Boundary | Condition | W/D/L/T | Score rate |
| ---: | --- | ---: | ---: |
| 4,096 | no refresh | 3/2/67/1 | 5.56% |
| 4,096 | refresh once | 3/2/67/1 | 5.56% |
| 8,192 | no refresh | 6/4/62/0 | 11.11% |
| 8,192 | refresh once | 3/4/65/1 | 6.94% |
| both | no refresh | 9/6/129/1 | 8.33% |
| both | refresh once | 6/6/132/2 | 6.25% |

The predeclared contrast is `no-refresh - refresh-once`. It is exactly zero at
4,096 and `+0.04167` at 8,192, below the required aggregate effect
`0.08333`. All 72 early pairs are identical. At the final boundary, 63 of 72
pairs are identical, six favour no refresh, and three favour refresh. All five
safety and persistence gates are false, and no long-run condition is selected.

The combined phase and colour strata show a severe measurement floor:

| Stratum | No-refresh W/D/L | No-refresh score | Refresh W/D/L | Refresh score |
| --- | ---: | ---: | ---: | ---: |
| placement | 0/2/46 | 2.08% | 0/1/47 | 1.04% |
| movement | 3/3/42 | 9.38% | 2/5/41 | 9.38% |
| flying | 6/1/41 | 13.54% | 4/0/44 | 8.33% |
| candidate White | 0/2/70 | 1.39% | 0/1/71 | 0.69% |
| candidate Black | 9/4/59 | 15.28% | 6/5/61 | 11.81% |

The outcome measurement therefore supplies useful counterevidence but has
little power to distinguish two weak candidates, especially when the
candidate plays White.

## Training curves, baselines, and outcome semantics

All 128 post-fork update rows in all six arms have finite policy loss, value
loss, entropy, learning rate, and behavior temperature. The final 4,096
transition window is:

| Seed | Condition | Mean policy loss | Mean value loss | Mean entropy |
| ---: | --- | ---: | ---: | ---: |
| 67 | no refresh | 0.0138 | 0.1401 | 2.1891 |
| 67 | refresh once | -0.1208 | 0.0545 | 2.2360 |
| 68 | no refresh | -0.0155 | 0.1928 | 2.0883 |
| 68 | refresh once | -0.1429 | 0.0236 | 2.2071 |
| 69 | no refresh | -0.1392 | 0.0586 | 2.2078 |
| 69 | refresh once | -0.1288 | 0.0443 | 2.1915 |

Every arm uses fixed learning rate `0.0001`, and paired mean behavior
temperature is byte-equal (`0.85130` over this final window). Loss directions
are not consistent across seeds. These are non-stationary online RL losses,
not supervised train/validation curves. No supervised validation split or
validation-loss curve exists. The 64-state corpus and 288 common-anchor games
are development evidence, not held-out evidence.

Sanmill's terminal reason names the losing condition, for example
`lose_fewer_than_three`; it does not state whether the learner won. The
trainer separately maps Sanmill's authoritative `winner` to learner-relative
`WIN_REWARD`, `DRAW_SHORT`, or `LOSS_REWARD`. Training W/D/L must therefore be
derived from `outcome`, not from the terminal-reason prefix.

Using that authoritative field, the training strata were:

| Seed | Condition | Frozen W/D/L/T | Sanmill W/D/L/T |
| ---: | --- | ---: | ---: |
| 67 | no refresh | 8/14/194/6 | 0/0/164/3 |
| 67 | refresh once | 0/1/288/0 | 0/0/217/4 |
| 68 | no refresh | 13/97/35/33 | 0/0/98/1 |
| 68 | refresh once | 0/2/312/0 | 0/0/197/1 |
| 69 | no refresh | 0/27/259/0 | 0/0/179/3 |
| 69 | refresh once | 0/2/325/0 | 0/0/197/2 |

The no-refresh condition sometimes beats its deliberately stale frozen
target. Neither condition wins a Sanmill training game. The stale target is
not a common baseline, so these rows cannot establish that no refresh is
stronger.

## Hypotheses, support, and counterevidence

### Hypothesis 1: one target refresh creates a delayed policy effect

Supporting evidence:

- same-seed branches share an exact fork;
- consumed transitions, update batches, behavior temperatures, and Sanmill
  node work are paired exactly; and
- seeds 67 and 68 show material final-boundary distribution separation.

Counterevidence:

- separation is absent at the required confirmation boundary;
- seed 69 remains below every material threshold; and
- the common-anchor outcome effect is below the frozen decision threshold.

### Hypothesis 2: no refresh improves Malom-preserving behavior

Supporting evidence:

- all nine final seed/phase signed Malom-mass differences favour no refresh;
  and
- the two materially separated seeds have the largest placement effects.

Counterevidence:

- the direction is observed only clearly at the final boundary;
- no-refresh trains against an easier stale frozen target; and
- Malom-preserving probability on a development corpus is not playing
  strength or trap-conversion evidence.

### Hypothesis 3: the fixed Sanmill anchor masks a real relative difference

Supporting evidence:

- both candidates score below 12% at the final boundary;
- candidate White scores below 1.4% overall; and
- 135 of 144 paired outcomes are identical at the two measured boundaries.

Counterevidence:

- no refresh has a small positive final aggregate effect; and
- flying positions produce more decisive separation than placement or
  movement.

## Decision and next validation experiment

Do not choose either target-refresh policy for a retained or long run. Do not
repeat the six-arm training or this consumed recovery. Do not start held-out
evaluation from this result.

The next bounded diagnostic should use the already completed 8,192-transition
checkpoints in direct `refresh-once` versus `no-refresh` play. Sanmill should
act only as strict portable referee, not as the playing opponent. The design
should use fixed audited placement, movement, and flying replay starts, swap
candidate colours, use common random numbers, and keep `max_ply=120`. It must
remain no-update and report seed, phase, colour, termination, and paired
outcomes separately.

This direct comparison tests whether the late policy-distribution difference
has a gameplay consequence without the present 1,000-node-anchor floor. It is
still development mechanism evidence, not held-out strength, promotion, or
publication evidence. It requires a new immutable contract, readiness record,
and one bounded product authorization before launch.

## Verification

Before this evidence was committed:

- the one-shot recovery completed once with 288 ledger rows;
- all input and output identities above were recomputed;
- the trainer's learner-relative outcome mapping was inspected directly;
- HumanDB, Malom, and sidecar observations were byte-metadata identical before
  and after analysis; and
- 39 focused schedule-isolation tests passed;
- 103 mandatory Malom/provenance tests and 498 subtests passed;
- Ruff passed on the affected analysis and test paths; and
- `git diff --check` passed.

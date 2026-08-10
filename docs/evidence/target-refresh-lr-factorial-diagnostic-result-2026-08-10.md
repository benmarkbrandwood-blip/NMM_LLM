# Target-refresh and learning-rate factorial diagnostic result

Status: completed development mechanism diagnostic. This is not held-out
validation, playing-strength evidence, model promotion, publication, or
authority for a retained or long training run.

## Evidence identity

- Training and analysis source: `d8c680b57498c822971da015e2e07c625fa359a6`
  on clean, published `dev`.
- Frozen experiment identity:
  `94f6381a40ab86401cb0e957677dd3a21dde01ed9ffd4c69b3fa252b21787e58`.
- Readiness identity:
  `893c38fa8f4ba82cea74136a558e62a9634959856fe05bedd51ec0c6bc894f09`.
- Readiness SHA-256:
  `5dfa976579955309563157e2ada342f30be6bf03176f3908dd7dd0d189e9cca3`.
- Raw result identity:
  `fb6ca7f58212a0598948e3205ff57d84f42bc56576dc0f7decc7c5337e36158c`.
- Raw result SHA-256:
  `6ea82762d22d03de68d25defcaba4e9efd1b48a953a4b36ff9061c162cdf1e59`.
- Raw result location:
  `out/target-refresh-lr-factorial-diagnostic-v1/result.json` (ignored,
  machine-local immutable evidence).

All eight authorized arms completed exactly 100 games, for 800 games and 256
optimizer updates in total. Their segment processes consumed 1,052.44 seconds
(`0.2923` active hours), below both frozen resource limits. All eight manager
states are `completed`, all eight checkpoint/lifecycle audits passed, and all
eight 29-state policy-health gates passed. No arm was retried, resumed,
extended, promoted, published, or followed by a held-out run.

## Frozen design and provenance

This was a two-seed, paired 2x2 ablation. The historical control was a hard
copy of the learner into the frozen training opponent every 50 games plus the
historical adaptive learning-rate rule. The factors were:

- frozen-opponent refresh at game 50 or no refresh inside the 100-game arm;
- adaptive learning rate, which changed `0.0001` to `0.00005` after game 50,
  or fixed learning rate `0.0001`.

Every arm used fresh weights, A2C, `batch_games=1`, a 60% frozen-model and 40%
Sanmill opponent schedule, Sanmill level 1 at 1,000 nodes, `max_ply=120`, no
branch rollouts, and no Sentinel, ValueNet, GapNet, recovery, imitation,
opening forcing, empirical SpecialistDB reads, or Malom policy auxiliary.
Only the preregistered games 51-100 score against the frozen-model opponent
contributed to the factorial decision.

The rules and data identities remained fixed:

- MIF Suite 1.0 release `a0a0f21c`, suite JCS identity `81a5feab...ae6f`;
- ruleset semantic digest `sha256:52f6ad24...31f6a`;
- Sanmill runtime commit `a6623f88`, binary SHA-256 `5fbf3cba...619`, and
  strict-referee semantic digest `sha256:1b2b88cf...b1a94`;
- corrected Malom manifest identity `f4c52b00...8747`;
- eight isolated, initially empty `sector-corrected-v1` SpecialistDB copies
  from template SHA-256 `5a5d8eb1...540d`; and
- HumanDB identity `8662e333...4d31`, used only for empirical frequency and
  outcome information. Its unversioned historical Malom columns remained
  masked. The repeated warning about those columns was therefore expected
  enforcement, not a run anomaly.

## Observed facts

The same-seed arms were exactly paired through game 50. Their canonical game
ledger hashes were:

- seed 64: `2aedc2c482cb92372ef71f16f08c6042f9522b07369f8ee239402ea0ecf8ae7a`;
- seed 65: `72410c5f4e965d907375c882e6e403dfe7715d9896b779a9f1ed1cf14ef8c1e4`.

In every arm, game 50 had target age 50 and learning rate `0.0001`. At game
51, refresh arms had target age 1 and no-refresh arms had target age 51;
adaptive arms used `0.00005` and fixed arms retained `0.0001`. This proves
that both interventions engaged at the intended boundary without a
pre-boundary schedule difference.

The primary post-boundary observations were:

| Seed | Condition | Frozen W/D/L | Frozen score | Sanmill W/D/L | Sanmill score |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | refresh + adaptive | 0/0/28 | 0.0000 | 0/1/21 | 0.0227 |
| 64 | refresh + fixed | 0/0/28 | 0.0000 | 0/1/21 | 0.0227 |
| 64 | no refresh + adaptive | 17/10/1 | 0.7857 | 0/1/21 | 0.0227 |
| 64 | no refresh + fixed | 17/10/1 | 0.7857 | 0/1/21 | 0.0227 |
| 65 | refresh + adaptive | 0/14/18 | 0.2188 | 0/0/18 | 0.0000 |
| 65 | refresh + fixed | 0/14/18 | 0.2188 | 0/0/18 | 0.0000 |
| 65 | no refresh + adaptive | 31/1/0 | 0.9844 | 0/0/18 | 0.0000 |
| 65 | no refresh + fixed | 31/1/0 | 0.9844 | 0/0/18 | 0.0000 |

Here, score is `(wins + 0.5 * draws) / games`. The preregistered signed
refresh contrast is implemented as `no refresh - refresh`; it was `+0.7857`
for seed 64 and `+0.7656` for seed 65, with median `+0.7757`, above the frozen
material threshold `0.10`. The result schema calls this a supported
`target_refresh` factor signal. The positive direction means that *not
refreshing* scored higher against the frozen opponent; it must not be read as
evidence that refresh improved strength.

The fixed-minus-adaptive learning-rate contrast was exactly `0.0` in both
refresh conditions and both no-refresh conditions for both seeds. The
difference-in-differences interaction was also `0.0`. Consequently this short
diagnostic did not support a learning-rate or interaction effect. The post-
boundary parameter updates are not byte-identical, so this does not prove the
two learning-rate policies are generally equivalent.

Several disaggregated observations constrain interpretation:

- The learner recorded no post-boundary win against Sanmill in any arm. The
  Sanmill outcomes were invariant within each seed and do not support an
  external-strength benefit for either factor.
- Neither colour rescued the refresh arms. Seed 64 scored `0.0000` as White
  and `0.0227` as Black across all post-boundary opponents; seed 65 scored
  `0.1136` as White and `0.1607` as Black. These are schedule-mixed training
  statistics, not colour-balanced evaluation estimates.
- Refreshed arms mostly ended in losses by material or mobility. No-refresh
  wins were material wins against the old frozen target; their Sanmill games
  remained losses by mobility. Threefold repetition and max-ply truncation
  account for the draws and are separately identified in the raw result.
- Mean post-boundary game length changed with the treatment: seed 64 was
  `32.62` plies with refresh versus `59.16` without; seed 65 was `44.02`
  versus `48.48`. The resulting optimizer-update counts were 30 versus 37 for
  seed 64 and 29 versus 32 for seed 65. A fixed game count therefore did not
  hold gradient-step exposure constant; this is a post-treatment mediator and
  a limit on mechanistic interpretation.
- Refreshed arms had higher mean Malom-preserving rates (`0.911` and `0.926`)
  than no-refresh arms (`0.738` and `0.798`) while scoring worse against their
  training opponents. This metric depends on the encountered position
  distribution and cannot by itself establish strength.
- Post-boundary mean policy entropy remained finite, between about `2.26` and
  `2.49`; chosen probability remained between about `0.12` and `0.17`. The
  policy-health audits also remained green. The result is not a non-finite or
  simple policy-collapse event.

This ordinary RL diagnostic has raw policy/value loss and update-entropy
curves, but no supervised train/validation split or validation curve. The raw
result states that limitation explicitly. It contains the complete curves and
per-seed, per-condition, pre/post-window, opponent-source, learner-colour, and
termination-reason records; the summary above does not pool them into a
strength claim.

## Hypotheses

1. The large discontinuity is primarily an opponent-identity reset. Before
   game 51, the frozen opponent is the initial random snapshot. Refresh copies
   the trained learner into that opponent, which then selects legal moves by
   deterministic maximum logit while the learner continues sampling from its
   temperature-scaled policy. The post-refresh opponent is therefore a much
   harder and behaviorally different reference.
2. A hard refresh may also create a genuine curriculum shock that changes
   subsequent learning. The current primary metric cannot separate this from
   the fact that it evaluates against the newly strengthened opponent itself.
3. The historical learning-rate reduction was not the cause of the observed
   W/D/L discontinuity over games 51-100. Its longer-run effect remains
   unknown because the window was short and action outcomes were insensitive
   to the small parameter differences.
4. Unequal game lengths and update counts may amplify later differences even
   when the only direct intervention is opponent refresh.

## Supporting evidence

- Both independent seeds show the same large signed refresh contrast, and all
  four same-seed arms are byte-identical before the intervention.
- Fixed and adaptive learning-rate arms have identical W/D/L within each
  refresh condition, whereas changing refresh status changes the frozen-
  opponent outcomes in both seeds.
- The code refreshes the target before scheduling game 51 and the target uses
  deterministic argmax selection. The logged target ages prove that this path
  executed.
- All identities, finite-value checks, controller ledgers, checkpoints,
  isolated databases, referee pins, and policy-health gates passed.

## Counterevidence and limits

- The primary opponent changes when the refresh factor changes. A score
  against the old random target and a score against the current learner copy
  are not measurements against a common-strength baseline.
- Sanmill outcomes do not distinguish any condition: the learner remained
  winless against the 1,000-node external opponent. The experiment therefore
  cannot show whether refresh helps or hurts transferable learning.
- Only two seeds and a 50-game post-boundary window were used. This is enough
  for the preregistered mechanism signal, not for a robust long-run setting
  selection.
- Game-count pairing does not equal optimizer-step pairing after the boundary.
- No held-out corpus, common fixed checkpoint opponent, deeper Sanmill level,
  trap corpus, or formal playing-strength baseline was evaluated.

## Next validation and decision

Do not disable target refresh, select a learning-rate mode, run held-out games,
or start long training from this result alone. The immediate design task is a
separately frozen successor probe that distinguishes the *training opponent*
from the *measurement anchor*:

1. preserve a common, never-refreshed anchor (or fixed anchor set) solely for
   comparable no-update measurements;
2. vary the training-target update policy without changing that anchor;
3. compare checkpoints at equal optimizer-step exposure as well as equal game
   count;
4. retain the fixed-node Sanmill stratum as an external reference, with its
   results reported separately; and
5. preregister whether the successor tests hard refresh, a lagged snapshot
   pool, or a soft/EMA target before viewing its outcomes.

This design work is permitted by the result's claim boundary, but execution
requires a new immutable plan, readiness review, resource envelope, and
explicit product authorization. Current readiness verdict: `needs_decision`.

## Verification

After publishing the ignored raw result once:

- `tests/test_target_refresh_lr_factorial_diagnostic.py`: `9 passed`;
- mandatory Malom/DB-teacher/label-provenance group: `103 passed, 498
  subtests passed`.

The first focused invocation encountered an inaccessible system pytest temp
directory after seven tests had passed. It was rerun without changing code or
assertions using a repository-local `--basetemp` and `-p no:cacheprovider`;
all nine tests then passed. No full-suite claim is made.

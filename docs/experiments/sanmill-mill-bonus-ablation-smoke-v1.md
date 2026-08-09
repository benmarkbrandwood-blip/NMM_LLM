# Sanmill Mill-Bonus Ablation Smoke v1

## Status and question

Status: `designed_unlaunched_needs_product_authorization`

Experiment family:
`dev-v4-sanmill-mill-bonus-ablation-smoke-v1`

The retained-v2 policy finished its managed run but lost the frozen held-out
comparison. A read-only transition audit then found 19 first WDL downgrades;
16 were Mill-forming turns with compulsory removal. The full-oracle audit
found a value-preserving alternative at every state, and the historical
trainer had awarded `+0.25` to all 16 contradictory Mill formations. The
no-update probe proved that the new `malom-preserving-only` mode removes those
four total reward units while retaining the bonus for a value-preserving
Mill.

Those facts establish a plausible mechanism, not a learning effect. This
bounded experiment asks whether the corrected reward changes learning in the
intended direction on fresh policies.

## One-factor paired design

Seeds 42, 43 and 44 each have two arms. Every arm starts from fresh random
weights and a byte-identical empty `sector-corrected-v1` SpecialistDB. Same-seed
pairs share the generated learner-colour, opponent-source, retry-ply and
per-game Torch schedules. The only permitted learning-semantic difference is:

- control: `--mill-bonus-mode legacy-unconditional`;
- treatment: `--mill-bonus-mode malom-preserving-only`.

Experiment ID, plan ID, output directory and writable database path must also
differ for isolation, but they do not change gameplay. Runs are sequential.
The order alternates within seed blocks: legacy first for 42, corrected first
for 43, and legacy first for 44. No imported, retained-v2, maintainer or smoke
checkpoint may seed an arm.

Each arm runs exactly one 500-game segment on the unchanged 5,000-game global
temperature and resource schedule. It therefore remains at the measured
1,000-node Sanmill level and ends at the first curriculum boundary. This
mirrors the earlier corrected-learning smoke and avoids introducing a second
change while testing the reward gate.

## Frozen training contract

- A2C, learning rate `0.0001`, gamma `0.99`, entropy coefficient `0.01`, and
  update interval 64 steps;
- `batch_games=1`, checkpoint/log cadence 50 games, and target refresh every
  50 games;
- Sanmill authoritative referee for all turns and fixed-node opponent for the
  40% search stratum; frozen target for 60%;
- five-level global ladder `1,000, 5,000, 25,000, 100,000, 500,000` with stage
  lengths `500, 500, 500, 1,000, 2,500` games;
- `max_ply=120`, `sim_ply_depth=5`, minimal rollouts, and no branches or
  recovery;
- temperature `0.90` to `0.20` over 80% of the 5,000-game schedule;
- no PPO, Sentinel, ValueNet, GapNet, imitation warm-start or mixing, S1B
  refresher, or opening forcing; and
- masked HumanDB Malom columns, corrected live Malom, and one isolated
  SpecialistDB per arm.

The machine-readable contract freezes all six arm identities, schedule counts,
data and runtime identities, metrics, stop rules and resource ceilings:
[`sanmill-mill-bonus-ablation-smoke-v1.json`](sanmill-mill-bonus-ablation-smoke-v1.json).

## Evidence and analysis

Training evidence must show raw rows and 50-game rolling curves. The primary
measure is calculated over games 301–500 for each arm:

```text
known Mill-forming actions with exact Malom WDL downgrade
---------------------------------------------------------
all known Mill-forming actions
```

The full 500-game value is a sensitivity result. Every numerator and
denominator is reported; zero-support arms are inconclusive rather than zero.
Results are shown per seed before median, range and outliers. The corrected
mode is supported for a longer experiment only when all three corrected arms
pass safety, at least two of three paired differences favour it, and the
median absolute reduction is at least five percentage points.

The 29-state phase corpus is an inspected development safety diagnostic. It
must retain direct-signal preserving rate 1.0, candidate preserving rate at
least 0.50, and preserving-minus-downgrading logit margin at least -0.10. It
is not renamed as a validation set or held-out strength evidence. Ordinary RL
training has no supervised validation curve here. Any later playing-strength
claim requires a newly frozen evaluation corpus that was not used to diagnose
or select this reward change.

Also report policy/value losses by optimizer update; entropy, temperature,
chosen probability and top-1 curves; reward components; all-action Malom
quality; and W/D/L separated by opponent, colour, node level and termination.
W/D/L is a training diagnostic only, not the selection metric.

## Resource and authority boundary

The complete envelope is at most 3,000 games, six sequential processes and six
active wall hours. The deterministic schedules contain 1,220
Sanmill-opponent games across all six arms and at most 73,200,000 requested
Sanmill nodes under the 120-ply ceiling.

No arm is authorized yet. After the source and contract are published, six
ignored managed plans must be generated at one clean `dev == origin/dev` tip,
the exact empty database template copied to each arm, and a read-only
cross-plan audit must prove that only the allowlisted fields differ. Product
authorization may then approve the one six-arm resource envelope. A result,
timeout or promising curve never authorizes a second segment, a 5,000-game
continuation, promotion or publication.

## Main-branch boundary

`origin/main` was reviewed through `bc46b51e69724e12a8e5f17e3ff696b9f88456d9`.
Its v2c recovery, repetition, GapNet, Sentinel, optimizer and UI changes are
not cherry-picked. They alter multiple variables and are observation-driven
experiments in another trainer lineage; including them would destroy this
one-factor comparison.

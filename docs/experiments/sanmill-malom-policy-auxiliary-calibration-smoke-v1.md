# Sanmill Malom Policy-Auxiliary Calibration Smoke v1

## Status and decision boundary

Status: `designed_unlaunched_needs_product_authorization`

Experiment family:
`sanmill-malom-policy-auxiliary-calibration-smoke-v1`

This is a bounded optimizer-integration and coefficient-scale calibration. It
is not an effectiveness comparison, held-out validation, playing-strength
test, promotion decision, retained-run decision, or long training run. Its
machine-readable contract is
[`sanmill-malom-policy-auxiliary-calibration-smoke-v1.json`](sanmill-malom-policy-auxiliary-calibration-smoke-v1.json).

No arm is authorized by this document. Technical preparation must first occur
at one clean published `dev == origin/dev` tip. A later product authorization
may approve one four-arm envelope only; it cannot authorize an extension,
continuation, promotion, or publication.

## Observed facts

The preceding three-seed downgrade-penalty experiment completed safely but was
inconclusive. Its primary tail-rate median improvement was about 0.61
percentage points, below the frozen two-point threshold. Repetition draws rose
in all three treatment pairs, and the Sanmill stratum changed little. That is
counterevidence against increasing reward-only pressure.

The successor implementation instead labels every legal complete turn by its
exact Malom WDL change and applies a preserving-set policy objective:

```text
-log P(any legal action that preserves the root WDL)
```

It does not rank actions tied inside the preserving set. States where every
legal action preserves WDL contribute no preference gradient. Malom remains a
training label source; no Malom feature is added to inference.

The immutable three-seed, 64-state development probe established the following
facts for each seed:

- 1,583 complete legal actions received exact labels;
- 1,168 actions preserved WDL and 415 downgraded it;
- 29 states were informative and 35 were all-safe;
- informative states covered placement, movement, and flying;
- the auxiliary gradient was finite and its descent direction increased
  preserving probability analytically; and
- the original model, HumanDB, SpecialistDB, Malom anchor, and tracked
  worktree were unchanged.

The direct float32 parameter step at the production learning rate was below
observable probability resolution. That is explicit counterevidence: the
probe proves label coverage and gradient direction, but it does not establish
an effective Adam coefficient, an optimizer trajectory, validation
improvement, or playing strength.

## Hypothesis and four-arm design

The falsifiable hypothesis is that one of the tested coefficients integrates
the exact-WDL preserving-set objective strongly enough to produce detectable
fixed-state policy movement without dominating ordinary A2C learning,
collapsing exploration, or mainly increasing repetition draws.

All four arms use seed 51, fresh random weights, the same deterministic game
schedule, `malom-preserving-only` reward shaping, and byte-identical isolated
empty `sector-corrected-v1` SpecialistDB copies. The sole permitted
learning-semantic difference is `--malom-policy-aux-coef`:

1. control: `0.00`;
2. low: `0.03`;
3. medium: `0.10`;
4. high: `0.30`.

They run sequentially in that order. Each arm is exactly one 100-game segment,
with at most 0.5 active hours. All retain the global 5,000-game schedule so
temperature and curriculum progress have the same meaning as the intended
successor, but 100 games keep every arm at the observed 1,000-node Sanmill
level. The complete envelope is at most 400 games, two active hours, and
11,520,000 requested Sanmill search nodes. Seed 51 fixes 48 Sanmill-opponent
games per arm: 30 with the learner black and 18 with the learner white.

## Frozen training variables

- A2C, learning rate `0.0001`, gamma `0.99`, entropy coefficient `0.01`, and
  updates every 64 learner steps;
- `batch_games=1`, checkpoint/log cadence 50 games, target refresh every 50
  games, 60% frozen-target and 40% Sanmill-opponent scheduling;
- fixed Sanmill ladder `1,000, 5,000, 25,000, 100,000, 500,000` with global
  stage lengths `500, 500, 500, 1,000, 2,500` games;
- `max_ply=120`, simulation depth 5, minimal rollouts, no branches, and no
  recovery;
- temperature `0.90` to `0.20` over the first 80% of the global schedule;
- no PPO, Sentinel, ValueNet, GapNet, S1A warm start, imitation mixing, S1B
  refresher, or opening forcing; and
- Sanmill as the authoritative referee on every game, Sanmill fixed-node
  search only in the scheduled search stratum, corrected live Malom, and
  masked historical HumanDB Malom columns.

The frozen target is a same-arm snapshot of the fresh learner, not a model
from `main` or an earlier retained run.

## Evidence to inspect

The following are observations, not predictions:

- raw and 50-game policy loss, value loss, and entropy curves;
- raw auxiliary loss, labelled/informative support, and preserving mass by
  optimizer update;
- scaled auxiliary loss, defined as coefficient times auxiliary loss, beside
  absolute ordinary policy loss at the same update;
- temperature, chosen probability, policy top-1, and heuristic top-1;
- exact selected-action downgrade rates by phase and opponent source;
- W/D/L by opponent source and learner colour; and
- termination reasons, especially repetition and max-ply truncation.

The fixed 29-informative-state development diagnostic must separately compare
the resulting checkpoints. It is inspected development data, not held-out
validation. The diagnostic retains its existing direct-signal, preserving-rate
and logit-margin safety gates and additionally supplies the fixed-state
preserving-mass comparison needed for coefficient calibration.

There is no ordinary supervised train/validation split in this RL smoke.
Training W/D/L and the inspected development corpus therefore cannot support a
generalization or strength claim. A later effectiveness experiment must use
multiple fresh seeds, and any strength or promotion claim still requires a
newly frozen held-out evaluation that was not used to choose the coefficient.

## Selection and stop rules

An arm is technically eligible only if it has finite losses and logits,
complete exact labels wherever the auxiliary is active, valid identities,
clean checkpoints, passing referee/database/policy-health gates, no policy
collapse, and no material entropy or repetition-draw safety regression.

Among eligible nonzero arms, select the lowest coefficient that produces a
detectable fixed-state preserving-mass improvement while its median scaled
auxiliary loss is no greater than the median absolute A2C policy loss. If none
does, stop and redesign normalization. This calibration does not select by
training-game W/D/L.

Any identity drift, non-finite value, incomplete exact action labelling,
SpecialistDB or rules mismatch, Sanmill error, checkpoint corruption,
policy-health failure, or resource ceiling stops the entire four-arm sequence.
There is no result-based extension, automatic promotion, or automatic second
segment.

## Fail-closed preparation

After the design and readiness implementation are published, the audit entry
point requires a clean `dev == origin/dev`, the reviewed `origin/main` tip, the
exact tracked gradient-evidence manifest, its ignored raw probe, the closed
database template, and the frozen MIF, rules, and Sanmill identities:

```powershell
.\.venv\Scripts\python.exe `
  scripts\prepare_malom_policy_auxiliary_calibration.py
```

Adding `--prepare` copies four isolated databases, creates four managed plans,
runs a read-only long-run preflight for each proposed first segment, audits
one-factor equivalence, and writes one ignored readiness report:

```powershell
.\.venv\Scripts\python.exe `
  scripts\prepare_malom_policy_auxiliary_calibration.py --prepare
```

Every technical preflight must contain zero errors and only the missing product
authorization decision. The resulting state is
`ready_for_product_authorization` with `launch_authorized=false`. Preparation
does not create `authorization.json`, a segment directory, a checkpoint, or a
training process. It refuses existing targets instead of overwriting or
resuming them.

## Main-branch boundary

`origin/main` was reviewed through
`bc46b51e69724e12a8e5f17e3ff696b9f88456d9`. Its recovery, auxiliary-network,
curriculum, and exploratory trainer changes are excluded because they alter
multiple factors. No `main` checkpoint or training artifact enters this fresh
lineage.

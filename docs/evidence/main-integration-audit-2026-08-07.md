# Maintainer Main Integration Audit — 7 August 2026

## Scope and Claim Boundary

This is a read-only audit of `origin/main` after the last integrated maintainer
tip `67af016`. The remote was fetched on 7 August 2026 and the reviewed tip was
`bc46b51e69724e12a8e5f17e3ff696b9f88456d9`. The audit used the commit graph
and per-commit diffs, not a blind tip-to-tip diff.

No maintainer checkpoint, database, trainer fork, or model component is
activated by this record. It does not authorize a smoke, long run, evaluation,
merge, cherry-pick, push, or model promotion.

## Decision

No post-`67af016` maintainer commit should be cherry-picked wholesale into the
current Generalist training route. The useful changes either already have an
independently tested `dev` implementation or are experimental policies that
need their own experiment decision. The UI, puzzle, autosave, web, and GapNet
changes do not belong to the disabled-component corrected-v4 baseline.

One behavior-neutral idea was independently implemented after review:
`3a4e19ffd934a7e30393a7a7b28ff9b7153d0f39` exposes completed heuristic search
depth through the public agent wrapper and records the per-move mean. It does
not copy the maintainer trainer or reach through the wrapper to `_inner`.
Ninety focused trainer, identity, route, and dependency tests pass for that
change and its surrounding contracts.

## Commit Classification

| Maintainer change | Disposition | Evidence-based reason |
| --- | --- | --- |
| `00e9d07` v2b fork | Do not import | It is a fork of quarantined v2a and lacks the current managed launch and exact-resume contract. |
| `6263759` per-level temperature and boost | Defer | It changes exploration policy and recovery behavior. A separate experiment must define and test it; the first corrected baseline explicitly excluded unvalidated recovery reheat. |
| `fd04a5d` trainer draw detection | Reject implementation | It counts placement states, does not reset repetition at placement/removal boundaries, starts without the stable origin, labels rule draws as long truncations, and has no focused tests or retry/branch state continuity. `a0877c4` independently implements the frozen rules contract. |
| `4f6862e` exact resume | Superseded on `dev` | The current route already persists optimizer, RNG, target, counters, histories, pending experience, database identity, and checkpoint envelopes with focused equivalence tests. |
| `41c2c13` EMA-driven learning rate | Defer | This changes optimization policy and is not required by the corrected-v4 experiment. |
| `700d6b9` opponent tag | Already represented | Structured diagnostics already record `game_type`; no console-only patch is needed. |
| `dafae4a`, `dec366c` advancement isolation | Independently implemented | `8e7ea26` separates primary full-difficulty evidence from retry, branch, recovery, and lower-difficulty games with tests. |
| `e99bfa4` infrastructure-failure filtering | Independently stricter | Current required-component and fixed-work failures stop or quarantine the run rather than silently becoming neutral experience. |
| `cdc880c` realized search depth | Independently implemented | `3a4e19f` adds a public wrapper contract, backward-compatible diagnostics, and focused tests. |
| `6bea0aa` opponent and Sentinel curriculum | Defer | It changes the opponent mixture and activates a component deliberately disabled for the first corrected baseline. |
| `8358042` uniform policy fallback | Reject | Converting a collapsed or invalid policy row into random experience violates the fail-closed training contract. |
| `7b9d6bb` PPO temperature, non-finite guard, fixed probe | Split decision | A2C rollout/update temperature is fixed by `9a15def`; non-finite failures are already guarded. PPO remains disabled. A fixed probe is a possible future monitoring feature, not a reason to import the mixed commit. |
| `03e5e95` and `6796ff2` through `810b440` v2c shaping | Do not import | These are observation-based reward-shaping experiments followed by several corrective commits. They explicitly approximate engine reconciliation and are not a proven rules or baseline fix. |
| `d1df6de` GapNet v3 | Out of scope | GapNet is disabled and requires its own provenance, training, and promotion gates. |
| Later puzzle, web, UI, and autosave changes | Out of scope | They do not affect the Generalist training route. |

## Rules Observation Requiring External Follow-up

Current `dev` ruleset `nmm-training-core@2` uses MIF
`stable-moving-v1`: an already stable moving origin is occurrence one for
threefold repetition, and placement/removal reset the repetition window.

Sanmill `master` at
`57f41c1d0dae90e6f614c6aa9b2c177e9df4ffc0` also resets on placement/removal
and implements the 100-logical-ply no-progress rule. Its strict-UCI regression,
however, starts a moving FEN and declares threefold only after action 9 in a
four-action cycle. Under origin-counted `stable-moving-v1`, the supplied origin
would be observed again at actions 4 and 8 and the draw would occur at action
8. This is a precise convention mismatch, not evidence that either engine's
move legality is wrong.

The mismatch does not block a self-contained NMM_LLM training smoke whose MRS
identity is recorded. It does block claiming exact live-rule parity with
Sanmill or using Sanmill as the formal referee until the convention is made
explicit and tested.

## Verification

- `git fetch origin --prune` left `origin/main` at `bc46b51e` and
  `origin/dev` at `b599e0d4`.
- The reviewed post-merge range was `67af016..origin/main`.
- `tests/test_train_s_gen_v2*.py` plus training identity, route, and strict
  dependency tests: 90 passed.
- The earlier rules commit passed 106 focused game/trainer tests and the
  mandatory Malom/provenance suite: 103 passed, 498 subtests passed.
- No maintainer gameplay or trainer source was copied into the active route.

The next long-run gate remains `needs_decision`: a new objective, resource
envelope, truncation policy, isolated SpecialistDB/output lineage, update-
reaching smoke, and explicit launch authorization are still required.

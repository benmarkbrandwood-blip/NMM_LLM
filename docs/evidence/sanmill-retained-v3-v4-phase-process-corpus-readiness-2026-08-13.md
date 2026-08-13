# Retained-v3/v4 phase-process corpus readiness — 13 August 2026

Status: `completed_evaluation_from_source_readiness_0ff79e39`

Current verdict: `completed_evaluation_needs_decision`

The source corpus, successor-owned inputs, evaluator core, fail-closed runner
and immutable machine-readable plan were technically ready on clean published
`dev`. Two complete preflights produced identical source readiness
`0ff79e398233c7ed9fcdec4cc5cd406837330140a3c1cec720e11eaa274ae365`.
At this readiness capture no product authorization existed. Direct authority
was subsequently bound to that exact source identity and plan `4c85ff33`, and
the 156-game evaluation completed once. See the
[completion evidence](sanmill-retained-v3-v4-phase-process-generalization-v1-result-2026-08-14.md).
The preparation itself requested no corpus policy move, game, Sanmill search,
training step, optimizer update, database mutation or checkpoint mutation.

## Observed facts

The source-only builder began with the 64-entry phase-covered review corpus.
Only histories that can be replayed from the standard initial position without
the earlier colour-only transform were eligible. It then excluded all 12
source entries already frozen in the prior phase-replay development corpus.
This left 42 candidate-blind histories before the current strict-rule gate.

Two fresh strict-Sanmill replay passes agreed byte-for-byte over all 42
histories. Three movement histories, source entries 29, 31 and 32, reach the
same strict threefold-repetition terminal at logical ply 142 before their
requested source states at plies 188, 159 and 177. They are invalid starts for
the current referee and were excluded by the pre-result replayability rule.
The remaining 39 starts are non-terminal under the pinned current referee.

| Gate | Observed | Expected | Result |
| --- | --- | --- | --- |
| Candidate blindness | no checkpoint or route bundle loaded; zero candidate outcome rows read | source membership independent of v3/v4 outcomes | pass |
| Prior phase-development use | 12 frozen source-entry indices excluded | no reuse of the 12-start measurement corpus | pass |
| Strict history replay | 84 fresh processes; two equal passes over 42 histories | deterministic current-rule replay | pass |
| Strict start validity | 39 non-terminal; 3 pre-start threefold terminals excluded | every accepted start ongoing | pass |
| Prior 64-opening overlap | 0 exact FEN; 0 `ring16` orbit | disjoint from plan `035c68f8` development openings | pass |
| HumanDB exposure | 0/39 D4 matches | zero | pass |
| retained-v3 SpecialistDB exposure | 0/39 D4 matches | zero | pass |
| retained-v4 SpecialistDB exposure | 0/39 D4 matches | zero | pass |
| Candidate database side effects | both sidecar-free snapshots remained sidecar-free | read-only | pass |
| Evaluator core | variable replay, relative 108-ply snapshot, canonical ledger, start clustering and live web implemented at `f8070d1` | exact source support | pass |
| Successor inputs | snapshot identity `b35ecc06`; both bundles identity-equal; both DBs byte-equal, read-only and sidecar-free | no completed-plan runtime path reuse | pass |
| Fail-closed runner | stable source-readiness binding, non-skippable tests/history replay, semantic fail-close and host-interruption exact suffix resume | exact controller | pass |
| Machine plan | identity `4c85ff33`, 156 games, two active hours, implementation `5a318a0`, plan commit `117a5be` | canonical published plan, not authority | pass |
| Full source preflight | two equal source identities `0ff79e39`; ten technical gates pass; zero corpus candidate moves and zero games | stable technical evidence before authority | pass |
| Launch authority | absent at source-readiness capture; subsequently supplied and consumed as `ceedd13a...` | plan-bound product authorization | historically absent; completed later |

The accepted corpus has 18 placement, 14 movement and seven flying starts;
22 have White to move and 17 Black. Source-history lengths range from 7 to
178 logical plies. Initial strict no-capture counts range from 0 to 52. The
side-to-move Malom labels are 13 W / 15 D / 11 L. These strata are descriptive
properties of a deterministic source pool, not a population sample.

## Frozen identities

| Item | SHA-256 / identity |
| --- | --- |
| Corpus identity | `3be3d76c34511e0f78d0f5bfe4a338c415c393306a955538bb85823e9d62c080` |
| Corpus file SHA-256 | `8353ff3e52465bf99f7cf468a9cbcb4681a673ac2cebcdae00c253df8a22670b` |
| Ordered records identity | `8fdf3adf60857543a440aee4b354938bec32a6f6f667effa381774abadf7d95d` |
| Strict replay audit identity | `6f65dc3dfa52fd6c7aaae23698e57df51239729105bd6ca79f9dd26d7815d349` |
| Exposure audit identity | `f9b2ebb1fdec2dfdfa451aba90c340b8f59de2d2fd456869e09604bd268b0b2b` |
| Completed diagnostic plan | `035c68f80b94dddb8d139d56c38c86c4fde29fa13de5e19db1f4e1fe484c318e` |
| Successor input snapshot identity | `b35ecc061e53a35e227c69ff886a7c6534e707bd124abdbe13acbbf9647f48ac` |
| Successor input manifest SHA-256 | `cda9456e0234a9532ddfb1b90e3a78bb6a35ef788c0eddfca607e9f33cb1942a` |
| Evaluator core commit | `f8070d125844635bde8095079cbe3ea5d36e99dd` |
| Stable readiness implementation commit | `5a318a063b561b12bafe5e72e44ff6fdc9426f1e` |
| Frozen machine plan identity | `4c85ff3362927db9b63014e0c91022a5d169d19efa4aa85b3a643febd0ce3256` |
| Frozen machine plan file SHA-256 | `09245e5f66af3d18ba2818d1dfac70b4c7eec8d63c9388d501b32846dfccf9d3` |
| Frozen machine plan commit | `117a5be8086af04ba0b311f44a23cdc9804a7284` |
| Source readiness identity | `0ff79e398233c7ed9fcdec4cc5cd406837330140a3c1cec720e11eaa274ae365` |
| Post-authorization readiness | `aeea1625ce38607aea4c08a0c0f363ba0c1d29c7584d4f498f562b6300f58677` |
| Runtime spec identity | `bb349a96df3e8445d3687c7c24dc474fe595d63aa890085ed6c6b2a94574fe72` |
| Result / completion identities | `6007af186b9a7ce908416f4578ebc31c0c19fc27733c32ed44751bb39cc3c812` / `48ac2ad4c6abc79b69c7de597ad46a5197949b4dcdee1f962e621d1be2fc57c8` |

The tracked artifact is
[the phase-process corpus](../experiments/sanmill-retained-v3-v4-phase-process-corpus-v1.json).

## Precision basis

The completed 64-start development ledger was reaggregated without new games:
the two colour-specific horizon differences were averaged within each start.
The start-level differences were `-0.5` six times, `0` 43 times, `+0.5`
14 times and `+1` once. The mean remained `+7.8125pp`; start-level standard
deviation was `29.8392pp`; the 95% engineering half-width was `7.3106pp`, for
interval `[+0.5019pp, +15.1231pp]`.

That observed standard deviation implies 35 starts / 140 games for an
estimated fixed half-width of 10pp. With all 39 accepted starts, the planning
estimate is 9.3651pp and the workload is 156 games. This is a reused-corpus
planning estimate, not a guarantee: the new phase corpus may have a different
variance and can still finish `inconclusive_precision`.

Because accepted histories start at different absolute plies, the prior
absolute-ply-120 endpoint cannot be copied literally. The comparable future
endpoint is survival through 108 additional logical plies after each frozen
start, matching the prior 12-to-120 observation window.

## Hypotheses

1. The named retained-v4 route's higher 108-ply continuation survival may
   generalize from twelve-ply opening starts to placement, movement and flying
   histories disjoint from both route databases.
2. A generalized difference may be associated with no-capture-clock changes,
   repetition state or phase-specific opportunity exposure.
3. Eventual W/D/L may remain too sparse to distinguish playing strength.

## Supporting evidence

- The original process endpoint already produced a directional fixed-corpus
  result, and the start-clustered pilot precision is sufficient to justify a
  bounded 39-start confirmation attempt.
- All 39 accepted histories are reproducible under the exact current strict
  referee and are D4-independent of HumanDB and both candidate-owned
  SpecialistDBs at the start state.
- The source membership predates this v3/v4 comparison and was filtered only
  by prior-use, replayability and exposure gates, not by candidate output.

## Counterevidence and limitations

- The phase source is deterministic seeded rule replay, not expert or
  population play, and it is phase-skewed 18/14/7.
- The original phase review corpus was visible to the project and was used for
  other development probes. This can support a new-corpus process
  confirmation for the named v3/v4 routes, but not a held-out strength claim.
- D4 absence at the frozen start does not prove absence of semantic
  near-neighbours or successor states from training data.
- V3 and v4 still differ in seed, source, target age and accumulated database;
  a confirmation would remain descriptive and non-causal.

## Completion reconciliation

The requested direct authorization was supplied, consumed when the first game
opened, and not reused. All 156 games reached strict rules terminals in
399.619311 active seconds; no host interruption, retry, semantic recovery,
extension, training or update occurred. The predeclared primary decision was
`inconclusive`: v4-minus-v3 survival was `-2.5641pp`, interval
`[-6.0707pp, +0.9425pp]`. No further game or training is authorized. Preserve
the frozen machine plan and completed runtime under their recorded identities.

The reviewed source-only commands were:

```powershell
.\.venv\Scripts\python.exe tools\freeze_retained_phase_process_corpus.py
.\.venv\Scripts\python.exe tools\prepare_retained_phase_process_inputs.py prepare
.\.venv\Scripts\python.exe tools\freeze_retained_phase_process_plan.py
.\.venv\Scripts\python.exe scripts\run_retained_phase_process_generalization.py preflight
```

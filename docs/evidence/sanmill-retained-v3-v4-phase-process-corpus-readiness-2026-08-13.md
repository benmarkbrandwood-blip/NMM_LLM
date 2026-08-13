# Retained-v3/v4 phase-process corpus readiness — 13 August 2026

Status: `source_corpus_inputs_and_evaluator_core_ready_no_evaluation_authority`

Technical-readiness verdict: `not_ready`

The source corpus, successor-owned inputs and evaluator core are ready, but the
fail-closed runner, immutable machine-readable plan, final readiness identity
and product authorization do not yet exist. Route bundles were opened only
for identity verification after corpus membership was frozen. No policy move,
game, search, training step, optimizer update, database mutation or checkpoint
mutation occurred in this preparation.

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
| Fail-closed runner | no successor CLI/controller yet | preflight, exact suffix resume and launch gates | absent |
| Launch authority | none | plan-bound product authorization | absent |

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

## Next validation

Implement the successor fail-closed controller around the committed evaluator
core. It must bind the successor input manifest, audit every variable strict
history without loading a policy, forbid skipped tests at launch, and allow
only an explicitly authorized same-spec missing-suffix resume after host
interruption. Complete-result safe-capture and full-order reports must be
identity-bound zero-game reanalyses of the new ledger. Only after focused
tests, the mandatory provenance gate, immutable plan and readiness identity
pass may the product owner be asked once to authorize the bounded 156-game
run.

The reviewed source-only command was:

```powershell
.\.venv\Scripts\python.exe tools\freeze_retained_phase_process_corpus.py
```

# Retained-v2 frozen held-out evaluation result — 9 August 2026

## Outcome

The one authorised retained-v2 held-out evaluation completed all 64
colour-swapped pairs, or 128 games. Independent recomputation from the
hash-chained ledger exactly matched the persisted report.

The frozen decision is **`candidate_behind`**:

| Result | Value |
| --- | ---: |
| Wins / draws / losses | 3 / 102 / 23 |
| Candidate score rate | 42.1875% |
| Mean paired score difference | -0.15625 |
| 95% paired interval | [-0.23146, -0.08104] |

This rejects promotion of this checkpoint under this contract. It does not
show that the checkpoint is weak against humans, establish an Elo, compare
training algorithms, or justify publication. It is a relation to one pinned
500,000-node Sanmill profile on one frozen corpus.

## Disaggregated observations

| Source | Games | W / D / L | Score rate | Paired decision |
| --- | ---: | ---: | ---: | --- |
| Book | 44 | 1 / 34 / 9 | 40.91% | candidate behind |
| HumanDB | 42 | 2 / 37 / 3 | 48.81% | inconclusive |
| Perfect DB | 42 | 0 / 31 / 11 | 36.90% | candidate behind |
| Strict zero-match sensitivity subset | 68 | 1 / 49 / 18 | 37.50% | candidate behind |

Candidate White scored 45.31% overall and candidate Black 39.06%. The Book
split is more asymmetric: White scored 50.00%, while Black scored 31.82%.
Perfect DB was weak in both colours, at about 38% as White and 36% as Black.

At pair level, 40 pairs were equal, 19 were down half a point, two Perfect DB
pairs lost both games, and three pairs were up half a point. The three wins
came from one Book start and two HumanDB starts; there was no Perfect DB win.

All 23 losses were rules-terminal: 16 material losses and seven blocked-side
losses. Across both players, the complete termination mix was 57 fifty-move
draws, 45 threefold draws, 16 material losses and ten blocked-side losses.
There were no max-ply truncations or manufactured results.

## Execution and recovery evidence

The exact published implementation was
`23dd90008b1d260a054e0c3cb471b8aad71e99a6`. The final active evaluator time
was 483.720479 seconds. Sanmill served 5,475 search turns, averaging about
493,286 observed nodes and completed depth 12.64 under the 500,000-node
ceiling.

After four completed games, a read-only progress inspection overlapped a
Windows `os.replace` of `progress.json` and produced `WinError 5`. The
exception occurred while writing progress, not while validating or scoring a
game. The fifth game had not entered the ledger. The original failure record
was archived without changing its bytes:

```text
host-interruption-0001.failure.json
sha256:63615c1460fe0fc6c567c234bf1b2e368355b6c5fc2f758aa5ae870d38eab6af
```

The safe-resume preflight then verified the same repository commit, plan,
authorisation, runtime specification, host, candidate, Sanmill process,
four-record ledger tail and 124-game missing suffix. Only that suffix ran.
No completed game was replayed, no result was invented, and the final ledger
contains exactly 128 records.

Key immutable identities are:

| Artefact | Identity / SHA-256 |
| --- | --- |
| Runtime specification | `ceb6f2230c448abd23b3e2cdffe479bb697b953a76d0072852bf99b131282332` |
| Candidate bundle | `c2652119b64a2808ebcd5e7dc661873f3f897065b7d529bd9e261328f0981f23` |
| Game ledger | `100863efa58381fc736096440bf8ff4a178cd34215ac7b43e3d6f6767fae7892` |
| Ledger tail record | `fb57fd8eeb96c0b84db68ff05fbd9ddc81486fd2471269915e1194a7672f03d0` |
| Report file | `463579526a7f42ca136fcca9cf03a9668db18c21055261086e45ed800b395c91` |
| Result identity | `8848ad32e588daf2fcd0686be65b337e7fc621faaebdb58bd1dbefc73bcdff81` |

## Observation, hypotheses, and counterevidence

### Observed facts

- The primary paired interval and the strict-subset interval are wholly below
  zero.
- The deficit is concentrated in Book and Perfect DB strata; HumanDB is
  inconclusive.
- Black from Book starts and both colours from Perfect DB starts are the
  clearest weak cells.
- The training lineage did not simply collapse: every fixed policy-health
  gate passed, ending at a 96.55% preserving rate and +2.39808 logit margin.

### Working hypotheses

1. The policy may fail to generalise from theory-controlled starts,
   particularly as Black. The strict subset and per-stratum results support
   this, but the HumanDB result and single seed prevent a broad conclusion.
2. The final-200 training diagnostic of 47.50% may overstate fixed-baseline
   strength because it mixes frozen targets, curriculum levels and training
   starts. It is not a controlled comparison with the 42.19% held-out rate.
3. The decisive failures may be tactical conversion or defence errors. The
   terminal reasons support that possibility, but they do not locate the
   first bad decision.

### Counterevidence and missing evidence

- HumanDB is near even and statistically inconclusive.
- Book play as White is exactly even in this sample.
- There is one training seed and one retained checkpoint; no seed-stability
  claim is possible.
- This reinforcement-learning run has no conventional independent validation
  curve. Training curves and the fixed 29-state health gate are diagnostics,
  not substitutes for the held-out result.
- The full corpus is not completely data-disjoint. The 34-start strict subset
  reduces that concern and is worse, but it remains sensitivity analysis.
- No controlled ablation accompanied this run, so the result cannot identify
  which hyperparameter or component caused the gap.

## Decision and next experiment gate

Do not promote, publish, or rerun this candidate. Preserve it as the first
Sanmill-refereed retained research baseline.

The next safe step is a read-only first-losing-turn audit over the existing
ledger. It should compare all 23 losses with colour- and stratum-matched draws
and record phase, material, mobility, corrected value transitions and the
first candidate action for which independent evidence changes the outcome
class. It must not ask the candidate to replay corpus moves.

Only after that audit should one successor-training change and a matching
ablation be frozen. Curriculum, reward and architecture changes must not be
bundled. A later general strength claim needs independently trained seeds and
a separately authorised evaluation.

The machine-readable companion is
[`sanmill-corrected-retained-v2-heldout-result-2026-08-09.json`](sanmill-corrected-retained-v2-heldout-result-2026-08-09.json).

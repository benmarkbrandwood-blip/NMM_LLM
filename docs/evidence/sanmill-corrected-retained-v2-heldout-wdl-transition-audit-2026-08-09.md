# Retained-v2 held-out WDL transition audit — 9 August 2026

## Outcome

The read-only diagnostic replayed all 23 candidate losses and 23 deterministic,
non-reused draw controls from the completed held-out ledger. It did not load or
query the candidate policy, create games, or change the frozen result.

Corrected `sector-corrected-v1` Malom WDL covered every probed transition. Of
the 23 losses:

| Classification | Games |
| --- | ---: |
| First candidate WDL downgrade found | 19 |
| Candidate already losing at the 12-ply prefix | 4 |
| Insufficient Malom coverage | 0 |

The 19 first downgrades comprise 18 draw-to-loss transitions and one
win-to-draw transition. They occur at post-prefix logical plies 2 through 92,
with a mean of 30.21.

This is diagnostic evidence, not a statistical treatment effect. Controls were
matched by source stratum, candidate colour and strict-subset membership, but
the cohorts were selected after outcomes were known. Eighteen of the 23 pairs
also have equal prefix WDL; five do not.

## Two observed failure regimes

| Candidate phase | First downgrades | Mean ply | Mean material difference |
| --- | ---: | ---: | ---: |
| Placement | 6 | 3.67 | -1.00 |
| Movement | 4 | 34.25 | -1.50 |
| Flying | 9 | 46.11 | -1.22 |

All six placement downgrades occur only two to five logical plies after the
12-ply prefix. All nine flying downgrades occur with exactly three candidate
pieces. The flying mobility difference is large and positive because flying
permits a three-piece side to move to any empty point; it must not be read as a
general strength advantage or compared directly with movement mobility.

These facts argue against treating the losses as one undifferentiated collapse.
There is evidence of both immediate post-prefix tactical errors and late
three-piece conversion or defence errors.

## Mill closure and removal signal

Sixteen of 19 first downgrades are complete logical turns containing a capture:
all nine flying downgrades, three of four movement downgrades, and four of six
placement downgrades. Before the first downgrade, the 19 games contain 290
probed candidate turns, of which 62 contain captures.

This makes mill closure and compulsory removal a strong diagnostic lead, not a
proven cause. A capture-bearing complete turn can lose value because the
candidate chose the wrong removal target, chose the wrong primary move, or
formed a mill at the wrong time. Captures are also naturally concentrated in
high-leverage positions. The present audit deliberately did not enumerate
counterfactual alternatives, so it cannot distinguish those explanations.

## Pair context changes the interpretation of four losses

Four losses are already theoretically lost for the candidate at the prefix:

```text
source-core-002
source-core-009
source-core-020
source-core-032
```

In every corresponding colour-swapped game, the candidate has the favourable
side but draws. The pair deficit therefore cannot be attributed simply to the
unfavourable colour eventually losing. The more useful question is why the
candidate did not convert the favourable side before threefold or fifty-move
adjudication.

The two full-point pair deficits are both Perfect DB starts:
`source-core-049` and `source-core-058`. The candidate loses both colours from
each start. These should remain explicit high-priority diagnostic cases.

## Evidence limits

- Malom WDL does not include the live repetition or fifty-move history.
- Coarse WDL does not expose full Malom move ordering or conversion progress.
- One loss first falls from win to draw; this first-downgrade audit does not
  continue to its later draw-to-loss transition.
- The candidate was never loaded, so the evidence says nothing direct about
  logits, sampling temperature or feature-route attribution.
- There is one training seed and no controlled ablation.

The audit's ignored canonical output is bound by:

| Artefact | Identity / SHA-256 |
| --- | --- |
| Audit identity | `6bbb4a50aa7999d06679c802cfeb5b913f0f5abf0689aa0291ec55459304b504` |
| Audit file | `871dd7935f7aa3231e6e364974e5207ef272501483e29338014cde16525b5692` |
| Implementation commit | `c00ccb1b7c72885551141f32a84db2f5d5a0acec` |
| Held-out ledger | `100863efa58381fc736096440bf8ff4a178cd34215ac7b43e3d6f6767fae7892` |
| Held-out result | `8848ad32e588daf2fcd0686be65b337e7fc621faaebdb58bd1dbefc73bcdff81` |
| Malom manifest | `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |

## Next falsifiable validation

The next safe step is a read-only full-oracle alternative audit of the 19
first-downgrade states. It must enumerate complete legal turns and use the
corrected Malom comparator to separate:

1. a wrong capture target under the same primary action;
2. a wrong primary action or premature mill closure;
3. no value-preserving alternative because the earlier position was already
   strategically compromised beyond coarse WDL.

The four favourable-colour counterpart draws need a separate conversion audit
using full Malom ordering plus repetition and no-progress counters. Only after
these checks should one successor-training change and one matching ablation be
frozen.

The machine-readable companion is
[`sanmill-corrected-retained-v2-heldout-wdl-transition-audit-2026-08-09.json`](sanmill-corrected-retained-v2-heldout-wdl-transition-audit-2026-08-09.json).

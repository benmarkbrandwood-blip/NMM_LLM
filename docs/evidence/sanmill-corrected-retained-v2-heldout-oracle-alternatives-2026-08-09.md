# Retained-v2 held-out full-oracle alternatives — 9 August 2026

## Outcome

The read-only audit enumerated all complete legal turns at the 19 committed
first-downgrade states found in the prior WDL audit. Every legal turn received
a lossless corrected Malom value in one common parent context. The candidate
policy was not loaded or queried and no new game was played.

The result is unambiguous at the move level:

| Finding | Count |
| --- | ---: |
| First-downgrade states | 19 |
| Chosen actions that are full-oracle best | 0 |
| Capture-bearing chosen actions | 16 |
| Same-primary preserving capture alternatives | 0 |
| Non-capturing primary-action errors | 3 |

Across the 19 states there are 612 complete legal turns, 351 WDL-preserving
alternatives and 347 full-oracle-best alternatives. None of the preserving or
full-best alternatives contains a capture.

This changes the earlier interpretation. The evidence does not support a
simple “wrong removal target” diagnosis. In all 16 capture cases, retaining
the same `from`/`to` primary action and changing only the compulsory removal
cannot preserve WDL. The candidate needs a different primary action, which in
these positions also means not closing that mill on that turn.

## Mechanism visible in the retained trainer

The exact `train_s_gen_v2.py` blob is unchanged between the managed plan
commit, the final training segment and the current audit:

```text
ed89c7cd40666340b738d0470b7fc905324084a7
```

That trainer has two relevant facts:

- every newly formed mill receives an unconditional immediate `+0.25`; and
- the immediate Malom reward weight is `0.0`, although complete-turn Malom
  quality is queried and logged.

During placement, heuristic-delta and Sentinel rewards are gated off. Four of
the six placement downgrades are mill-forming captures only two to five plies
after the prefix. Their immediate positive shaping is therefore the mill bonus,
while the exact value downgrade receives no immediate penalty. Delayed outcome
credit still applies, so this is a concrete reward conflict, not proof that the
bonus alone caused the final policy.

## Training-log context

Accepted per-game logs are available for 4,800 of 5,000 games. The missing
range, games 3,001–3,200, is the already documented segment-13 recovery gap.
For the available 115,765 learner steps, the reward components reconstruct
exactly 5,955 mill-forming actions and the Malom diagnostic records 4,097
value-downgrading actions. The old log schema does not preserve their
intersection.

Winning trajectories contain 540 mill actions in 2,550 learner steps, much
more densely than losses or draws. This is important counterevidence: globally
removing all mill shaping could damage a useful signal. A surgical gate is a
better experiment than setting the bonus to zero everywhere.

## Evidence limits

- The 19 states are outcome-selected diagnostics from one seed.
- The audit proves action ordering, not which reward term produced the logits.
- The old logs lack a formed-mill by Malom-downgrade cross-tab.
- Movement and flying also receive heuristic shaping and all phases receive
  delayed outcome credit.
- This audit does not address conversion progress, repetition or the
  fifty-move rule in the four favourable-colour counterpart draws.

Key immutable identities are:

| Artefact | Identity / SHA-256 |
| --- | --- |
| Oracle audit identity | `7cfa9ede873ae4fb34d7821472c62bba540f1b509476073062d52b487995cf65` |
| Oracle audit file | `29e3ed6d2af1389a90ef46869db5a2b8800e8c9c3993e13dd80af72ef07a7f28` |
| Oracle-auditor commit | `51cf3b7d39bf2624842fb7ee1b43f3e42cfdbc97` |
| Source WDL audit | `6bbb4a50aa7999d06679c802cfeb5b913f0f5abf0689aa0291ec55459304b504` |
| Held-out ledger | `100863efa58381fc736096440bf8ff4a178cd34215ac7b43e3d6f6767fae7892` |
| Corrected Malom | `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |

## Next falsifiable experiment

Add an explicit, backward-compatible mill-bonus mode:

```text
legacy-unconditional
malom-preserving-only
disabled
```

The successor should explicitly select `malom-preserving-only`. A required
Malom lookup failure must stop rather than silently award or suppress the
bonus. The log must separately record formed mills, awarded mill reward,
Malom downgrades, and their intersection per game and phase.

Before a retained run, execute fixed reward tests and a no-update integration
probe on the 19 frozen states. Then freeze one matched smoke/ablation pair with
identical seed, data, curriculum, baseline and evaluation; the only learning
difference should be the mill-bonus mode.

The machine-readable companion is
[`sanmill-corrected-retained-v2-heldout-oracle-alternatives-2026-08-09.json`](sanmill-corrected-retained-v2-heldout-oracle-alternatives-2026-08-09.json).

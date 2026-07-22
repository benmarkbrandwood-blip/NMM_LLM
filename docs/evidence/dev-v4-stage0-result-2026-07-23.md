# Dev v4 Stage-0 Result — 23 July 2026

Status: **completed; protocol decision `accepted`; no promotion claim**.

The product owner's one-run authorization was consumed by one immutable CPU
specification and its 212-game ledger. The run completed normally and the
final ledger independently recomputed to the same result. No training,
checkpoint mutation, second evaluation, promotion, or publication action was
performed.

## Frozen identities

| Field | Value |
| --- | --- |
| Evaluation ID | `dev-v4-formal-paired-eval-v1` |
| Runtime Git commit | `a45b44e2c5965b215c2fb699b58dd92d135f0303` |
| Runtime | Windows 11, PyTorch `2.8.0+cu129`, CPU, float32 |
| Route | `policy-argmax-v1`; lookahead features `zeroed-72` |
| Components | Sentinel, ValueNet, GapNet, HumanDB override, and SpecialistDB override off |
| Spec identity | `26f80c14d70320aa025c85319791c625e821babb2e542095aeb4711d4c11d48b` |
| Spec file SHA-256 | `c04e2e097803d19593a7eb748643e177b673614fc2fed81fe98485be4daf9cf5` |
| Candidate bundle | `ab2c8f38570c14ec839d2e516732b22e1c811bf5911843b5eafee9cbaf3fb483` |
| Scratch-init bundle | `058145238ac03f006779689e14af35eac1d3128921d4b2cddfb698d457fdd86f` |
| Start-position identity | `04bc5782ab79ebeba34d0ff91bcd40fe05e823d539b16ba234b5eedcd123bb9d` |
| Final ledger SHA-256 | `6800d95ca8e968e2b7ded3f02b87451e1a84c7f47a068a72e36d53bcd1978848` |
| Result identity | `0280a4a8eee8ee39506dcef0816a3265bc64229985699c63a3420e34978bbc99` |

The ignored local artifacts are under
`learned_ai/checkpoints/evaluation/dev-v4-formal-paired-eval-v1/` as
`stage0-spec.json` and `stage0-games.jsonl`. They must be preserved together.
The final ledger is 135,235 bytes and 212 newline-terminated records. No
`.partial` ledger remains. A separate `stage0-result.json` was not generated;
the immutable ledger is the primary evidence and the aggregate below is
recomputable from it.

## Recomputed result

| Measure | Result |
| --- | --- |
| Games | 212 |
| Candidate wins / draws / losses | 193 / 8 / 11 |
| Pair-score-difference mean | `0.8584905660377359` |
| Frozen normal interval | `[0.7972174156720373, 0.9197637164034345]` |
| Protocol decision | `accepted` |

The candidate-as-White half was 103 wins, 1 draw, and 2 losses, with mean
score `0.9764150943396226`. The candidate-as-Black half was 90 wins, 7 draws,
and 9 losses, with mean score `0.8820754716981132`.

Across the 106 colour-swapped pairs:

- 87 had pair difference `1.0` (candidate won both games);
- 8 had pair difference `0.5` (one win and one draw);
- 11 had pair difference `0.0` (one win and one loss);
- no pair had a negative difference.

There were 204 rules-terminal games and 8 repetition draws. Game length was
6–51 ply, with median 26 and mean `27.014150943396228`; no game reached the
200-ply adjudication limit.

## Integrity verification

The post-run recomputation command loaded only the frozen spec and final
ledger. It returned the same game count, W/D/L totals, mean, interval,
decision, ledger hash, spec identity, and result identity as the runner. The
ledger audit also found exactly 212 unique game IDs, 106 unique pair IDs, one
spec identity, and no partial file.

```powershell
.\.venv\Scripts\python.exe scripts\paired_evaluation.py recompute `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\stage0-spec.json `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\stage0-games.jsonl
```

## Interpretation boundary

`accepted` answers only the preregistered Stage-0 question: under a
deterministic zero-lookahead policy-argmax ablation, the trained candidate
shows a strong signal against architecture-matched random initialization on
this fixed 106-position convenience corpus.

It does not establish general playing strength. Every start is in placement,
the corpus is source-overlapping and not demonstrated held out, the control is
random initialization rather than a competent player, and the evaluator
zeroes a 72-feature block used during training. The result does not authorize
promotion, publication, another run, or a changed analysis. A route-aligned,
phase-covered evaluation with a competent frozen baseline remains necessary
before a formal strength claim.

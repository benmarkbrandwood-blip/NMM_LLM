# Sanmill Malom policy-auxiliary no-update batch-capture v1 readiness — 9 August 2026

## Verdict

`ready_for_final_published_preflight`

The bounded three-seed batch capture is technically ready for one final
preflight from the documentation-complete published `dev` tip.  It remains
unlaunched.  It is a read-only gradient-distribution measurement, not a
training run, coefficient selection, strength evaluation, promotion, or
publication decision.

The passing preliminary preflight ran from clean, published implementation
commit `ba74a35a59af1f4d52d67982c6916f139dfe7f51`, with
`dev == origin/dev`.  Its local report has:

- readiness identity
  `7bb7c33990e3d83d55b9402760e863ecb1516ad36bf0690f9e15e3ee23d234eb`;
- file SHA-256
  `fbf7df74c178ccf76639a92281dbd27aece88c07a8d85eaeffd7599ca0b2ce32`;
  and
- source tree `79b7ef376e3e684d9b5b9373f12572a7569189bb`.

That preliminary identity is evidence that the implementation preflight
passed.  It must not be used to launch after this documentation commit changes
the source tip.  The final preflight must be generated from the final clean,
published `dev`, and any one-run authorization must name that later identity.

## Observed facts

The immutable plan is
`docs/experiments/sanmill-malom-policy-auxiliary-no-update-batch-capture-v1.json`:

- plan identity
  `a5c85ed13baecf3efed6780effdf590e97560a12e9ab197c5fc7bb4bf7341fab`;
- raw SHA-256
  `5813dbf320bc7de6a492ed975e6a807285ffa1f9728e7d15648c2771510b6a49`;
- parent route identity
  `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb`;
  and
- parent raw SHA-256
  `b422ea44fad5fa4181294a4bc3d6e77a48692b43672ccc2ed6976d87f1aabe36`.

The preflight created three distinct deterministic fresh initializations.  In
each case the frozen target was byte-identical to the learner before any game:

| Seed | Learner and frozen-target state SHA-256 |
| ---: | --- |
| 52 | `87d7b0ef02f4a02142be60c1762ca79598bde63999355771e34c5255b116738a` |
| 53 | `e487e10f28d251fedd244a0dc40ef883afc97e498fed9a74c0544b3ea2c6933d` |
| 54 | `ebf2c608072a6fcb81129404a9398e1b6e6a5cf27528e9a2aad32d7f0a14baf4` |

No optimizer was constructed, no optimizer step or backward call occurred,
and model parameters did not require gradients during preflight.  The
unscheduled route check completed two logical plies, used two Sanmill referee
plies, made zero opponent search calls, and ended only at its diagnostic
max-ply boundary.

The bounded execution remains exactly 60 complete games: 24 against the
pinned 1,000-node Sanmill route and 36 against per-seed fresh frozen targets.
Its hard ceilings are 7,200 logical plies, 1,440 search calls, 1.44 million
requested nodes, 33 measured batches, and two active hours.  The schedule is
source- and color-balanced and rotates one deep route across the three seeds.

The closed HumanDB route snapshot is a dedicated byte-identical local copy of
the frozen twelve-ply audit snapshot.  It has SHA-256
`97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c`,
size 738,091,008 bytes, `quick_check=ok`, and no WAL or SHM sidecars.  The
source audit snapshot was not modified or relabelled.  Its unversioned legacy
Malom columns remain disabled; only human frequencies and outcomes are usable.

The isolated SpecialistDB has SHA-256
`b4d522d23720ab86013bbadd6b3414fba1e205ab988f3949c6ed417dca486b7f`,
is empty, reports `quick_check=ok`, has no sidecars, and declares
`malom_label_version=sector-corrected-v1`.  The corrected Malom source has 512
components, manifest SHA-256
`f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747`,
and anchor SHA-256
`5078bf84505fe2845a4af7c36907efa2d66b2eb76f149ce12faa248117405b68`.

The ruleset is `nmm-training-core@2`, semantic digest
`sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`.
The pinned Sanmill runtime remains commit
`a6623f88959f7453594df274fbe1f128af7ff55e`, binary SHA-256
`5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`,
with strict runtime identity
`705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea`.

## Hypothesis

The earlier four-arm calibration was inconclusive because a fixed raw
auxiliary coefficient produced highly variable policy-head gradient influence
across production batches.  A multi-seed distribution of ordinary and
auxiliary gradients is needed before a normalization target, cap, or
low-gradient behavior can be designed without post-hoc coefficient selection.

## Supporting evidence

- The four-arm result identity
  `d11384f661192db84662a6e43e85cdd6eb299672724178a83a02858b0b12113f`
  showed auxiliary-to-policy gradient ratios of approximately 0.69 and 26.7
  on the two recoverable production batches.
- The separate gradient-interaction audit identity
  `9d338f3b0e3c2a8d4a94a480a814ebc1137c5d3ffe60f1141865ceedd1b715e0`
  established a read-only measurement route without optimizer construction,
  `backward()`, or parameter mutation.
- The batch-capture implementation reproduces whole-game production batching:
  it measures only after the accumulated batch reaches 64 learner steps and
  does not split a trajectory to manufacture fixed-size batches.
- The preliminary preflight independently reverified source publication,
  data identities, three fresh model identities, and the strict referee route.

## Counterevidence and limits

- No capture games have run, so there is not yet a gradient distribution,
  train curve, validation curve, per-phase empirical support distribution, or
  outcome comparison.
- The three seeds increase initialization coverage but do not turn this
  diagnostic into a strength or generalization evaluation.
- Candidate target ratios `0.25`, `0.5`, and `1.0` are descriptive only.  The
  probe cannot select one, and it cannot establish optimizer stability because
  no update is permitted.
- Sanmill games use the 1,000-node diagnostic route.  They exercise integration
  and source diversity, not a retained strength baseline.
- HumanDB emits an expected warning that its historical Malom label version is
  missing.  Those columns are disabled by design; this is not evidence of a
  failed or partially enabled teacher.

## Verification

The implementation and readiness fix passed:

- 181 focused batch-capture, auxiliary, rollout, referee, route, and preflight
  tests;
- 103 mandatory Malom, Sentinel DB-teacher, and label-provenance tests plus
  498 parameterized subtests;
- Ruff formatting and lint checks over the modified capture code and tests;
- Pyright with zero errors or warnings; and
- `git diff --check`.

The first real preflight exposed a report-publication defect: preflight reports
are signed by `readiness_identity`, while run and failure reports are signed by
`report_identity`.  Commit `ba74a35a59af1f4d52d67982c6916f139dfe7f51`
fixed the schema-aware identity selection and added a serialization regression
test.  The next attempt correctly refused an uncommitted tracked worktree.  The
passing preliminary preflight then ran only after that fix was published.

## Next validation experiment

After this evidence and the handover update are published, run preflight once
more to a new exclusive-create output from the final clean `dev == origin/dev`.
If every identity and route check remains unchanged, obtain explicit authority
bound to that final readiness identity for exactly this one command:

```powershell
.\.venv\Scripts\python.exe `
  scripts\capture_malom_policy_auxiliary_batches.py `
  --launch capture `
  --plan `
  docs\experiments\sanmill-malom-policy-auxiliary-no-update-batch-capture-v1.json `
  --paths-config data\training_paths.local.json `
  --readiness `
  out\diagnostics\malom-policy-auxiliary-no-update-batch-capture-v1-final-readiness-2026-08-09.json `
  --expected-readiness-identity <final-readiness-identity> `
  --run-id `
  sanmill-malom-policy-auxiliary-no-update-batch-capture-v1-20260809-001 `
  --output `
  out\diagnostics\sanmill-malom-policy-auxiliary-no-update-batch-capture-v1-20260809-001.json
```

The final readiness, result, failure, partial, retry, continuation, training,
promotion, and publication paths were absent during this audit.  The output
must be checked again immediately before launch.  Silence, timeout, earlier
experiment authority, or this document cannot authorize the 60 games.

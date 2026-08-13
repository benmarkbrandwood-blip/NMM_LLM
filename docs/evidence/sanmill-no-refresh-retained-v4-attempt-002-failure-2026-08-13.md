# Sanmill no-refresh retained-v4 attempt-002 failure

Status: `failed_closed_consumed_never_retry_or_resume`

Date: 13 August 2026

## Bound launch

The product owner authorized exactly:

- plan identity
  `2a59a93f2ae384c6768bf1b584291636a93cf3edfa82f6b7266562c7f927a65d`;
- readiness identity
  `a6cd2cd15c8e78f9128aae9c8f1e6229cef93412f98d69df45f8411ec7bd1c76`;
- source commit
  `12ecd9341a9a7a25c3391b3ea0ef2ece6b91652c`;
- seed 70, at most 5,000 games, 12 active hours and 250 games per
  segment; and
- managed exact-resume only between accepted segments, with no automatic
  retry, failure recovery, extension, held-out evaluation, promotion,
  publication or release.

The authorization file SHA-256 is
`bfd031cf623a81130c5dec355f8258f18bcf9afc282554ca7a0c9588cdedfe5b`.
Immediately before launch, a real fetch confirmed that `HEAD`, `origin/dev`
and `plan.git_commit` were identical. `origin/main` remained at the reviewed
`40da3ddfced972c418541665ec739b3752edcd1f`, the tracked worktree was clean,
and no Python, `tgf-cli`, controller lock or segment directory existed. The
stored first-segment preflight was rerun and was byte-identical at 14,398
bytes with SHA-256
`1fbab55734e7c491d4cd2278251c913d40c529df1c173756478fb6acb85bbeab`.
The empty SpecialistDB also retained its frozen SHA-256
`5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`,
`quick_check=ok`, corrected label version and no sidecars.

## Failure

The managed supervisor started segment 0001 at
`2026-08-13T04:28:33.167924Z`. The trainer passed its launch-time preflight,
selected CUDA, loaded the fixed Sanmill runtime and began from fresh random
state. It then exited with code 1 at
`2026-08-13T04:29:02.186786Z`, before any segment or checkpoint could be
accepted. The controller recorded `managed_segment_failed` with reason
`trainer_exit_nonzero` and stopped. It did not retry or select a recovery.

The exception was:

```text
ValueError: min() iterable argument is empty
scripts/train_s_gen_v2.py:4010
```

The periodic legacy A2C path assigned `update_steps = ep_steps`. After the
optimizer update it cleared `ep_steps`, which also emptied the aliased
`update_steps`, and only then tried to calculate the batch's minimum and
maximum behaviour temperatures for `updates.jsonl`. The failure is therefore
an update-evidence aliasing defect, not a Sanmill, Malom, CUDA, data-identity
or optimizer non-finite failure.

## Mutated state and preserved evidence

The first rollout reached an in-memory A2C update before evidence logging
failed. No model or optimizer checkpoint was written, so there is no safe
resume parent. The lineage-owned SpecialistDB was nevertheless mutated to
73,728 bytes, SHA-256
`5acb8251fe601f2708082632520ec9d462ca3140563ec0cdc4b2e8001f0f5a0c`,
with 188 positions, one winning line, zero preferred plays and no sidecars.
It must remain with the failed attempt and must not seed a successor.

Preserve these ignored artifacts in place:

| Artifact | SHA-256 |
|---|---|
| `plan.json` | `5a068f9317be345c522b905514ada3a5cfd19015ac85b8ea785354aa7bb0fa95` |
| `technical-readiness.json` | `c2bd5ab126f742837b7627ff5a8a0e2d949695acb04c8661a12ed1c7e176beb7` |
| `controller-events.jsonl` | `8eb34dd84634971b3efcb84d93b57cae2101e33d3c83dd98ad04bec0f885b2a4` |
| `supervisor.stdout.log` | `3d4d605bbbf1a5efc4e64756a1d287b3ee3829e9868ae8ce90e0fac3cc1aaa82` |
| `supervisor.stderr.log` | `ee695ca0356114a1c6f2c47902836e853fcdf8edc651bf90cfb22db7ec1def80` |
| segment 0001 `run-manifest.json` | `aca5a9452bb41487be268bb5915bd6caa1c3915e71dd860f52e762bf36bcca28` |
| segment 0001 `run-events.jsonl` | `ae6c8b1c8fbbcd57279407bd6feeddca2a037f434fbbc467e15dd96fc8eb229e` |

There are zero accepted managed games and zero accepted segments. The partial
rollout, in-memory update and database writes are failure evidence only; they
are not training results or evidence about the no-refresh hypothesis.

## Disposition

Attempt 002 and its authorization are consumed. Do not run `run-authorized`
again, resume its segment, reuse its database, repair its files in place or
describe it as a completed smoke or long run. A successor requires a tested
source correction plus new experiment, plan, control-directory, database,
readiness and authorization identities. Seed 70 may remain the intended fresh
seed because attempt 002 retained no checkpoint, but a successor is still a
new lineage and not a recovery of this attempt.

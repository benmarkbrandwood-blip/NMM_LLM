# Sanmill Strict Referee Parity — 7 August 2026

## Status

`implementation_verified_publication_pending`

Sanmill local commit
`a6623f88959f7453594df274fbe1f128af7ff55e` adds an explicit
`mif-stable-moving-v1` strict-referee profile. Read-only source inspection and
a local black-box invocation confirm that it uses the same origin-counted
`stable-moving-v1` repetition convention as NMM_LLM ruleset
`nmm-training-core@2`.

At verification time Sanmill `master` was one commit ahead of
`origin/master`, whose tip remained
`57f41c1d0dae90e6f614c6aa9b2c177e9df4ffc0`. The tracked worktree was clean;
only the unrelated untracked `.codex-remote-attachments/` directory remained.
This record therefore does not yet promote the new commit to a remotely pinned
referee dependency.

## Reviewed Boundary

The new CLI option is:

```text
setoption name StrictRefereeProfile value mif-stable-moving-v1
```

The default remains `sanmill-live-v1`. The opt-in profile is confined to the
CLI import and history-rebuild path. It does not change Flutter/FRB play,
search, database use, patch behavior, or failure fallback.

Source inspection confirmed:

- a stable imported moving or flying origin is seeded as occurrence one only
  for the MIF profile;
- placement and removal reset the repetition window, after which a new stable
  moving boundary becomes occurrence one;
- a pending removal is not observed;
- a primary action and compulsory removal remain one complete logical turn;
- the 100-logical-ply no-progress rule is unchanged;
- changing the profile clears the loaded position and requires the caller to
  issue a new `position` command rather than silently relabel history; and
- `statejson` publishes the selected profile and a composite deterministic
  referee identity.

## Independent Black-Box Result

The locally built debug executable from the reviewed worktree was started as a
fresh process. It received the fixed moving origin and a reversible four-ply
cycle under strict failure policy and the MIF profile.

| Logical ply | Current occurrence count | History length | Result |
| ---: | ---: | ---: | --- |
| 0 | 1 | 1 | ongoing |
| 4 | 2 | 5 | ongoing |
| 8 | 3 | 9 | `draw_threefold_repetition` |
| attempted 9 | no state | no state | `position_history_illegal_action` |

The process returned:

- rules identity
  `3e62cb93a1e0afe4534ce4824d233344816050b547bb8761dd7fe985d8ad399f`;
- strict-referee format `SANMILL-STRICT-REFEREE-RULES/1`;
- profile `mif-stable-moving-v1`;
- repetition observation `stable-moving-v1`;
- `originCounted=true`; and
- composite semantic digest
  `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`.

This directly resolves the previously observed action-8/action-9 convention
difference. The Sanmill maintainer also reported 14/14 strict-UCI tests,
173/173 `tgf-mill` library tests, a clean complete workspace test run, Clippy
with warnings denied, formatting, cross-process byte equality, and
`git diff --check`. Those reported results support the change but remain
distinct from this NMM_LLM-side black-box probe.

## Publication and Adoption Gate

No MIF wire or reference-runner change is required. Before Sanmill is used as
the formal portable referee:

1. push `a6623f88959f7453594df274fbe1f128af7ff55e` normally to
   `origin/master`;
2. verify `master == origin/master` and a clean tracked worktree;
3. build and hash a clean pinned release executable;
4. update the NMM_LLM bridge contract to select
   `StrictRefereeProfile=mif-stable-moving-v1` explicitly and bind both rules
   identities; and
5. repeat the strict bridge rule, history, logical-turn, failure, determinism,
   and performance probes before any candidate-versus-referee evaluation.

This publication gate does not block self-contained NMM_LLM Generalist
training. It blocks only a claim that a floating or historical Sanmill binary
is the exact formal referee.

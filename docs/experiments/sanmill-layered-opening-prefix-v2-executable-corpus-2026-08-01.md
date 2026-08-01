# Twelve-ply layered-prefix executable corpus

Status: `executable_64_prefix_corpus_frozen_evaluation_not_authorized`

Decision date: 2026-08-01

The accepted `22 Book / 21 HumanDB / 21 Perfect DB` source membership is now
assembled as one portable executable corpus. The machine-readable artifact is
[stored beside this document](sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.json).
Its executable-corpus identity is
`417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9`;
the ordered 64-record identity is
`e8a1828cb1d7e0e86c686d934e87934c6c12e6a8cf7610974ed8035937ab8cff`.

## Inputs and binding

The corpus rederives and exactly matches the frozen
[source core](sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json).
Every executable record is bound to its ordered `source-core-NNN` member,
stratum member, source subtype, source-history identity, action history, final
NMM FEN, and final D4/`ring16` FEN.

The 22 Book records come from the frozen
[Sanmill Book audit](../evidence/sanmill-layered-book-source-audit-2026-07-25.json)
and
[expert Book audit](../evidence/sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json).
The 21 HumanDB records come from the later
[strict execution overlay](sanmill-layered-opening-prefix-v2-human-execution-2026-08-01.json).
The 21 Perfect DB records come from the frozen
[Perfect DB audit](../evidence/sanmill-layered-perfect-source-audit-2026-07-25.json).
All source, selection, runtime, and execution identities are included in one
input-identity set with identity
`b153ce589c67bce9fedcad1e4a9e18942f81bcedd452a0021cdb8a6dff222147`.

## Frozen result

| Measure | Result |
| --- | ---: |
| Executable records | 64 |
| Book | 22 |
| HumanDB | 21 |
| Perfect DB | 21 |
| Logical plies | 768 |
| Compound Mill-and-removal turns | 39 |
| Unique action histories | 64 |
| Unique final NMM FENs | 64 |
| Unique D4/`ring16` endpoints | 64 |
| Unique final Sanmill history identities | 64 |
| Unique executable-prefix identities | 64 |

The Book stratum contains 15 maintainer-curated plays and seven Sanmill named
variations. HumanDB contains 21 complete histories observed in the frozen
PlayOK sample. Perfect DB contains 21 deterministic StrictSteps routes. These
strata remain distinct in storage and later reporting.

## Sanmill runtime boundary

All 64 records bind the same pinned Sanmill source commit
`db65eb3e73189d934d615d0f47519d395193c646`, tree
`b8fa6c0119c2dec4443efc59deab8b7d835e0c88`, and strict protocol v1. They
correctly preserve two different binary records:

- 43 Book/Perfect records retain the historical audit binary SHA-256
  `cac2ec6fe45a9d798a89c6b8a5f52c767aa1c885a1156a96269b44ebf81976cc`;
- 21 HumanDB records retain the separately rebuilt exact-HEAD binary SHA-256
  `6502f7a2180769666c1ba6c801288a5ba079920e2bd6c1121f0e8b0c27e11e53`.

The binary identities are not normalised or claimed to be byte-equivalent.
Their common pinned source identity and each record's original execution
provenance remain explicit.

## Evidence boundary and next gate

This freeze makes the 64 placement prefixes executable and portable. It does
not turn them into match results, validate a candidate's strength, replace the
separate movement/flying corpus, or authorise a launch. No candidate was
loaded and no game was played.

The five focused corpus checks pass, including exact regeneration from tracked
inputs and rejection of a rehashed source-binding change. The wider relevant
layered-prefix run passes 81 tests after explicitly deselecting one historical
machine-local Perfect DB regeneration test. That test correctly rejects the
now-advanced `sanmill_checkout` because protected source paths changed; its
unavailable historical binary is not replaced and the test is not weakened.

Before any candidate-versus-baseline run, a separate formal-evaluation
readiness decision must still freeze and verify the baseline identity,
fixed-work search contract, paired colour assignment, rules-compliant
termination policy, workload, reporting intervals, and launch authorization.

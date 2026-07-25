# Sanmill Prefix Diversity Audit — 25 July 2026

Status: source-only, pre-result evidence; no prefix policy or evaluation is
authorized by this record.

Related:

- [machine-readable audit](sanmill-prefix-diversity-audit-2026-07-25.json)
- [complete book-path contract](../experiments/sanmill-book-path-corpus-v1.md)
- [complete book-path artifact](../experiments/sanmill-book-path-corpus-v1.json)
- [paired-prefix infrastructure](../experiments/sanmill-paired-prefix-infrastructure-v1.md)

## Question

The provisional opening smoke proposed 75% corrected-book prefixes and 25%
StrictSteps Perfect DB prefixes. Its stated motivation included opening
diversity. The complete book-path inventory made it possible to test that
motivation before choosing a policy or observing any candidate result.

This audit asks only:

1. how many structurally distinct eight-logical-ply endpoints the corrected
   book supplies; and
2. how much structural diversity the existing deterministic Perfect DB
   sampler supplies under the same prefix length.

It does not measure playing strength, human likeness, opening frequency, or
the quality of either source.

## Reproducible boundary

Audit script commit
`1594275b17425c972a31264c5813e43e08dd082c` ran from a clean NMM_LLM
worktree. It used:

- pinned Sanmill commit
  `db65eb3e73189d934d615d0f47519d395193c646`;
- pinned Windows binary SHA-256
  `cac2ec6fe45a9d798a89c6b8a5f52c767aa1c885a1156a96269b44ebf81976cc`;
- frozen book corpus identity
  `3bc9bc05a66a1a53255444266388838489020667272fc2ffa7445e7cf44be985`;
- complete standard Perfect DB portable identity
  `d6a85d25e87e28cc33e1408539585dadd83349da1cb63aa3f1a0f32307087508`;
- experiment namespace `perfect-prefix-diversity-probe-v1`;
- pair IDs `pair-000` through `pair-063`;
- seed 42, `uniform_candidate`, `cache_sectors=8`; and
- exactly eight logical plies per prefix.

Two fresh Sanmill data-query processes generated all 64 Perfect DB prefix
records independently. Their complete canonical records were equal. The
record-set SHA-256 is
`10655c2e37fcf2c6d75ede6b552755a23be9deacae58c0991e97e50b3a7c6c44`.

The machine-readable audit identity is
`a7bc734ad3f85d2ae3ab75c901467da7b1835932fefa9aadd6067e1f4a982990`;
its file SHA-256 is
`1965866901d99dc40c95773d51f6901a883d738594e35c5fe98b6630d454d213`.
It contains no host path. No candidate was loaded and no game was played.

## Result

| Source population | Histories | Exact final FENs | Final ring16 orbits |
| --- | ---: | ---: | ---: |
| All complete corrected-book paths | 192 | 84 | 7 |
| First 64 fixed-seed StrictSteps prefixes | 64 | 64 | 64 |

Every book endpoint is duplicated by history: 72 exact FENs have two
histories and 12 have four. Under the book's declared `ring16` symmetry, two
orbits contain 16 histories each and five contain 32 each. Therefore uniform
selection over 192 paths, or even over 84 exact FENs, does not produce 192 or
84 structurally independent openings. It produces seven structural endpoint
classes.

The 64 Perfect DB prefixes have 64 distinct histories, exact FENs, and
`ring16` orbits. Their orbit set has zero overlap with the seven book orbits.
This is a deterministic observation at the recorded identities, not a
guarantee that every future seed, database, or larger sample will remain
duplicate-free.

## Interpretation

The provisional 75/25 mix is not supported as a diversity-first policy. In a
64-pair workload it would allocate 48 book prefixes to only seven structural
book endpoints while allocating just 16 prefixes to the source that produced
16 distinct structural endpoints in its first 16 samples. Symmetric book
variants may still test coordinate handling and may preserve curated opening
style, but they must not be counted as independent strategic positions merely
because their FEN strings or histories differ.

This result does not show that the book is poor. It shows that the corrected
book and the Perfect DB serve different possible objectives:

- the book supplies a small curated family with ranked recommendations;
- Perfect DB sampling supplies much broader solved-opening structure in this
  fixed probe; and
- HumanDB would address human frequency or naturalness, not automatically
  optimality or structural coverage.

## Recommended pre-result design

If the next task is a **64-prefix diversity and bridge smoke**, the recommended
draft is orbit-stratified rather than percentage-random:

1. include one history from each of the seven corrected-book `ring16` orbits;
2. fill the remaining 57 slots with StrictSteps prefixes whose orbits are
   absent from the book and from earlier accepted Perfect DB prefixes;
3. choose each book representative by minimum total source-rank penalty, then
   lexicographic action history;
4. scan domain-separated Perfect DB pair IDs in order and accept only a new
   orbit, failing closed if the required count cannot be reached;
5. freeze all 64 complete histories before loading a candidate; and
6. replay the same history in both colour-swapped games.

At the recorded source identities, `pair-000` through `pair-056` already
supply 57 unique Perfect DB orbits, all disjoint from the seven book orbits,
so this construction would yield 64 unique structural endpoints without
source fallback.

This 7/57 recommendation replaces 75/25 only when the declared objective is
structural diversity. If the objective is instead exposure to the curated
book distribution, keep that as a separately named book-style stratum and do
not describe repeated `ring16` variants as additional independent evidence.

HumanDB should remain outside the first smoke. Its active local file currently
fails the immutable-source gate because of a non-empty SQLite sidecar, and the
product role of human frequency remains unfrozen. After an immutable snapshot
exists, its eight-ply coverage and `ring16` diversity can be audited under a
separate identity before deciding whether it is evidence, a third source, or
a human-likeness stratum.

Finally, these placement-only prefixes do not replace the reviewed
phase-covered 64-position corpus. Opening diversity and movement/flying phase
coverage answer different questions and must remain separate strata unless a
later formal contract explicitly combines them.

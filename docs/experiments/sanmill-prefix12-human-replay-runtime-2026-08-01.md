# Sanmill runtime for twelve-ply HumanDB replay

Status: `pinned_for_source_only_human_history_replay`

Decision date: 2026-08-01

The active Sanmill development checkout has advanced beyond the strict bridge
pin and contains changes inside the protected CLI and rules scope. Its current
binary also no longer has the identity recorded by the historical strict-v2
smoke. It therefore remains a reference checkout and is not accepted for the
21 pending HumanDB replays.

An isolated, clean checkout at commit
`db65eb3e73189d934d615d0f47519d395193c646`, tree
`b8fa6c0119c2dec4443efc59deab8b7d835e0c88`, was built with the repository's
pinned Rust 1.95.0 toolchain. The machine-local checkout is resolved only from
the ignored `sanmill_prefix12_checkout` path-registry key.

The accepted replay binary is `target/release/tgf.exe`, 4,109,312 bytes, with
SHA-256
`6502f7a2180769666c1ba6c801288a5ba079920e2bd6c1121f0e8b0c27e11e53`.
The complete machine-readable contract is
[stored beside this document](sanmill-prefix12-human-replay-runtime-2026-08-01.json).
Its runtime identity is
`645aab8157458ca9ff70f47fc39385c3ef28affda2e22c83e3a4d9ae84af1df8`.

## Historical-binary boundary

The historical strict-v2 smoke binary had SHA-256
`cac2ec6fe45a9d798a89c6b8a5f52c767aa1c885a1156a96269b44ebf81976cc`.
Those bytes are no longer available locally. Two builds of the same pinned
source and toolchain in different checkout locations produced the same file
size but different hashes. The project therefore does not claim that a fresh
build reproduces the historical binary byte for byte.

This runtime has a new, explicit identity. It does not rewrite or replace the
historical smoke evidence. Its sole current purpose is to replay the already
frozen 21 HumanDB histories, through two fresh strict data-query processes,
and record complete per-step history state. Any source, binary, protocol,
history, legality, FEN, count, or cross-process difference stops the replay.

No candidate model is loaded, no game is played, and no evaluation or training
is authorised by this runtime decision.

## Verification

The preserved checkout passed all 11 `strict_uci` cross-process tests and all
five `data_query` integration tests. NMM_LLM's runtime-contract tests also
verified the exact commit, tree, binary, clean-worktree policy, and identical
history summaries from two fresh processes. Sixty-one synthetic bridge and
prefix tests remained green.

Tests that intentionally inspect the moving `sanmill_checkout` still reject
that checkout because protected source paths have changed. They were not
skipped or weakened; the separate exact-head lookup key is the explicit
resolution for this replay workflow.

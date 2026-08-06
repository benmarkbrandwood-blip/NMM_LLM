# NMM_LLM MIF 1.0 independent adapter

Status: implemented against the frozen candidate wire contract; no MIF Suite
1.0 has been published, so this document makes no suite-conformance claim.

## Frozen source identity

The implementation is locked to MIF commit
`f37ddfeb5fb8479991fa38eeb03c797bef8ae408`. The formal frozen inputs and
additional comparison inputs at that commit have these raw SHA-256 identities:

| Input | Role | SHA-256 |
|---|---|---|
| `mif-1.0.md` | formal English specification | `330e65145ceb26fe582e58b89405d87bd73e8be200b476aef82c0ee27731d995` |
| `docs/zh-CN/mif-1.0.md` | formal Chinese specification | `9cc06abb57425e2bc2e26432b6da53abe503e9b5415ea0b4f854f19f68722cc1` |
| `artifacts/mif-1.0/index.json` | formal artifact index | `3849a70897829d6d994c790b64e63484469483a940887fe828a1a0d421d78e90` |
| `artifacts/mif-1.0/corpus/executable/reference-cases.json` | formal executable corpus | `a48c50352caebce30deb1de11f8f73dbc4540ee538651c3a139d9bcb166ba983` |
| `interop/adapter-protocol-v1.md` | additional process protocol | `a59e5e5af3e948f6c7cac6a39a490c6eae6338151741b6c7fcdde5c88d991e2d` |
| `interop/cases/smoke-v1.json` | additional smoke comparison corpus | `a6d292f4d19381172fbc19f89d3ee42145a6d5533d6d81fd719394e25342bb53` |

The adapter has no runtime dependency on the MIF repository. The checkout is
needed only by the external comparison harness and its test cases.
The command-array generator verifies the exact commit, rejects all worktree
changes, and hashes the formal four inputs plus the two additional execution
inputs before emitting a comparison configuration. A matching `HEAD` alone is
not accepted as locked evidence.

## Independence boundary

The NMM_LLM adapter implements MFEN parsing, the finite-rules transitions,
claim/repetition state, RFC 8785 identities, sparse-Merkle summaries, D4
transforms and complete logical-turn projection in
`learned_ai/interop/mif_v1/`.

It does not import, vendor or copy the MIF Python reference runner's gameplay
implementation. Interoperability verification may start that runner as a
separate black-box process, which is the isolation required by
`MIF-INTEROP/1`.

## Operation coverage

The process entry point is `tools/nmm_llm_mif_adapter.py`. It accepts LF-only
UTF-8 NDJSON and implements:

- `capabilities`;
- `canonicalize` for MFEN/1.0 and structural-d4-v1 MPK/1.0;
- `execute` and checkpoint-verifying `replay`;
- `transform` for MSTATE, MIFPOS and materialized decision state; and
- `project-logical-turns` for complete primary-plus-removal fragments.

The capability response advertises the two candidate corpus rulesets. The
executor also accepts manifest patches that remain inside their implemented
semantic subset and fails closed outside it. In particular, enabled
leap/intervention/custodian mechanisms, non-empty semantic-state extensions,
non-standard mill effects and non-loss/draw stalemate actions are not claimed.
This narrower, explicit claim is preferable to silently approximating a
variant with NMM_LLM's legacy `BoardState` rules.

The three published resource limits are executable contract, not descriptive
metadata. Request framing is capped at 16,777,216 bytes, event arrays at
100,000 entries and active/materialized repetition history at 100,000 entries.
The execute, replay, transform and logical-turn paths share those limits.
Every over-limit condition returns a `resource` MIFDIAG with the exact name,
limit and actual count; an oversized NDJSON record is drained with bounded
memory so the process can return a protocol response without accepting a
truncated request.

## Three independent command arrays

The comparison processes are launched with these command-array shapes:

1. MIF candidate reference:
   `["{python}", "-B", "tools/mif_1_0_reference_adapter.py"]`
2. NMM_LLM:
   `["{python}", "-B", "<NMM_LLM_ROOT>/tools/nmm_llm_mif_adapter.py"]`
3. Sanmill through Cargo:
   `["cargo", "run", "--quiet", "--manifest-path",
   "<SANMILL_ROOT>/Cargo.toml", "-p", "tgf-cli", "--", "mill",
   "mif-interop"]`

All three use the MIF checkout as the harness working directory. The adapter
commands themselves are absolute where they cross repository boundaries, so
neither implementation depends on the caller's current directory.

Generate a host-local, schema-valid three-party configuration while also
checking the exact MIF commit:

```powershell
.\.venv\Scripts\python.exe tools\mif_interop_adapter_commands.py `
  --mif-root <MIF_ROOT> `
  --sanmill-root <SANMILL_ROOT> `
  --output out\mif-interop-three-party.local.json
```

Then run the comparator from the locked MIF checkout:

```powershell
python -B tools\compare_mif_1_0_adapters.py `
  --config <NMM_LLM_ROOT>\out\mif-interop-three-party.local.json `
  --cases interop\cases\smoke-v1.json
```

For a prebuilt Sanmill binary, pass `--sanmill-binary <path>` to avoid a Cargo
startup build. The generated configuration remains machine-local under
`out/`; absolute host paths are never committed as project configuration.

## Current verification

The focused adapter set contains 45 passing tests. In addition to the frozen
candidate identities, it covers process framing, duplicate JSON names,
checkpoint audit, pre-origin claim preservation, placement and flying mills,
compulsory removal, material termination, full-state transforms and logical
turn grouping. It now also covers dirty/hash-mismatched source checkouts and
every operation path governed by the published resource limits.
Reference-derived constants were obtained through independent process calls,
not by importing the reference runner.

The previous M2 evidence compared all 16 cases at MIF commit
`83e4b758f624f3059c7ba289d4d4429eed0a710a`. The post-JCS baseline was then
verified on 6 August 2026 at these exact implementation identities:

- MIF reference and corpus commit
  `f37ddfeb5fb8479991fa38eeb03c797bef8ae408`;
- NMM_LLM binding and adapter commit
  `d631c54f56bcc736ebd5d59d896846d4d2fd485e`;
- Sanmill adapter commit
  `54623a6c5d66ffcbfb6e61ed295a20885ed7920d`.

At the clean NMM_LLM commit, all 45 focused tests and Ruff passed. The command
generator accepted the clean MIF checkout and all six hashes above, and the
official `smoke-v1.json` comparison passed all 17 cases across the three
independent adapter processes. The added JCS case required no NMM_LLM state
machine or canonicalization change; the existing implementation already
matched the corrected reference behaviour.

This closes NMM_LLM's post-JCS pin and comparison evidence. The comparator
labels the result candidate interoperability evidence, not suite conformance;
a three-project M3 freeze still requires the corresponding Sanmill pin and
evidence publication. A full 1,138-test NMM_LLM run was also attempted without
skips at the previous baseline, but reached the 15-minute command limit at
roughly 15% with no reported failure. It must not be described as a complete
repository-suite pass.

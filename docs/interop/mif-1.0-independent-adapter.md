# NMM_LLM MIF 1.0 independent adapter

Status: implemented against the frozen candidate wire contract; no MIF Suite
1.0 has been published, so this document makes no suite-conformance claim.

## Frozen source identity

The implementation is locked to MIF commit
`7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978`. The formal frozen inputs and
additional comparison inputs at that commit have these raw SHA-256 identities:

| Input | Role | SHA-256 |
|---|---|---|
| `mif-1.0.md` | formal English specification | `330e65145ceb26fe582e58b89405d87bd73e8be200b476aef82c0ee27731d995` |
| `docs/zh-CN/mif-1.0.md` | formal Chinese specification | `9cc06abb57425e2bc2e26432b6da53abe503e9b5415ea0b4f854f19f68722cc1` |
| `artifacts/mif-1.0/index.json` | formal artifact index | `5acbb714bed77e24eaac72fa5f24d2e54d1e17aaf568a8b60718c840281a6541` |
| `artifacts/mif-1.0/corpus/executable/reference-cases.json` | formal executable corpus | `350b7ff02772e820a57431e11c4e2f15a874d0779fb6e7afb01e9b16f6992741` |
| `interop/adapter-protocol-v1.md` | additional process protocol | `253c1d201ea1db625e0c534da445ca4ecaa0b07597dfc7dbf59fbd6adf89874f` |
| `interop/cases/smoke-v1.json` | additional smoke comparison corpus | `a6d292f4d19381172fbc19f89d3ee42145a6d5533d6d81fd719394e25342bb53` |
| `interop/cases/deterministic-v1.json` | candidate-4 deterministic comparison corpus | `d11317a090300f8a47f77afed647bdbd236dcdb1996c0147a81c874fa39dfd82` |

The adapter has no runtime dependency on the MIF repository. The checkout is
needed only by the external comparison harness and its test cases.
The command-array generator verifies the exact commit, rejects all worktree
changes, and hashes the formal four inputs plus the two additional execution
inputs and deterministic corpus before emitting a comparison configuration. A
matching `HEAD` alone is not accepted as locked evidence.

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
- `transform` for MSTATE, MIFPOS and materialized decision state;
- `project-logical-turns` for complete primary-plus-removal fragments; and
- `project-legal-actions` for the closed, canonically ordered
  `legal-actions-v1` harness projection.

The capability response advertises the two candidate corpus rulesets. The
executor also accepts manifest patches that remain inside their implemented
semantic subset and fails closed outside it. In particular, enabled
leap/intervention/custodian mechanisms, non-empty semantic-state extensions,
non-standard mill effects and non-loss/draw stalemate actions are not claimed.
This narrower, explicit claim is preferable to silently approximating a
variant with NMM_LLM's legacy `BoardState` rules.

The capability binds all seven candidate-4 source and harness identities in its
annotations. Its `testedCorpora` retains the 17-case smoke identity and adds the
58-case deterministic identity after the NMM_LLM adapter matched the separate
MIF reference process. The deterministic record covers identity, key,
position, replay, ruleset and transform. The separate `suites` array remains
empty because this evidence is not a published MIF Suite.

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
  --cases interop\cases\deterministic-v1.json
```

For a prebuilt Sanmill binary, pass `--sanmill-binary <path>` to avoid a Cargo
startup build. The generated configuration remains machine-local under
`out/`; absolute host paths are never committed as project configuration.

## Current verification

The focused adapter set contains 55 passing tests. In addition to the earlier
framing, identity, replay, resource, transform and logical-turn coverage, it
now fixes the candidate-3 MPK diagnostics, classifies a claim made during a
compulsory removal as inconsistent, and exercises every legal-action stratum.
The legal-action tests also expose and prevent phase-p flying: placing movement
may be enabled, but the frozen contract restricts flying to phase m. Ruff passes
for the complete focused implementation and test set.

The candidate-3 gameplay implementation is the three-commit chain ending at
NMM_LLM commit `121b663951fcc69e90e956d35c3d44d8118bb066`. Candidate-4 changes
only the locked identities at clean NMM_LLM commit
`bbbde2ee4bf1ba0e45e259baa595a29cb85895b9`; no gameplay code changed. Before
the pin was updated, the same implementation already matched all 58
candidate-4 cases, including the three asymmetric-reserve origin cases. At the
clean pin commit, 55 focused tests and Ruff pass, and the command generator
accepts clean MIF commit
`7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978` plus all seven hashes above.

The exact two-adapter comparator output is preserved as the
[candidate-4 NMM/reference report](../evidence/mif-interop-candidate-4-nmm-reference-report-2026-08-06.json),
SHA-256
`89dfcd97c914764aa95bcb5e6b6ecdb23686591037dbf8c5493fe8b3dfbc142f`.
It records 58/58 equality, cases digest
`sha256:d11317a090300f8a47f77afed647bdbd236dcdb1996c0147a81c874fa39dfd82`
and machine-local config digest
`sha256:1d04f6f2f775239110ff00a1f97bb129fe13f1d903dd284f6f3905810b1b7889`.
The generated configuration remains ignored because it contains host paths.

This closes the NMM_LLM side of the candidate-4 pin and deterministic
comparison. It does not complete three-project M3: the persisted report
intentionally contains only the published MIF reference and clean NMM_LLM
commit. Sanmill is still published at its candidate-3 commit
`6f56c8efcba753001d8e07398c8c262d2aa6c481`; a new three-party 58-case
report must bind a later Sanmill candidate-4 pin and all three immutable
commits. Neither result is MIF Suite conformance. The historical
[candidate-3 report](../evidence/mif-interop-candidate-3-nmm-reference-report-2026-08-06.json)
and
[candidate-2 report](../evidence/mif-interop-candidate-2-report-2026-08-06.json)
remain valid only for their recorded identities.

A full 1,138-test NMM_LLM run was attempted without skips at the previous
baseline, but reached the 15-minute command limit at roughly 15% with no
reported failure. It must not be described as a complete repository-suite
pass.

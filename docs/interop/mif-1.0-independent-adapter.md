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
comparison. Sanmill subsequently published its candidate-4 adapter and
evidence at commit `e6d639d41f079b15ca697268d0c2c21dad5c2bc3`. Its tracked
`interop/evidence/mif-interop-candidate-4-three-project-report-2026-08-06.json`
has SHA-256
`895c04cd69fc00e50bdcd349b150293e52fcc4150c63321d8c9771015f70aaaf`
and records 58/58 across the MIF reference, Sanmill, and NMM_LLM. It records
cases digest
`sha256:d11317a090300f8a47f77afed647bdbd236dcdb1996c0147a81c874fa39dfd82`
and config digest
`sha256:4184d56c696b2e5031d95cc18918757af9f91fa5634dded579ecfae2ef3cf70f`.
An independent rerun at MIF
`7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978`, Sanmill
`e6d639d41f079b15ca697268d0c2c21dad5c2bc3`, and NMM_LLM
`11bebd14e0d538a41a4b43aebfe57ee74c2a2601` reproduced the same report
hash.

Sanmill closed that M3 evidence-chain gap at evidence commit
`9431b95f151502f415f096c7d96ca944e5d578de`. Its companion manifest binds
MIF `7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978`, Sanmill
`e6d639d41f079b15ca697268d0c2c21dad5c2bc3`, NMM_LLM
`11bebd14e0d538a41a4b43aebfe57ee74c2a2601`, all seven candidate-4 input
hashes, and the exact 58/58 report identity above. M3 is therefore closed for
those recorded candidate identities. None of these results is MIF Suite
conformance. The historical
[candidate-3 report](../evidence/mif-interop-candidate-3-nmm-reference-report-2026-08-06.json)
and
[candidate-2 report](../evidence/mif-interop-candidate-2-report-2026-08-06.json)
remain valid only for their recorded identities.

## Candidate-4 M4 differential evidence

M4 uses the unchanged candidate-4 wire commit
`7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978` through the separately published
MIF launch commit `40718e80d36ec9c060fc17997568d637a74e6d9f`. The fixed launch has
SHA-256
`560ef369fde248bd96d3468a4336442db1d970ede04f488821509e69925fd48e`
and transitively binds 29 resources. The reference baseline has SHA-256
`29d198dbcf8221fa0235af6a72db9d6a82646b45fc653c584071821a9a4bb61b`.
Neither file, its seeds, nor its mutations was regenerated or edited here.

The pre-fix NMM_LLM commit
`e2ab05d29885af9a16a9aa5d5f62b1517cf6d91b` reproduced the prescribed
three-party preflight: 10/10 seeded runs, 3/5 negative mutations, and config
digest
`sha256:4184d56c696b2e5031d95cc18918757af9f91fa5634dded579ecfae2ef3cf70f`.
The two failures were diagnostic-shape differences rather than state or move
differences. Tested implementation commit
`6c1538082fc551203d827782d137a5799c810535` classifies an attempted removal
without a pending obligation as `inconsistent`, retains code
`remove-without-obligation` and its event sequence, and omits non-contract
`expected` and `actual` members from checkpoint, repetition-history, and claim
replay mismatches. It changes no transition, replay, legal-action, or other
gameplay semantics.

At that clean, pushed implementation commit, 62 focused MIF tests pass. Ruff,
a focused mypy check for the new M4 regression module, and bytecode compilation
also pass. The candidate-4 deterministic corpus remains 58/58 across the MIF
reference, Sanmill, and NMM_LLM. A source scan found no runtime import of the
MIF reference runner by the NMM_LLM adapter or entry point.

The exact two-adapter output is preserved as the
[M4 Reference/NMM_LLM report](../evidence/mif-interop-candidate-4-m4-reference-nmm-report-2026-08-07.json),
SHA-256
`2bc434699902a1c468b604797d4456ee0c968817b057ec4dc8254a623a1ba64c`.
It records 10/10 seeded runs, 5/5 negative mutations, launch digest
`sha256:560ef369fde248bd96d3468a4336442db1d970ede04f488821509e69925fd48e`
and machine-local config digest
`sha256:c6eb5edc21773c017e7a2d5d9050b38cb08450658a286e64a395f1edc6b7074e`.
The adjacent
[evidence manifest](../evidence/mif-interop-candidate-4-m4-reference-nmm-evidence-manifest-2026-08-07.json)
binds the report to the exact MIF and NMM_LLM commits and records the external
Sanmill implementation commit
`ae9a1d8a16261478631a3a7583cbf35c7b6e0df5`, evidence commit
`9431b95f151502f415f096c7d96ca944e5d578de`, and Reference/Sanmill report
SHA-256
`0135ba7778a4623cecc0fe07173f50d76d3f06b6afd7830269b2c01e168604a7`.

The final three-party preflight at these published external identities and the
tested NMM_LLM implementation also records 10/10 seeded runs and 5/5 negative
mutations with config digest
`sha256:4184d56c696b2e5031d95cc18918757af9f91fa5634dded579ecfae2ef3cf70f`.
This closes NMM_LLM's Candidate-4 M4 differential evidence only. It is not,
and must not be cited as, MIF Suite 1.0 conformance.

A full 1,138-test NMM_LLM run was attempted without skips at the previous
baseline, but reached the 15-minute command limit at roughly 15% with no
reported failure. It must not be described as a complete repository-suite
pass.

## Suite 1.0 final adapter pin

The independent adapter is now bound to the finalization baseline at MIF
commit `3ee7e57c7d4c7208be91f62914f344a587fb0f70`, while the implemented wire
semantics remain pinned to commit
`7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978`. The Suite raw-file SHA-256 is
`088ca33234289b06d9276aa4c430758222aa85d61621dee7bef4bfc6dcc069a4` and
its RFC 8785 JCS SHA-256 is
`81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f`.
The clean, pushed NMM_LLM implementation commit is
`a7e7dbd5461cc2d8d8c0a09317d6091598202214`.

The published `MIFCAP/1.0` document lists that Suite JCS identity, marks only
`identity`, `key`, `position`, `replay`, `ruleset`, and `transform` as
`tested`, and binds the two Suite ruleset semantic digests. It retains
`conversion=none`, has no `full` class, and makes no conversion claim. Its
[raw capability](../evidence/mif-suite-1.0-nmm-capability-2026-08-07.json)
has SHA-256
`cd661b1156bf7269f976e050446d01797c9959482f1e1843e21ae3ea7f70dcce`.

The exact Reference/NMM_LLM finalization reports are preserved as follows:

- the [58-case deterministic report](../evidence/mif-suite-1.0-nmm-deterministic-report-2026-08-07.json)
  has SHA-256
  `3463f438531fd52847df44fa4186dcba13ed22c7c570a0cc216d9a7eaa797665`;
- the [differential report](../evidence/mif-suite-1.0-nmm-differential-report-2026-08-07.json)
  has SHA-256
  `4c86725bfcd1759433374938c8d8eb2a1dacfa6ea3723592eff759162fce8da6`
  and records 10/10 seeded runs plus 5/5 mutation families; and
- both reports use config digest
  `sha256:c6eb5edc21773c017e7a2d5d9050b38cb08450658a286e64a395f1edc6b7074e`.

All three raw artefacts were generated twice and were byte-identical. A final
three-adapter preflight against the current published Sanmill Suite adapter
also passed 58/58, 10/10, and 5/5 with config digest
`sha256:133cc572ba786ebd544e9fe5fc89c67248432952a1a2fce451a3e1ec6bfda0f2`.
The
[Suite-bound evidence](../evidence/mif-suite-1.0-nmm-adapter-evidence-2026-08-07.json)
records the exact commits, raw identities, tested domain, and zero unexplained
differences under protocol `MIF-SUITE-ADAPTER-EVIDENCE/1`.

The focused MIF set passes 66 tests; Ruff, focused mypy, and bytecode
compilation also pass. The complete 1,179-test repository collection was
exercised in four process-isolated shards. Of those tests, 1,170 passed on the
first run, one Windows Chroma SQLite cleanup failure passed when rerun alone,
and eight machine-local Sanmill tests remained intentionally fail-closed
because the historical strict-v2 binary bytes are documented as unavailable.
No failure reaches the changed MIF subsystem, but this is not a clean full
repository-suite result.

The evidence classification is `exact-for-tested-domain`. Its
`suiteConformance=true` applies only to the six declared classes, two declared
rulesets, and fixed Suite corpus. It is not `full` conformance and does not
claim `conversion`. The finalization notice is still a candidate-Suite gate,
not the immutable signed Suite release. Engineering smoke additionally needs
an independent experiment digest; formal long-running archival training
remains unauthorized until the signed tag and both Suite-bound adapters are
accepted.

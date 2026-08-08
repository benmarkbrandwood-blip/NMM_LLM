# Sanmill Frozen-Target Index 31 Reproduction v1 Result — 8 August 2026

## Verdict

`terminal_turn_fix_reproduced_successfully; remaining_controls_required`

The exact immutable parent index 31 completed after the terminal-turn mirror
fix. The captured frozen-target game reached the same 85-ply terminal state
without a mirror error. The one-run authority is consumed; this result is not
training or strength evidence.

| Field | Value |
| --- | --- |
| Run ID | `sanmill-no-update-frozen-target-index31-reproduction-v1-20260808-001` |
| Published source | `3260cab13c1ecbd8326e8f0f9040c922f0287f6d`; clean `dev == origin/dev` |
| Plan identity / raw SHA-256 | `8aa3d733d20fa58fa105eee992cab6f000507e22ffe59a97767681425d7259ec` / `65fb386c42548fbde49ef066a5de70c05f7a3dfc38371dafdeb7a49419818e71` |
| Report identity | `6754f5ab46f7df8d64ffdc37e89b1709d7424897caa7103f34ececf10afd1710` |
| Raw result SHA-256 / size | `f3f70495084ce65b7360f0edcc56989cb5b7b94d3c0c8fec690f089d4d1c4d08` / 62,554 bytes |
| Parent entry | index 31; `frozen-target-normal-0-B`; seed `5768742839362539388` |
| Result | learner loss; `lose_fewer_than_three`; White winner |
| Route | 85 logical plies; 42 learner steps; 12 compound turns; zero opponent search calls |
| Final history | 97 action tokens; SHA-256 `663908d2ee6f37fdc6c347191a323c25c8bd3b584dd0ec0dd0948dfa350ccec2` |
| Timing | 21.414348 game seconds; 0.251934 seconds per logical ply |
| Integrity | completed result present; failure result absent; no remaining `tgf` process |

The result validator independently recomputed the stored report identity and
verified the exact one-entry schedule. The learner and frozen target both
remained byte-identical at
`15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00`.
There were zero backward calls, optimizer constructions, checkpoint writes,
or rollout persistence. HumanDB, SpecialistDB, Malom, and ruleset records were
identical before and after the run.

The final Sanmill state was terminal with `side_to_move=null`, winner White,
and reason `lose_fewer_than_three`. Its board, placement counts, terminal
winner, and complete action history matched the local mirror. This directly
reproduces the captured failure path and supports accepting commit
`bb4fe56e1af4488df0d6a8338e0ff1114f5f9e6c` as the adapter-level root-cause
fix. It does not alter or validate training updates.

## Next gate

Parent indices 32 through 35 are the remaining frozen-target controls. They
must complete under a new bounded, no-retry continuation before the
no-update integration route can be closed. Only then may an update-capable
training smoke be designed. No long training is authorized by this result.

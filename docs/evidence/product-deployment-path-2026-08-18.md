# Product Deployment Path Audit

Date: 2026-08-18

Scope: read-only repository and bounded host inspection

Inspected repository tips:

- `origin/dev`: `30390bf050e2ae30fd402c6277266ae709c75b28`;
- `origin/main`: `4e4a7241e9d5427100b46dfe34f5ae384ff9f613`;
- common ancestor: `67af016381ee499ae89c5e8710c653fbe74a1875`.

No product service, application process, game, model load, deployment,
checkpoint change, database write, or protected-data access was performed.
The investigation covered the repository, its parent container, registered
Git worktrees, selected common project roots, GitHub repository metadata,
Windows service/startup registries, ports, process names, project logs, shell
history, and exact localhost browser-history rows. It was not a full-disk scan.

## 1. Confirmed

### Direct answers

At the inspection time, there was no detected local instance of the tracked
product runtime. Nothing was listening on TCP 8000 or 8080, and no `python`,
`pythonw`, or `uvicorn` process was running. Therefore no detected product
instance had loaded a checkpoint, and `30390bf` was not active through the
repository's recorded user-facing launch path.

The only bounded, self-contained tree on this host that contains all three of
the following together is `I:\Mill_Training\NMM_LLM` on `dev`:

1. the documented Windows launcher and Web/FastAPI host;
2. all three active specialist checkpoint files; and
3. the `30390bf` position-level Malom safety filter.

That makes it the only presently complete *candidate* manual product source.
It does not prove that the product owner has launched it. The machine evidence
for past launch is negative rather than affirmative, as detailed below.

If this exact `dev` tree is launched without changing its current runtime
configuration, the filter code is reached during application import but is
disabled. `web/app.py` reads `malom_db_path` from tracked
`data/settings.json`; that value is a stale WSL-style path which does not
exist on this Windows host. The valid machine-local Malom path is present in
ignored `data/training_paths.local.json`, but `web/app.py` does not read that
registry. Its secondary Sentinel fallback is empty. The resulting declared
behavior is therefore:

- specialist `A_pos` filter disabled at startup;
- `/api/overseer_status` reports `playable=false` and the reason;
- difficulty 9/10 uses the classical coordinator result instead of an
  unfiltered specialist result.

Consequently, `30390bf` is present in the likely manual source tree but is not
currently running and would not be operationally enabled under the current
Web settings.

To make the existing `dev` path reach a user with filtering active, an
operator must:

1. select `I:\Mill_Training\NMM_LLM` at `30390bf` or a verified descendant;
2. configure the Web host with the existing valid
   `sector-corrected-v1` Malom location, either through the supported settings
   surface or a separately reviewed resolver change;
3. launch that tree with `run_nmm.bat`; and
4. verify that `/api/overseer_status` reports `playable=true`, the expected
   manifest identity, and no startup failure.

None of those runtime actions was performed in this audit.

### Branch topology and integration history

`origin/main` is the GitHub default branch, but it is not an ancestor of
`origin/dev`, and `origin/dev` is not an ancestor of `origin/main`. Their
common ancestor is `67af016` from 22 July 2026. Current divergence is:

| Side | Unique commits after the common ancestor |
| --- | ---: |
| `origin/main` | 149 |
| `origin/dev` | 650 |

The 22 July integration audit records two temporary integration merges:
`8717f1c` imported the then-current `main` checkpoint and product material,
and `4593034` imported the later `67af016` persistence change while retaining
the `dev` safeguards. That document is an integration and provenance audit;
it explicitly does not authorize publication or deployment.

The 7 August and 11 August audits record that later `main` commits were
reviewed, not merged wholesale. They classify later UI, Generalist, GapNet,
and trainer changes separately and do not define a release pipeline.

The governing product decision says the Windows Web/FastAPI application is
the current integration and measurement host, while the release device,
local-versus-server architecture, and final deployment form remain open. This
is consistent with a manually launched development worktree, not a frozen
external release architecture.

### What `main` and `dev` contain

`dev` contains the three specialist checkpoints, the enabled difficulty 9/10
specialist route, the position-only safety module, status/UI diagnostics, and
the real-downgrade regression fixtures introduced by `30390bf`.

Current `main` contains none of the three specialist checkpoint files and
does not contain `learned_ai/agents/positional_safety.py` or its tests. More
importantly, this is not merely missing packaging:

- commit `3940de8` deleted the old specialist checkpoint files from `main` on
  1 August while adding later Generalist/HumanPolicy artifacts;
- commit `6bb3216` had already hard-coded `_spec_mode = False`, hidden the
  Specialist AI control, and described the specialist path as a failed
  experiment on 28 July;
- difficulty 9/10 on `main` is presented as classical deep search rather than
  a specialist route.

Thus applying `30390bf` to `main` would reverse a recorded product decision,
not merely copy a safety helper.

### Specialist checkpoint identity and transport

The three files on `dev` are ordinary Git blobs:

| Phase | Path suffix | Bytes | SHA-256 | Git blob |
| --- | --- | ---: | --- | --- |
| opening | `s_open_v2/best.pt` | 523,389 | `d020e1442676e16cdced6c91dac958817c3a22a283cc293d6e19930a87703701` | `8ef5e0c71cf9591b966f451d6425dcaad2ff800e` |
| movement | `s_mid_v2/best.pt` | 523,389 | `a587ab995224a1d43c99fd2f42e4bff9c060ac6da55edcddb43a39fc07ef26d2` | `3a71153cf4c901ff9ff083e51163beab7f40c7c1` |
| endgame | `s_end_v2/best.pt` | 523,389 | `5de51a1afd5794374d4394cce2950957a23f02504b5c5952a062d91414b94be8` | `d316c891320735a833d92203581e08b52bb335b8` |

Their combined size is 1,570,167 bytes. There is no `.gitattributes` file and
`git lfs ls-files` is empty, so these are not Git LFS pointers.

Commit `89e96ab` added both the checkpoint files and explicit `.gitignore`
negations for the reviewed `s_open_v2`, `s_mid_v2`, and `s_end_v2` trees.
Commit `1335536` updated the files, and merge `8717f1c` carried those updated
blobs into `dev`. This is the explicit tracked exception to the default rule
that generated checkpoints are ignored.

The `main` ignore exceptions still exist, but the files themselves do not.
Neither installer nor launcher downloads specialist checkpoints. The
repository has no checkpoint-fetch script, no LFS inventory, no GitHub
Release, and no package manifest that supplies them. A clean current `main`
clone therefore cannot obtain these three files through any recorded product
setup step.

### Launch and packaging surface

The tracked product launch surface is manual:

- `run_nmm.bat` derives its working directory from its own location;
- it runs `.venv\Scripts\python.exe -m uvicorn web.app:app`;
- it binds only `127.0.0.1`, normally port 8000, falling back to 8080 if the
  port is occupied;
- it polls `/api/ping` and opens the local browser;
- `run_nmm.sh` is the analogous non-Windows launcher;
- `install.bat` and `install.ps1` create an in-tree virtual environment and
  install dependencies. They do not install an application service or copy a
  release artifact elsewhere.

No Dockerfile, Compose file, Windows service definition, Procfile, package
entry point, GitHub workflow, or tracked deployment manifest exists. There
are zero Git tags. A live GitHub metadata query found:

| GitHub surface | Count |
| --- | ---: |
| Releases | 0 |
| Deployments | 0 |
| Actions workflows | 0 |
| Environments | 0 |

Webhook metadata was inaccessible to the available token and GitHub Pages
could not be distinguished from unconfigured versus inaccessible. Neither is
reported as absent.

### Runtime data and independent-copy requirements

The Web host anchors most runtime paths to the directory containing
`web/app.py`. Important paths include:

- `data/logs/server.log`, `data/games`, and autosave state;
- repository-local HumanDB, full-game and solved-endgame data;
- repository-local Sentinel, ValueNet, GapNet, and specialist checkpoints;
- templates, static files, sounds, openings, and personalities;
- `data/settings.json` for product paths, including Malom;
- Ollama at localhost as an optional commentary dependency.

The Web host does not consume `data/training_paths.local.json`. That ignored
registry exists on this machine and points to valid repository, parent-
container, and external research resources, but copying it to another host is
not part of any product installer. An independent deployment copy would need
its own data files and a valid product Malom setting; possession of source
code alone is insufficient to enable the safety filter.

### Bounded host evidence

The inspected host contains these relevant trees:

| Location | Identity | Checkpoints | `30390bf` filter | Product-use evidence |
| --- | --- | --- | --- | --- |
| `I:\Mill_Training\NMM_LLM` | `dev` at `30390bf` | all three exact hashes | yes | no current process or server log |
| registered `phase-heldout-prep` worktree | detached `7832fe1` | same three hashes | no | research worktree; no server log |
| `D:\Repo\NMM_LLM_old` | old `calcitem/NMM_LLM` clone at `01fc132` | none | no | no server log |

The registered detached worktree contains untracked held-out research files
and predates the filter. It is not a clean deployment candidate. The old
`D:` clone points to a different GitHub remote, lacks all three specialist
files, lacks the filter, and has no machine-local path registry.

The parent `I:\Mill_Training` contains no second directory with the complete
project markers. A bounded one-level scan of Desktop, Documents, Downloads,
`C:\Projects`, `D:\Repo`, `D:\Projects`, and `I:\` found no further
NMM_LLM product copy. Other Morris/NMM repositories under `D:\Repo` do not
contain this product's launcher and Web application pair.

A Windows Recent shortcut named `NMM_LLM` points to the current
`I:\Mill_Training\NMM_LLM` tree. That proves recent navigation to this tree,
not application launch.

The following current or historical launch evidence was absent:

- no listener on TCP 8000 or 8080;
- no running Python or Uvicorn process;
- no matching Windows service registry value;
- no matching current-user or machine startup entry;
- no matching installed-program record or product directory under Program
  Files;
- no `server.log` in the current tree, detached worktree, or old clone;
- each inspected `data/games` directory contains only `.gitkeep`;
- no matching launch command in PowerShell history;
- Bash history records cloning `benmarkbrandwood-blip/NMM_LLM`, creating
  `dev`, and pushing it, but no product launch command;
- no exact localhost 8000/8080 URL in the available Chrome or Edge history.

`web/app.py` creates `data/logs/server.log` and writes `=== Server started ===`
at import time. The missing log is therefore meaningful negative evidence,
but it cannot rule out manual deletion, an uninspected browser/profile, or an
external host.

### What a `main` integration would require

If `origin/main` is selected as the release source, integration cannot be a
fast-forward. At minimum it requires:

1. an explicit product decision reversing or retaining `6bb3216`, which
   removed the specialist mode as a failed experiment;
2. a manual port of the position-safety module, trusted Malom startup gate,
   router diagnostics, status API, UI state, and regressions;
3. conflict resolution against `main` in the handoff, router imports,
   specialist activation block, and Specialist UI. A read-only three-tree
   application audit produced four conflict regions;
4. a deliberate checkpoint transport decision: restore the three reviewed
   ordinary Git blobs, or create a separately identified model package with
   the exact hashes above. The current installer supplies neither;
5. a valid deployed `sector-corrected-v1` Malom path and inventory manifest;
6. verification that `main`'s later Generalist, HumanPolicy, UI, settings, and
   checkpoint-loader changes do not alter the specialist contract;
7. focused route, failure-mode, real-downgrade, startup, and UI tests on the
   resolved `main` tree; and
8. an actual delivery mechanism. Merging code into the default branch alone
   would not deploy anything because no release or deployment workflow exists.

No such integration or deployment was attempted.

## 2. Excluded

The following accounts are contradicted by inspected evidence:

1. **The repository's tracked local product server is currently running.**
   Excluded by the empty 8000/8080 listener set and absence of Python/Uvicorn
   processes.
2. **This host runs the product as a registered Windows service or ordinary
   startup item.** The complete Windows service registry search and the
   current-user/machine Run plus Startup-folder search produced zero matches.
3. **The product was installed as a normal Windows application under Program
   Files.** The matching uninstall-registry searches and exact program
   directories were absent.
4. **GitHub automatically deploys this repository through a recorded native
   pipeline.** Releases, Deployments, Actions workflows, and Environments are
   all zero, and the tracked tree has no deployment workflow or container
   manifest.
5. **A clean current `main` clone automatically receives the three active
   specialist checkpoints.** The blobs are absent from `main`; LFS, Releases,
   installers, and download scripts provide no replacement.
6. **Current `main` already exposes the specialist route fixed by `30390bf`.**
   `main` lacks the module and tests, hides the UI, and hard-codes the route
   off.
7. **The old `D:\Repo\NMM_LLM_old` clone supplies the filtered product.** It
   lacks the filter and all three specialist files.
8. **The current `dev` Web settings would enable Malom filtering as-is.** The
   selected product path is absent, the correct ignored registry is not read,
   and the fallback configuration is empty.

These exclusions do not prove that no unrecorded external host, administrator-
only scheduled task, deleted log, or manually maintained copy exists.

## 3. Only the product owner knows

The remaining questions cannot be answered from the available repository and
host evidence. They are intentionally formatted for direct selection.

1. Which source has ever been used for the user-facing game?

   - [ ] No user-facing instance has been launched yet.
   - [ ] `I:\Mill_Training\NMM_LLM` on `dev`, via `run_nmm.bat`.
   - [ ] A clean/default-branch `main` checkout.
   - [ ] The registered detached `phase-heldout-prep` worktree.
   - [ ] `D:\Repo\NMM_LLM_old`.
   - [ ] Another local path or external URL: ____________________.

2. What is the intended release source?

   - [ ] There is no release source yet; the Windows worktree is only an
     integration/measurement host.
   - [ ] `origin/dev`.
   - [ ] `origin/main`.
   - [ ] Another repository or artifact: ____________________.

3. Does any external machine, cloud service, administrator-only scheduled
   task, or different Windows account launch this product?

   - [ ] No.
   - [ ] Yes; host/task/account: ____________________.
   - [ ] Unknown.

4. If `main` is the release source, should its recorded decision that
   Specialist AI was a failed experiment be reversed?

   - [ ] No; keep difficulty 9/10 as classical deep search. In that case
     `30390bf` should not be integrated as an active product route.
   - [ ] Yes; restore a specialist product mode and port the safety filter and
     exact checkpoint identities under a new integration decision.

5. How should the Windows product obtain its Malom path?

   - [ ] Continue using `data/settings.json`, configured per installation.
   - [ ] Add a reviewed product resolver for the existing machine-local path
     registry or environment variable.
   - [ ] Do not ship direct Malom; keep specialist filtering unavailable and
     use the visible classical fallback.

6. Were the missing `server.log` files deliberately deleted, or was the game
   used through an uninspected browser profile/incognito session?

   - [ ] No; the Web product has not been launched from the inspected trees.
   - [ ] Yes; details: ____________________.
   - [ ] Unknown.

The Windows Scheduled Tasks API and TaskCache registry were inaccessible to
the available account. This is why question 3 remains open rather than being
reported as an excluded scheduled-task deployment.

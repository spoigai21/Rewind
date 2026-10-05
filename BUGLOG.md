# Bug log

Every bug that could have produced a wrong number, what it would have done, and what caught it.
Kept because a measurement is only as trustworthy as the list of ways it was nearly wrong.

| # | Phase | Bug | Effect if missed | Caught by | Fix |
|---|---|---|---|---|---|
| 1 | 0 | `git_info()` ran `git status --untracked-files=no`, so brand-new source files never set `dirty` | A result produced by code that exists only on one laptop would claim to come from a clean commit, breaking traceability | Reading the results JSON: `dirty: false` while the code had never been committed | Check all files, excluding only `results/` |
| 2 | 0 | Training at L=1024, batch 128 needed ~11 GB on a 16 GB machine; macOS swapped to disk | Step time jumped x192 for one doubling (56 s/step). Fed into an extrapolation, it would have projected an absurd GPU bill | The x192 jump between neighbouring lengths, and a 22-80 s spread between steps of identical work | Abort any length where a single step exceeds `--max-step-s`; flag rows whose median step is >1.5x the fastest |
| 3 | 0 | The too-slow guard from #2 only timed the first step; at L=1024 the first step was fast and later steps swapped | Same as #2: a swapping row recorded as a measurement (36 s/step) | The rerun still produced a 36 s/step row with the guard in place | Check every step, warmup included |
| 4 | 0 | The first length measured in a process absorbed one-off GPU start-up cost (program compilation, memory pools) | L=16 measured slower than L=32, bending the fitted curve and giving the per-token term a negative coefficient | A physically impossible result: shorter history costing more | Run one throwaway length before measuring |
| 5 | 0 | A quadratic fit to laptop timings under-predicted cost at L=2048 by 25-50% | The GPU budget would have been set too low, with the longest (most expensive) length the most wrong | Out-of-sample check: L=2048 was measured separately at batch 8 and compared with the fit | Projection uses direct measurements at every sweep length; the fit is kept only to describe the curve |
| 6 | 0 | `json.dumps` crashed on a NumPy bool | Crash, no wrong number | The crash | Convert to Python types before writing |
| 7 | 0 | `git_info()` was read when a sweep finished, not when it started | Editing any file during a 10-minute sweep would mark a clean run dirty, or, worse, a run started from uncommitted code could be committed mid-sweep and then recorded as clean | Noticed while planning to write new files during a sweep | Read git state at the start of the run |
| 8 | 1 | `.gitignore` line `data/` matched every folder named `data`, including the source package `rewind/data/` | The Phase 1 code would have been silently left out of every commit; the repo would not run from a fresh clone | `git status` did not list the new `rewind/data/` files | Anchor the pattern to the repo root: `/data/` |

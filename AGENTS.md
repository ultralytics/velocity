# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

Ultralytics Velocity (AGPL-3.0) estimates vehicle speed from imagery using Structure from Motion: it tracks a license plate of known physical dimensions across frames, recovers camera pose from those correspondences, and differentiates the resulting positions over EXIF timestamps. The repository holds a Python implementation, an earlier MATLAB implementation, and a small sample dataset — it is research code, not a package.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
pip3 install -U -r requirements.txt # numpy, scipy, torch, opencv-python, exifread, bokeh
python vidExample.py                # main pipeline over data/IMG_4134.MOV, writes "bokeh plots.html"
```

```matlab
runExample  % MATLAB pipeline; needs https://github.com/ultralytics/functions-matlab on the path
```

There is no test suite. CI is `.github/workflows/format.yml` (Ruff, docformatter, Prettier, codespell auto-applied to PR branches) and `cla.yml`.

## Architecture

`vidExample.py` is the single Python entry point and drives everything in one loop over frames:

- `utils/images.py` reads EXIF (`importEXIF`, `fcnEXIF2LLAT` for lat/lon/alt/time) and returns per-device intrinsics from `getCameraParams`; camera calibrations live in `matlab/*.mat` files keyed by source filename.
- `utils/KLT.py` tracks the plate region between frames (`KLTmain`, `KLTregional`) on top of `cv2.calcOpticalFlowPyrLK`, with a SURF-based affine fallback.
- `utils/NLS.py` solves camera pose by nonlinear least squares (`estimateWorldCameraPose`, `extrinsicsPlanar`, `fcnNLS_*`) against the known plate geometry from `utils/common.py:worldPointsLicensePlate`.
- `utils/MSV.py` performs the multi-frame scene-velocity refinement invoked once at frame 5 (`fcnMSV1_t`), triangulating tracked points to update the 3D point cloud.
- `utils/common.py` and `utils/transforms.py` hold the shared math (image↔world projection, sigma rejection, rpy/DCM/quaternion conversions); `plots.py` renders the Bokeh result page.

`matlab/` is the original implementation of the same pipeline (`runExample.m` plus `matlab/functions/`) and is kept for reference; `data/` holds the sample stills and clips it was validated against.

## Conventions

- Every Python and MATLAB file starts with the `Ultralytics 🚀 AGPL-3.0 License` header — Ultralytics Actions adds it automatically; don't add or revert it manually.
- Function names deliberately mirror the MATLAB originals (`fcn*` prefix, camelCase); keep new Python helpers consistent with their MATLAB counterpart rather than renaming one side only.
- `plots.py` targets the Bokeh 1.x API (`plot_width`/`plot_height`, `legend=`); it will not run unmodified on Bokeh 3.x, and `requirements.txt` does not pin a version.
- `utils/vid2images.py` is a personal frame-extraction utility with a hardcoded local path and runs on import — it is not part of the pipeline.

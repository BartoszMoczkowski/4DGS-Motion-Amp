# scratch/ — unsupported one-off analysis & experiment scripts

Everything here is a frozen, as-is working script: one-off analyses, abandoned
experiment variants, and debugging harnesses that were useful once and are kept
for reference. They are:

- **not installed** as workspace package modules (run them with plain
  `uv run python scratch/<script>.py` from the repo root, at your own risk),
- **not maintained** — they may reference moved files, stale paths, or old APIs,
- **frozen as-is** — including any `sys.path` hacks they already contained
  (unlike installed packages, scratch is exempt from the no-`sys.path` rule).

Scripts moved here from `scene-gen/` in T26 (2026-10): `analyze_complexity_ladder.py`,
`partition_swap.py`, `re_render_grid_runs.py`, `render_seg_visualizations.py`,
`run_grid_identity.py`, `identity/` (the identity-encoding experiment package, incl. its
torch-dependent `model`/`train` modules), `run_grid_multicut_calibrated.py`,
`run_partition_swap_exp.py`, `run_pump01_boundary_refine.py`, `run_pump01_spatial_cc.py`,
`show_cubes_seg.py`. Note `run_grid_identity.py` imports the `identity` package as
top-level modules — run it from inside `scratch/` (or with `scratch/` on `PYTHONPATH`).

Recurring scene-generation/experiment drivers stayed in `scene-gen/` and become `amp exp`
commands in T27.

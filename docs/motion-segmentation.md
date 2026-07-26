# Motion segmentation: adapting MultiBodySync to 4DGS

Compiled from `.claude_notes/NOTES_4dgs_motion_segmentation.md` (full detail there).

## The core insight

MultiBodySync (MBS) solves a problem 4DGS doesn't have. MBS's front two-thirds (FlowNet scene-flow estimation + confidence + weighted permutation synchronization) exist to establish correspondence between **unordered, uncorresponded point-cloud scans**. In 4DGS, the scene is one canonical set of N Gaussians plus a deformation field `D(x, t)` — the same Gaussian keeps its identity at every timestep, so dense per-point trajectories `p_i(t)` come for free at arbitrary temporal resolution. The task collapses to **clustering known trajectories into rigid motion groups**. The only genuinely reusable part of MBS is its motion-segmentation core (MotNet affinity + spectral segmentation synchronization + per-group Kabsch fitting).

Key mismatches beyond correspondence: MBS assumes piecewise-rigid bodies and N ≈ 256–1024 points (its permutation sync is O((KN)²) — infeasible at 4DGS's N > 10⁵); 4DGS scenes have floaters/low-opacity noise and continuous, possibly non-rigid deformation.

## Adaptation options

- **Option A — reuse MBS's MotNet head, drop flow + permutation sync.** Feed exact analytic flow (`p(t_l) − p(t_k)`) into MotNet with identity permutations. Medium effort; carries the rigid assumption, needs the `ext/` CUDA ops and a downloaded checkpoint, with a known out-of-distribution risk (MotNet was trained on noisy FlowNet flow at unit scale).
- **Option B — lightweight trajectory clustering, MBS as reference only.** Pure numpy/scipy, no GPU for the clustering itself. Chosen as the first baseline.
- **Option C — full MBS retrain end-to-end.** Highest effort, deferred.

## Implementation: `motion_seg/`

- `extract_trajectories.py` — data adapter (needs the GPU env). Loads a trained model, samples `get_state_at_time` at T evenly-spaced times, writes `trajectories.npz` (`canonical_xyz`, `traj (N,T,3)`, `opacity`, `times`).
- `rigidity_graph.py` — Option B core (pure numpy/scipy): k-NN graph in canonical space; per-edge rigidity score = std-dev of pairwise distance over time (exactly 0 for a true rigid pair); **log-space Otsu** auto-threshold; connected components → segments; tiny components folded into nearest neighbor.
- `segment_rigid.py` — CLI wrapper (opacity-filters floaters, label −1). `--selftest` verifies on a synthetic 7-body scene with no GPU: **ARI 0.9988**.
- `mbs_infer.py` — Option A adapter: exact flow into MotNet per view pair, `compose_dense` + `sync_motion_seg` imported from MBS source, one shared FPS subsample across views (the "permutation sync is the identity" simplification), 3-NN label propagation back to the full set. Wired into the orchestrator as `segment.mbs` but **not yet run on a real GPU/checkpoint**.
- `metrics.py`, `evaluate_segmentation.py` — ARI + Hungarian IoU against `gt_segmentation.npz` (GT labels nearest-neighbor-propagated from the init cloud onto trained Gaussians), plus colored-PLY and PNG previews (`visualize.py`).
- `run.sh <scene>` — chains extract → segment → evaluate; supports `SKIP_EXTRACT=1` and passthrough tuning args.

## Results so far

- Synthetic self-test: ARI 0.9988, mean Hungarian IoU 0.982.
- First real `pump01` run (2026-07-06): poor (ARI 0.05, 4–5 segments vs 107 GT). Root cause: linear-histogram Otsu failed on the heavily right-skewed real edge-score distribution — fixed by Otsu in log-space. Residual caveat: the trained model's frame-to-frame position noise is comparable to the true mm-scale motion (median edge score 0.00125 vs p90 0.0035), so segmentation quality is ultimately gated by reconstruction quality; `--threshold-mult` and `-k` are the tuning knobs.
- A separate numerical quirk: the Otsu-log threshold degenerates (ARI 0.0) on an *exactly noiseless* synthetic scene under some numpy/scipy builds — worked around via `threshold_mult` (documented in the orchestrator's vertical-slice test).

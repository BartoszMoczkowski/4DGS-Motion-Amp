"""Vendored, verbatim copy of ``motion-seg/motion_seg/mbs_infer.py``'s inference logic (T10 copy-in, per
``planning/INSTRUCTIONS.md``'s "copy the logic in, don't call the original script" rule; see
``pipeline.vendored.cuda``'s package docstring). Body is byte-for-byte the reference script's
(``_load_mot_net``/``_select_working_set``/``run_mbs_segmentation``/``main``), including its own
argparse ``main()`` entry point — this file is executed as a separate process inside the ``cuda``
container (``pipeline/stages/segment_mbs.py`` builds the CLI invocation), never imported by the
orchestrator's own host process.

Two changes from the reference script, both purely mechanical relocation fixes (not logic
changes) — mirroring how ``pipeline.vendored.host.segment_rigid`` only changed its one sibling
import when *it* moved:

- ``MBS_ROOT`` now walks up from this file's new location (``pipeline/vendored/cuda/`` instead of
  ``motion_seg/``) to the same ``submodules/multibody-sync-4dgs`` the reference script pointed at.
- The reference script's own ``_REPO_ROOT`` sys.path insert and its ``main()``-tail lazy
  ``from motion_seg.visualize import render_segmentation_png`` preview-PNG block are dropped
  entirely — both existed only to reach into ``motion_seg``, the exact throwaway-script reference
  the copy-in rule forbids depending on. Preview-PNG generation is out of scope here, same as
  ``pipeline/stages/seg_eval.py``'s ``comparison_png`` (see that module's docstring) and
  ``segment_rigid.py``'s ``preview_png`` config field — neither is wired up yet either.

Original docstring:

    EXPERIMENTAL — "Option A" from .claude_notes/NOTES_4dgs_motion_segmentation.md: run
    MultiBodySync's MotNet (submodules/multibody-sync-4dgs) on 4DGS trajectory data, skipping
    FlowNet + permutation synchronization entirely because 4DGS already gives us EXACT,
    free per-point correspondence across time (the same Gaussian index at every timestep) —
    the whole reason those two stages exist in vanilla MBS (matching *unordered, uncorresponded*
    point-cloud scans) is a problem we don't have.

    STATUS: written from a careful reading of the MBS source (models/mot_net.py, test.py,
    models/full_net.py, utils/pointnet2_util.py) but NOT executed — this sandbox has no GPU, no
    compiled CUDA ext, and no downloaded pretrained weights, all three of which are hard
    requirements. Treat this as a documented starting point, not a verified pipeline; expect to
    debug shapes/behavior the first time it actually runs. See the module docstring sections
    below for exactly what each step is doing and which MBS source lines it mirrors.

    --------------------------------------------------------------------------------------------
    Prerequisites (all must happen in your GPU/CUDA devcontainer — same one train.py runs in):

    1. Build the CUDA ops. They JIT-compile on import (submodules/multibody-sync-4dgs/ext/__init__.py
       calls torch.utils.cpp_extension.load(...) at import time) — needs nvcc + a matching CUDA
       toolkit. Just importing `utils.pointnet2_util` from that package triggers the build; expect
       a one-time compile delay the first time this script runs.

    2. Get pretrained MotNet weights. There is NO local checkpoint (submodules/multibody-sync-4dgs/
       ckpt/ is empty). hubconf.py references a Google Drive-hosted checkpoint for the FULL
       pipeline (flow+conf+mot combined):
           https://drive.google.com/uc?export=download&id=1bomD88-6N1iGsTtftfGvAm9JeOw8gKwb
       `torch.hub.load_state_dict_from_url` may fail on this (Google Drive's large-file virus-scan
       redirect often breaks direct-download URLs) — if it does, download it manually in a browser
       and pass its local path via --checkpoint. Either way the checkpoint's state dict has
       `mot_net.*`-prefixed keys mixed in with `flow_net.*`/`conf_net.*` — this script strips the
       `mot_net.` prefix and loads only those into a standalone MotNet.

    3. Weight mismatch risk (flagged in NOTES_4dgs_motion_segmentation.md's Option A trade-offs):
       MotNet was trained on MBS's own noisy FlowNet-predicted flow, not exact zero-noise 4DGS
       flow, and on roughly unit/meter-scale point clouds (PointnetSAModuleMSG ball-query radii of
       0.1/0.2/0.4, and a `t.clamp_(-2.0, 2.0)` on translations elsewhere in test.py) — the pump01
       scene was normalized to a nerf++ camera radius of 4.0 (see omni_to_4dgs.py's
       --target-radius), which is at least in the same ballpark, but there's no guarantee MotNet
       generalizes to exact flow without fine-tuning. If results look bad, that's the likely reason
       — this is explicitly why the design notes call Option A "Medium effort, likely needs
       retraining/fine-tuning", not "drop-in".

    --------------------------------------------------------------------------------------------
    What this script does (mirrors test.py's TestTimeFullNet.forward, with FlowNet + permutation
    sync removed since we don't need them):

    1. Load trajectories.npz (from extract_trajectories.py) and opacity-filter + subsample down to
       a manageable working set N' (MBS's own machinery is built for N~256-1024, not 10^5-10^6 —
       see the module docstring in mbs_infer for why: the *real* MBS permutation-sync path we're
       skipping does an all-pairs (N,N) cdist + an (n_view*N, n_view*N) eigendecomposition, which
       is why vanilla MBS never runs at 4DGS point counts; we avoid that path entirely, but MotNet
       itself was still only ever trained/tuned at N~512).
    2. Pick K evenly-spaced timesteps ("views") — 4 by default, matching MBS's typical 2-4 view
       setup (NOTES_4dgs_motion_segmentation.md open-Q #5).
    3. For every view pair (i, j), build the EXACT analytic flow directly from the trajectory data
       (flow[:,0] = pos_j - pos_i, flow[:,1] = pos_i - pos_j) instead of calling FlowNet.
    4. Because every "view" is the *same* physical points at a different time, we use ONE shared
       256-point subsample index set for all views (instead of MBS's per-view-independent FPS +
       permutation sync needed only because different *scans* have different, uncorresponded
       points) — this is the concrete "skip permutation sync, it's the identity" from Option A.
    5. Run MotNet per view pair -> pairwise (256,256) same-body affinity (test.py:330-343:
       sigmoid, then rescale each pair to the first pair's mean scale, clamp to [0,1]).
    6. Assemble into one dense block matrix (`compose_dense`, from models/full_net.py) and run the
       spectral motion-segmentation step (`sync_motion_seg`, utils/sync_util.py) to get a soft
       per-(view,point) segment embedding, average across the K (redundant, same-point) views,
       and take the hard argmax as the label for the 256-point subsample.
    7. Propagate labels from the 256-point subsample to the full working set N' via 3-NN inverse-
       distance interpolation (`feature_propagation`, models/full_net.py) in CANONICAL (rest-pose)
       space — since our target is one static label per Gaussian (design decision, see
       NOTES_4dgs_motion_segmentation.md open-Q #2), not a per-view label like vanilla MBS.

    Usage (in the GPU devcontainer, from the repo root):
        uv run python -m motion_seg.mbs_infer \\
            --trajectories output/multipleview/pump01/trajectories.npz \\
            --out output/multipleview/pump01/segmentation_mbs.npz \\
            --checkpoint /path/to/downloaded/checkpoint.pth.tar \\
            --n-points 4000 --n-views 4 --alpha 0.05
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

#: pipeline/vendored/cuda/mbs_infer.py -> repo root -> submodules/multibody-sync-4dgs (four
#: hops up from this file's directory: cuda -> vendored -> pipeline -> orchestrator -> repo root).
#: The reference script computed this the same way relative to its own (different) location
#: (motion-seg/motion_seg/mbs_infer.py, one hop up to the repo root) — this is the one relocation fix that
#: actually changes behavior, everything else below is unchanged.
MBS_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "submodules", "multibody-sync-4dgs"
)


def _add_mbs_to_path():
    path = os.path.abspath(MBS_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_mot_net(checkpoint_path: str):
    """Build a standalone MotNet and load its weights out of a full-pipeline checkpoint
    (state dict keys prefixed `mot_net.`, alongside `flow_net.`/`conf_net.` we don't need)."""
    import torch

    from models.mot_net import MotNet  # noqa: E402 (needs _add_mbs_to_path() first)

    mot_net = MotNet().cuda().eval()
    state = torch.load(checkpoint_path, map_location="cuda")
    state = state.get("model_state", state)
    mot_prefix = "mot_net."
    mot_state = {
        k[len(mot_prefix):]: v for k, v in state.items() if k.startswith(mot_prefix)
    }
    if not mot_state:
        # Maybe it's already a mot-only checkpoint (no prefix).
        mot_state = state
    missing, unexpected = mot_net.load_state_dict(mot_state, strict=False)
    if missing or unexpected:
        print(f"[warn] mot_net.load_state_dict: missing={missing} unexpected={unexpected}",
              file=sys.stderr)
    return mot_net


def _select_working_set(canonical_xyz, opacity, n_points, opacity_thresh, seed=0):
    """Opacity-filter, then randomly subsample down to `n_points` (MBS's machinery targets
    N~256-1024, not the full 10^5-10^6 Gaussians — see module docstring)."""
    rng = np.random.RandomState(seed)
    keep = opacity > opacity_thresh if opacity is not None else np.ones(len(canonical_xyz), bool)
    idx_kept = np.where(keep)[0]
    if len(idx_kept) > n_points:
        idx_kept = rng.choice(idx_kept, n_points, replace=False)
    return idx_kept


def run_mbs_segmentation(
    canonical_xyz: np.ndarray,
    traj: np.ndarray,
    times: np.ndarray,
    opacity: np.ndarray | None,
    checkpoint_path: str,
    n_points: int = 4000,
    n_views: int = 4,
    n_sub: int = 256,
    opacity_thresh: float = 0.1,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Returns (labels (N_total,) int, working_idx (subset actually processed)) — labels for
    points outside `working_idx` are -1 (not processed)."""
    _add_mbs_to_path()
    import torch

    from models.full_net import compose_dense, feature_propagation
    from utils import pointnet2_util
    from utils.sync_util import sync_motion_seg

    working_idx = _select_working_set(canonical_xyz, opacity, n_points, opacity_thresh, seed)
    n_total = len(canonical_xyz)
    print(f"[mbs] working set: {len(working_idx)} / {n_total} points "
          f"(opacity > {opacity_thresh}, then random subsample to {n_points})")

    # K evenly-spaced timesteps ("views"). `times` are the sample points extract_trajectories
    # used (np.linspace(0,1,n_times,endpoint=False)); pick the closest available indices.
    target_t = np.linspace(0.0, 1.0, n_views, endpoint=False)
    view_t_idx = [int(np.argmin(np.abs(times - t))) for t in target_t]
    print(f"[mbs] using {n_views} views at t={times[view_t_idx]}")

    xyz_np = traj[working_idx][:, view_t_idx, :]  # (N', K, 3)
    xyz = torch.from_numpy(xyz_np).float().cuda().permute(1, 0, 2).unsqueeze(0)  # (1, K, N', 3)
    n_batch, n_view, n_point, _ = xyz.shape

    mot_net = _load_mot_net(checkpoint_path)

    # ONE shared FPS subsample, reused for every view: every "view" here is the SAME physical
    # points at a different time, so there's no need (unlike vanilla MBS) for per-view
    # independent subsampling + a permutation to line them back up.
    xyz0 = xyz[:, 0].contiguous()
    sub_inds_shared = pointnet2_util.furthest_point_sample(xyz0, n_sub).long()  # (1, n_sub)
    sub_inds = sub_inds_shared.unsqueeze(1).repeat(1, n_view, 1)  # (1, K, n_sub)

    motion_dict = {}
    canonical_scale = None
    with torch.no_grad():
        for i in range(n_view):
            for j in range(i + 1, n_view):
                pc_i, pc_j = xyz[:, i], xyz[:, j]
                # Exact analytic flow -- this is the whole point of Option A: no FlowNet.
                flow_ij = torch.stack([pc_j - pc_i, pc_i - pc_j], dim=1)  # (1, 2, N', 3)
                _, motion_ij, _, _ = mot_net(
                    xyz[:, [i, j]], flow_ij, sub_inds[:, [i, j]]
                )
                motion_ij = motion_ij.sigmoid()  # test.py:333 -- only activation needed
                scale = motion_ij.reshape(motion_ij.size(0), -1).mean(dim=-1, keepdim=True).unsqueeze(-1)
                if canonical_scale is None:
                    canonical_scale = scale
                else:
                    motion_ij = motion_ij / scale * canonical_scale
                motion_dict[(i, j)] = torch.clamp(motion_ij, 0.0, 1.0)

        motion_dense = compose_dense(motion_dict, n_view, torch.zeros_like(motion_dict[(0, 1)]))
        motion_absolute = sync_motion_seg(motion_dense, t=0.0, cut_thres=alpha)  # (1, K*n_sub, S)
        motion_absolute = motion_absolute / motion_absolute.sum(-1, keepdim=True).clamp_min(1e-8)
        n_segments = motion_absolute.shape[-1]
        print(f"[mbs] spectral cut found {n_segments} segments (alpha={alpha})")

        # Every view's row for a given sub-sampled point refers to the SAME physical point
        # (unlike vanilla MBS's per-view-distinct points) -> average across views instead of
        # keeping a separate per-view label, since we want one static label per Gaussian.
        motion_absolute = motion_absolute.reshape(n_batch, n_view, n_sub, n_segments)
        sub_soft = motion_absolute.mean(dim=1)  # (1, n_sub, S)
        sub_labels = sub_soft.argmax(dim=-1)  # (1, n_sub)

        # Propagate from the 256-point subsample to the full working set, in CANONICAL space
        # (one static label per Gaussian, not per-view).
        sub_xyz_canon = torch.from_numpy(canonical_xyz[working_idx]).float().cuda().unsqueeze(0)
        sub_xyz_sub = pointnet2_util.gather_nd(sub_xyz_canon, sub_inds_shared)  # (1, n_sub, 3)
        sub_soft_t = sub_soft.transpose(1, 2).contiguous()  # (1, S, n_sub) -- "transposed" layout
        propagated = feature_propagation(sub_xyz_canon, sub_xyz_sub, sub_soft_t, True)  # (1, S, N')
        working_labels = propagated[0].argmax(dim=0).cpu().numpy()  # (N',)

    labels = np.full(n_total, -1, dtype=np.int64)
    labels[working_idx] = working_labels
    return labels, working_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", required=True, help=".npz from extract_trajectories.py")
    ap.add_argument("--out", required=True, help="output segmentation .npz path")
    ap.add_argument("--checkpoint", required=True,
                     help="path to a downloaded MBS checkpoint (full-pipeline or mot-only .pth.tar)")
    ap.add_argument("--n-points", type=int, default=4000,
                     help="working-set size (MBS's machinery targets ~256-1024; more costs "
                          "more GPU memory/time with diminishing accuracy benefit)")
    ap.add_argument("--n-views", type=int, default=4, help="number of timesteps to use (2-4 typical)")
    ap.add_argument("--n-sub", type=int, default=256, help="FPS subsample size MotNet's "
                     "spectral step operates on (matches MBS's own nsample_motion=256)")
    ap.add_argument("--opacity-thresh", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.05,
                     help="spectral eigenvalue cutoff deciding the number of segments "
                          "(matches hubconf.py's model_articulated default)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = args.out
    # Guard against a directory (or trailing-slash) --out: `out.rsplit(".", 1)[0]` below would
    # otherwise silently produce a bogus/relative preview path (e.g. "./foo/".rsplit(".",1)[0]
    # == "" -> "_preview.png" written into whatever the cwd happens to be).
    if out_path.endswith(("/", "\\")) or os.path.isdir(out_path):
        out_path = os.path.join(out_path, "segmentation_mbs.npz")
        print(f"[warn] --out was a directory; writing to {out_path}", file=sys.stderr)

    data = np.load(args.trajectories)
    canonical_xyz, traj, times = data["canonical_xyz"], data["traj"], data["times"]
    opacity = data["opacity"] if "opacity" in data else None

    labels, working_idx = run_mbs_segmentation(
        canonical_xyz, traj, times, opacity, args.checkpoint,
        n_points=args.n_points, n_views=args.n_views, n_sub=args.n_sub,
        opacity_thresh=args.opacity_thresh, alpha=args.alpha, seed=args.seed,
    )
    np.savez(out_path, points=canonical_xyz.astype(np.float32), labels=labels)
    n_segments = len(np.unique(labels[labels != -1]))
    print(f"[ok] {n_segments} segments over {len(working_idx)} points "
          f"({len(canonical_xyz) - len(working_idx)} left unlabeled, -1) -> {out_path}")


if __name__ == "__main__":
    main()

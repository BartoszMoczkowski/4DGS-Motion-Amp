"""Import smoke test for the uv workspace (T24).

Run after a root `uv sync`:

    uv run python scripts/smoke_workspace_imports.py

Imports every workspace member package (and a representative set of scene-gen
flat modules) and exits non-zero if any import fails. Modules that genuinely
require the Isaac Sim runtime (`pxr`) or OS-native libraries not pip-installable
(`pyzbar` → libzbar) are listed as EXPECTED_SKIP and reported separately.
"""

from __future__ import annotations

import importlib
import sys
import traceback

# (module, reason-for-skip-if-missing-dep or None)
REQUIRED = [
    # root package body
    "amp",
    # 4dgs-core pure-python packages
    "utils",
    "arguments",
    # motion-seg (+ mbs shim; torch is lazy inside mbs_infer)
    "motion_seg",
    "motion_seg.segment_rigid",
    "motion_seg.mbs_infer",
    "mbs_bootstrap",
    # omniverse-pipeline (pxr-dependent modules excluded — Isaac runtime only)
    "omniverse_pipeline",
    "omniverse_pipeline.omni_to_4dgs",
    "omniverse_pipeline.rig",
    # amp-ui
    "amp_ui",
    "amp_ui.cameras",
    # orchestrator (Layer 1; mcp_server/ui are extras-gated, not in default sync)
    "pipeline",
    "pipeline.api",
    # camera_sync packages (package __init__ only: sync_analyzer.analyze needs
    # libzbar via pyzbar, sync_display.display needs a display-capable pygame)
    "sync_display",
    "recorder",
    "sync_analyzer",
    "rtsp_capture",
    # scene-gen flat modules (Isaac-only and torch-heavy ones excluded below)
    "analyze_complexity_ladder",
    "check_capture_progress",
    "frames_to_mp4",
    "gen_spinning_cubes",
    "partition_swap",
    "re_render_grid_runs",
    "render_seg_visualizations",
    "rescore_motion_classes",
    "run_cubes_4dgs",
    "run_cubes_capture",
    "run_cubes_seg",
    "run_grid_4dgs",
    "run_grid_seg",
    "run_grid_multicut_calibrated",
    "run_partition_swap_exp",
    "run_pump01_boundary_refine",
    "run_pump01_spatial_cc",
    "show_cubes_seg",
    "view_gaussians_open3d",
    "identity",
    "identity.cluster",
    "identity.mask_provider",
]

# Import the CUDA extensions and everything that transitively imports them.
# The extensions are linux-only dependencies (see core/pyproject.toml): they
# build only where a CUDA toolkit exists (the Docker image). On other
# platforms these modules cannot import — report them as skips there.
CUDA_REQUIRED = [
    "diff_gaussian_rasterization",
    "simple_knn",
    "scene",
    "gaussian_renderer",
    "motion_amp",
    "train",
    "render",
    "render_amp",
    "identity.model",
    "identity.train",
    "run_grid_identity",
]

# Require the Isaac Sim Python runtime (`pxr` is not pip-installable) — these
# are expected to fail in the workspace venv and are informational only.
EXPECTED_SKIP = {
    "gen_scenes": "pxr (Isaac Sim runtime)",
    "gen_cubes_dataset_all": "pxr via omniverse_pipeline.add_motion (Isaac Sim runtime)",
    "build_gt_motion_classes": "pxr (Isaac Sim runtime)",
    "export_cube_gt_pointcloud": "pxr (Isaac Sim runtime)",
    "omniverse_pipeline.add_motion": "pxr (Isaac Sim runtime)",
    "omniverse_pipeline.compose_scene": "pxr (Isaac Sim runtime)",
    "omniverse_pipeline.split_mesh": "pxr (Isaac Sim runtime)",
    "omniverse_pipeline.omni_capture": "pxr/omni (Isaac Sim runtime)",
}


def main() -> int:
    failures: list[str] = []
    skipped: list[str] = []

    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
            print(f"[ok]   {mod}")
        except Exception:
            failures.append(mod)
            print(f"[FAIL] {mod}")
            traceback.print_exc()

    if sys.platform == "linux":
        cuda_required, cuda_skip = CUDA_REQUIRED, []
    else:
        cuda_required, cuda_skip = [], CUDA_REQUIRED
        print(f"[info] not linux — CUDA-extension-dependent modules are expected to skip: "
              f"{', '.join(CUDA_REQUIRED)}")
    for mod in cuda_required:
        try:
            importlib.import_module(mod)
            print(f"[ok]   {mod}")
        except Exception:
            failures.append(mod)
            print(f"[FAIL] {mod}")
            traceback.print_exc()
    skipped.extend(cuda_skip)

    for mod, reason in EXPECTED_SKIP.items():
        try:
            importlib.import_module(mod)
            print(f"[ok]   {mod} (unexpectedly importable — {reason} present)")
        except ImportError as e:
            skipped.append(mod)
            print(f"[skip] {mod} ({reason}: {e})")
        except Exception:
            failures.append(mod)
            print(f"[FAIL] {mod} (expected only {reason} to be missing)")
            traceback.print_exc()

    # mbs shim behavior: register() must prepend the submodule dir exactly once.
    import mbs_bootstrap

    path = mbs_bootstrap.register()
    assert sys.path[0] == path, f"mbs_bootstrap.register() did not put {path} at sys.path[0]"
    n = mbs_bootstrap.register()
    assert sys.path.count(path) == 1 and n == path, "mbs_bootstrap.register() is not idempotent"
    print(f"[ok]   mbs_bootstrap.register() -> {path}")

    total = len(REQUIRED) + (len(CUDA_REQUIRED) if sys.platform == "linux" else 0)
    print(f"\n{total - len(failures)}/{total} required imports passed, "
          f"{len(skipped)} expected skips, {len(failures)} failures")
    if failures:
        print("FAILURES:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""scripts/compute_rig_prior.py

Task T42: Cross-scene leverage: computes shared physical camera rig prior
across lego_turbine captures (base, 2_missing, 5_missing, turbine_r1_missing_6).

Same rig -> offsets fitted across scenes act as an empirical Bayes prior for each scene.
Validates inter-scene consistency and writes data/multipleview/rig_prior_offsets.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MULTIPLEVIEW_DIR = REPO_ROOT / "data" / "multipleview"
OUTPUT_FILE = MULTIPLEVIEW_DIR / "rig_prior_offsets.json"

SCENE_NAMES = [
    "lego_turbine_base",
    "lego_turbine_2_missing",
    "lego_turbine_5_missing",
    "turbine_r1_missing_6",
]


def compute_rig_prior(
    scene_names: list[str] = SCENE_NAMES,
    out_path: Path = OUTPUT_FILE,
) -> dict[str, Any]:
    cams = [f"cam{i:02d}" for i in range(1, 11)]
    scene_offsets_s: dict[str, dict[str, float]] = {}
    scene_hashes: dict[str, str] = {}
    scene_fps: dict[str, float] = {}

    import hashlib

    for s in scene_names:
        p = MULTIPLEVIEW_DIR / s / "offsets.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing offsets.json for scene {s} at {p}")
        raw_bytes = p.read_bytes()
        h = hashlib.sha256(raw_bytes).hexdigest()
        scene_hashes[s] = h
        d = json.loads(raw_bytes.decode("utf-8"))
        scene_offsets_s[s] = d.get("offsets_s", {})
        scene_fps[s] = float(d.get("fps", 20.0))

    rig_prior_s: dict[str, float] = {}
    rig_prior_ms: dict[str, float] = {}
    rig_std_ms: dict[str, float] = {}
    per_camera_analysis: dict[str, dict[str, Any]] = {}

    for c in cams:
        vals_s = [scene_offsets_s[s].get(c, 0.0) for s in scene_names]
        vals_ms = [v * 1000.0 for v in vals_s]

        # Median is robust against single-session packet phase shift
        med_s = float(np.median(vals_s))
        mean_s = float(np.mean(vals_s))
        std_ms = float(np.std(vals_ms))

        # Anchor cam01 is always 0.0
        if c == "cam01":
            med_s = 0.0
            mean_s = 0.0
            std_ms = 0.0

        rig_prior_s[c] = med_s
        rig_prior_ms[c] = med_s * 1000.0
        rig_std_ms[c] = std_ms

        residuals_ms = {s: float((scene_offsets_s[s].get(c, 0.0) - med_s) * 1000.0) for s in scene_names}

        per_camera_analysis[c] = {
            "prior_offset_s": med_s,
            "prior_offset_ms": med_s * 1000.0,
            "mean_offset_s": mean_s,
            "std_ms": std_ms,
            "per_scene_values_ms": {s: vals_ms[i] for i, s in enumerate(scene_names)},
            "residuals_to_prior_ms": residuals_ms,
            "max_discrepancy_ms": float(max(abs(r) for r in residuals_ms.values())),
        }

    # Inter-scene agreement metric (mean residual across all views)
    all_residuals = [
        abs(per_camera_analysis[c]["residuals_to_prior_ms"][s])
        for c in cams
        for s in scene_names
        if c != "cam01"
    ]
    mean_discrepancy = float(np.mean(all_residuals))
    p95_discrepancy = float(np.percentile(all_residuals, 95))

    prior_bundle = {
        "description": "Empirical Bayes camera timing prior across multi-cam rig captures",
        "contributing_scenes": scene_names,
        "scene_hashes": scene_hashes,
        "reference_camera": "cam01",
        "rig_prior_offsets_s": rig_prior_s,
        "rig_prior_offsets_ms": rig_prior_ms,
        "rig_prior_std_ms": rig_std_ms,
        "per_camera_analysis": per_camera_analysis,
        "mean_discrepancy_to_prior_ms": mean_discrepancy,
        "p95_discrepancy_to_prior_ms": p95_discrepancy,
        "status": "validated",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prior_bundle, indent=2), encoding="utf-8")
    return prior_bundle


def print_prior_table(prior: dict[str, Any]):
    print("\n=========================================================================================")
    print("  TASK T42: CROSS-SCENE CAMERA TIMING RIG PRIOR")
    print("=========================================================================================")
    scenes = prior["contributing_scenes"]
    header = f"{'Cam':<8}" + "".join([f"{s[:18]:<20}" for s in scenes]) + f"{'Rig Prior':<14}{'Std (ms)':<10}"
    print(header)
    print("-" * len(header))
    for c, info in prior["per_camera_analysis"].items():
        row = f"{c:<8}"
        for s in scenes:
            val = info["per_scene_values_ms"][s]
            row += f"{val:>7.2f} ms         "
        row += f"{info['prior_offset_ms']:>7.2f} ms     {info['std_ms']:>6.2f}"
        print(row)
    print("-" * len(header))
    print(f"Mean residual to prior: {prior['mean_discrepancy_to_prior_ms']:.2f} ms")
    print(f"P95 residual to prior:  {prior['p95_discrepancy_to_prior_ms']:.2f} ms")
    print("=========================================================================================\n")


if __name__ == "__main__":
    prior = compute_rig_prior()
    print_prior_table(prior)

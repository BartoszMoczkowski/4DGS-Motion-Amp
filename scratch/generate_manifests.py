"""Helper to generate grid manifests and motion json files for capture_limitation and lego_turbine."""
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 1. Capture Limitation
CAP_SCENES_DIR = REPO_ROOT / "omniverse-pipeline" / "data" / "scenes" / "capture_limitation"

CELLS_INFO = [
    # Series A
    {"cell": "sA0", "name": "pump01_sA0_common_target_static", "series": "A", "desc": "baseline pastel, common-mode non-target, target static", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 0.0},
    {"cell": "sA1", "name": "pump01_sA1_deviant_2x", "series": "A", "desc": "deviant target 2x base motion (|B-A|=1A shear)", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 2.0},
    {"cell": "sA2", "name": "pump01_sA2_deviant_5x", "series": "A", "desc": "deviant target 5x base motion (|B-A|=4A shear)", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 5.0},
    {"cell": "sA3", "name": "pump01_sA3_deviant_10x", "series": "A", "desc": "deviant target 10x base motion (|B-A|=9A shear)", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 10.0},
    {"cell": "sA4", "name": "pump01_sA4_deviant_20x", "series": "A", "desc": "deviant target 20x base motion (|B-A|=19A shear)", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 20.0},
    {"cell": "sA5", "name": "pump01_sA5_target_only", "series": "A", "desc": "target only motion, non-target static", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 1.0},

    # Series R
    {"cell": "sR0", "name": "pump01_sR0_baseline_1600x900", "series": "R", "desc": "baseline reference (1600x900, R=420cm)", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 1.0},
    {"cell": "sR1", "name": "pump01_sR1_res_2560x1440", "series": "R", "desc": "res 2560x1440, R=420cm", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 1.0},
    {"cell": "sR2", "name": "pump01_sR2_res_3000x2000", "series": "R", "desc": "res 3000x2000 (6 MP), R=420cm", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 1.0},
    {"cell": "sR3", "name": "pump01_sR3_dist_half_1600x900", "series": "R", "desc": "dist half (1600x900, R=210cm)", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 1.0},
    {"cell": "sR4", "name": "pump01_sR4_res3k_dist_half", "series": "R", "desc": "res 3k + dist half (3000x2000, R=210cm)", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 1.0},
    {"cell": "sR5", "name": "pump01_sR5_res3k_dist_half_120f", "series": "R", "desc": "res 3k + dist half + 120 frames", "frames": 120, "n_times": 120, "base_amp_mm": 5.0, "multiplier": 1.0},

    # Series T
    {"cell": "sT0", "name": "pump01_sT0_mat_pastel", "series": "T", "desc": "pastel displayColors baseline", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 1.0},
    {"cell": "sT1", "name": "pump01_sT1_mat_metallic", "series": "T", "desc": "uniform brushed steel metallic", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 1.0},
    {"cell": "sT2", "name": "pump01_sT2_mat_metallic_texture", "series": "T", "desc": "metallic + procedural surface texture", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 1.0},

    # Series F
    {"cell": "sF0", "name": "pump01_sF0_combined_baseline", "series": "F", "desc": "compound baseline control (A0 x T0 x R0)", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 0.0},
    {"cell": "sF1", "name": "pump01_sF1_combined_honest", "series": "F", "desc": "compound honest (A3 10x motion x T1 metallic x R4 capture)", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 10.0},
    {"cell": "sF2", "name": "pump01_sF2_combined_honest_textured", "series": "F", "desc": "compound honest + texture (A3 10x x T2 texture x R4 capture)", "frames": 60, "n_times": 60, "base_amp_mm": 5.0, "multiplier": 10.0},
]

manifest_cells = []
for c in CELLS_INFO:
    cid = c["cell"]
    cname = c["name"]
    entry = {
        "cell": cid,
        "name": cname,
        "series": c["series"],
        "description": c["desc"],
        "capture_dir": f"Q:/Omniverse/renders/capture_{cname}",
        "dataset_name": f"pump01_{cid}",
        "num_frames": c["frames"],
        "fps": 60.0,
        "base_amp_mm": c["base_amp_mm"],
        "multiplier": c["multiplier"],
        "cams": 10,
    }
    manifest_cells.append(entry)

    # Copy {cname}_motion.json to {cid}_motion.json if present
    src_motion = CAP_SCENES_DIR / f"{cname}_motion.json"
    dst_motion = CAP_SCENES_DIR / f"{cid}_motion.json"
    if src_motion.is_file():
        shutil.copy2(src_motion, dst_motion)

cap_manifest = {
    "name": "capture_limitation",
    "seed": 0,
    "amplified_part": "part_008",
    "freq_hz": 10.0,
    "cells": manifest_cells,
}

cap_manifest_path = CAP_SCENES_DIR / "grid_manifest.json"
cap_manifest_path.write_text(json.dumps(cap_manifest, indent=2), encoding="utf-8")
print(f"Wrote {cap_manifest_path} ({len(manifest_cells)} cells)")

# 2. Lego Turbine Manifest
TURBINE_CELLS = [
    {
        "cell": "lego_turbine_base",
        "name": "lego_turbine_base",
        "capture_dir": "data/multipleview/lego_turbine_base",
        "dataset_name": "lego_turbine_base",
        "description": "Baseline Lego turbine with all blades intact",
        "cams": 10,
        "frames": 164,
    },
    {
        "cell": "lego_turbine_2_missing",
        "name": "lego_turbine_2_missing",
        "capture_dir": "data/multipleview/lego_turbine_2_missing",
        "dataset_name": "lego_turbine_2_missing",
        "description": "Lego turbine with 2 blades missing (mass imbalance)",
        "cams": 10,
        "frames": 164,
    },
    {
        "cell": "lego_turbine_5_missing",
        "name": "lego_turbine_5_missing",
        "capture_dir": "data/multipleview/lego_turbine_5_missing",
        "dataset_name": "lego_turbine_5_missing",
        "description": "Lego turbine with 5 blades missing (severe asymmetric damage)",
        "cams": 10,
        "frames": 164,
    },
    # Also include short-named aliases
    {
        "cell": "base",
        "name": "lego_turbine_base",
        "capture_dir": "data/multipleview/lego_turbine_base",
        "dataset_name": "lego_turbine_base",
        "description": "Baseline Lego turbine with all blades intact",
        "cams": 10,
        "frames": 164,
    },
    {
        "cell": "2_missing",
        "name": "lego_turbine_2_missing",
        "capture_dir": "data/multipleview/lego_turbine_2_missing",
        "dataset_name": "lego_turbine_2_missing",
        "description": "Lego turbine with 2 blades missing (mass imbalance)",
        "cams": 10,
        "frames": 164,
    },
    {
        "cell": "5_missing",
        "name": "lego_turbine_5_missing",
        "capture_dir": "data/multipleview/lego_turbine_5_missing",
        "dataset_name": "lego_turbine_5_missing",
        "description": "Lego turbine with 5 blades missing (severe asymmetric damage)",
        "cams": 10,
        "frames": 164,
    },
]

turbine_manifest = {
    "name": "lego_turbine",
    "cells": TURBINE_CELLS,
}

turbine_manifest_path = REPO_ROOT / "data" / "multipleview" / "lego_turbine_manifest.json"
turbine_manifest_path.write_text(json.dumps(turbine_manifest, indent=2), encoding="utf-8")
print(f"Wrote {turbine_manifest_path} ({len(TURBINE_CELLS)} cells)")

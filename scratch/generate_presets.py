import os
import yaml

presets_dir = "scene-gen/presets"
os.makedirs(presets_dir, exist_ok=True)

SEG_USD = "Q:/Omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_segmented.usd"
ENV_USD = "Q:/Omniverse/assets/physAI_start/Factory/Factory.usd"
TARGET_PART = "part_008"  # Prominent bottom manifold cylinder (3224 verts)

cells = [
    # SERIES A (deviant-part detectability; motion-only; capture R0; material pastel)
    {
        "name": "pump01_sA0_common_target_static",
        "series": "A",
        "desc": "A0: all 106 non-target parts vibrate in sync at amplitude A, target part static (B=0)",
        "motion": {"mode": "common_mode", "base_amp_mm": 5.0, "common_amp_mm": 5.0, "target_amp_mm": 0.0, "multiplier": 0.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sA1_deviant_2x",
        "series": "A",
        "desc": "A1: common-mode A, target part at B = 2A (relative interface motion = A)",
        "motion": {"mode": "common_mode", "base_amp_mm": 5.0, "common_amp_mm": 5.0, "target_amp_mm": 10.0, "multiplier": 2.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sA2_deviant_5x",
        "series": "A",
        "desc": "A2: common-mode A, target part at B = 5A (relative interface motion = 4A)",
        "motion": {"mode": "common_mode", "base_amp_mm": 5.0, "common_amp_mm": 5.0, "target_amp_mm": 25.0, "multiplier": 5.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sA3_deviant_10x",
        "series": "A",
        "desc": "A3: common-mode A, target part at B = 10A (relative interface motion = 9A)",
        "motion": {"mode": "common_mode", "base_amp_mm": 5.0, "common_amp_mm": 5.0, "target_amp_mm": 50.0, "multiplier": 10.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sA4_deviant_20x",
        "series": "A",
        "desc": "A4: common-mode A, target part at B = 20A (relative interface motion = 19A)",
        "motion": {"mode": "common_mode", "base_amp_mm": 5.0, "common_amp_mm": 5.0, "target_amp_mm": 100.0, "multiplier": 20.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sA5_target_only",
        "series": "A",
        "desc": "A5: inverse — non-target parts static, target part at amplitude A (relative motion = A)",
        "motion": {"mode": "common_mode", "base_amp_mm": 5.0, "common_amp_mm": 0.0, "target_amp_mm": 5.0, "multiplier": 1.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    # SERIES R (capture ladder; motion standard pump01; material pastel)
    {
        "name": "pump01_sR0_baseline_1600x900",
        "series": "R",
        "desc": "R0: baseline reference cell (1600x900, dome, current distance radius_scale 2.5)",
        "motion": {"mode": "random", "base_amp_mm": 5.0, "multiplier": 1.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sR1_res_2560x1440",
        "series": "R",
        "desc": "R1: resolution sweep 2560x1440 (1.6x pixel-motion vs R0), same distance 2.5",
        "motion": {"mode": "random", "base_amp_mm": 5.0, "multiplier": 1.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 2560, "height": 1440, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sR2_res_3000x2000",
        "series": "R",
        "desc": "R2: 6 MP resolution 3000x2000 (matches real cameras, 1.88x pixel-motion), same distance 2.5",
        "motion": {"mode": "random", "base_amp_mm": 5.0, "multiplier": 1.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 3000, "height": 2000, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sR3_dist_half_1600x900",
        "series": "R",
        "desc": "R3: CONTROL: cameras at ~1/2 distance (radius_scale 1.25), 1600x900 (2.0x pixel-motion via geometry alone)",
        "motion": {"mode": "random", "base_amp_mm": 5.0, "multiplier": 1.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 1.25, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sR4_res3k_dist_half",
        "series": "R",
        "desc": "R4: 3000x2000 AND ~1/2 distance radius_scale 1.25 (~4 px motion, realistic best case)",
        "motion": {"mode": "random", "base_amp_mm": 5.0, "multiplier": 1.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 3000, "height": 2000, "radius_scale": 1.25, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sR5_res3k_dist_half_120f",
        "series": "R",
        "desc": "R5 (optional): R4 capture + 120 frames (FFT frequency resolution axis; 2x temporal bins)",
        "motion": {"mode": "random", "base_amp_mm": 5.0, "multiplier": 1.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 3000, "height": 2000, "radius_scale": 1.25, "num_frames": 120, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    # SERIES T (material ablation; motion standard pump01; capture R0)
    {
        "name": "pump01_sT0_mat_pastel",
        "series": "T",
        "desc": "T0: current pastel per-part displayColors reference cell",
        "motion": {"mode": "random", "base_amp_mm": 5.0, "multiplier": 1.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sT1_mat_metallic",
        "series": "T",
        "desc": "T1: realistic metallic (brushed steel uniform alloy PBR), no per-part color coding",
        "motion": {"mode": "random", "base_amp_mm": 5.0, "multiplier": 1.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "metallic"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sT2_mat_metallic_texture",
        "series": "T",
        "desc": "T2: T1 metallic + procedural surface texture (lathe marks, rust, decals) — Diamond sub-pixel texture hypothesis",
        "motion": {"mode": "random", "base_amp_mm": 5.0, "multiplier": 1.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "metallic_texture"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    # FINAL COMBINED CELLS
    {
        "name": "pump01_sF0_combined_baseline",
        "series": "F",
        "desc": "F0: baseline control cell (A0 motion, T0 pastel material, R0 capture)",
        "motion": {"mode": "common_mode", "base_amp_mm": 5.0, "common_amp_mm": 5.0, "target_amp_mm": 0.0, "multiplier": 0.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "pastel"},
        "capture": {"width": 1600, "height": 900, "radius_scale": 2.5, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sF1_combined_honest",
        "series": "F",
        "desc": "F1: honest real pump cell (A3 motion B=10A x T1 metallic material x R4 capture 3k half-dist)",
        "motion": {"mode": "common_mode", "base_amp_mm": 5.0, "common_amp_mm": 5.0, "target_amp_mm": 50.0, "multiplier": 10.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "metallic"},
        "capture": {"width": 3000, "height": 2000, "radius_scale": 1.25, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    },
    {
        "name": "pump01_sF2_combined_honest_textured",
        "series": "F",
        "desc": "F2: honest real pump textured cell (A3 motion B=10A x T2 textured metallic x R4 capture 3k half-dist)",
        "motion": {"mode": "common_mode", "base_amp_mm": 5.0, "common_amp_mm": 5.0, "target_amp_mm": 50.0, "multiplier": 10.0, "target_part": TARGET_PART, "freq_hz": 10.0, "seed": 0},
        "material": {"preset": "metallic_texture"},
        "capture": {"width": 3000, "height": 2000, "radius_scale": 1.25, "num_frames": 60, "fps": 60.0, "rt_subframes": 16, "max_elev_deg": 32.0}
    }
]

for c in cells:
    p_data = {
        "name": c["name"],
        "series": c["series"],
        "description": c["desc"],
        "segmented_usd": SEG_USD,
        "env": ENV_USD,
        "motion": c["motion"],
        "material": c["material"],
        "capture": c["capture"],
    }
    path = os.path.join(presets_dir, f"{c['name']}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(p_data, f, sort_keys=False)
    print(f"Wrote {path}")
print(f"Total {len(cells)} presets written to {presets_dir}")

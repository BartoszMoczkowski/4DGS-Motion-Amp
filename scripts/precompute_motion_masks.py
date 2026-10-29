"""
scripts/precompute_motion_masks.py

Task T36: Motion-mask weighted loss (R2) precomputation.

Precomputes per-pixel temporal motion importance weight maps for each camera view:
1. Computes band-passed (6.0 - 9.0 Hz) frame differences across time to isolate
   the 7.11 Hz wobble signal while rejecting slow auto-exposure drift, illumination
   flicker, and high-frequency sensor noise.
2. Applies rig-boundary gating, noise floor thresholding, morphological dilation,
   and Gaussian boundary smoothing so the moving turbine structure and its wobble
   displacement envelope (±3 px) are solidly covered.
3. Provides a soft ROI rectangle fallback mode with Gaussian falloff.
4. Outputs:
   - <scene>/motion_masks/camXX.npy: float32 array in [0.0, 1.0] (motion intensity)
   - <scene>/motion_masks/camXX_vis.png: 8-bit grayscale inspection image [0, 255]
   - <scene>/motion_masks/camXX_heatmap.png: color-mapped inspection artifact
   - <scene>/motion_masks/manifest.json: execution metadata and camera stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
MULTIPLEVIEW_DIR = REPO_ROOT / "data" / "multipleview"
TURBINE_SCENES = [
    "lego_turbine_2_missing",
    "lego_turbine_5_missing",
    "lego_turbine_base",
    "turbine_r1_missing_6",
]


def get_camera_rig_roi(height: int, width: int, cam_idx: int) -> np.ndarray:
    """Return a binary mask excluding outer rig edges and room boundaries."""
    mask = np.zeros((height, width), dtype=np.float32)
    # Lower-ring cameras (cam01..cam05) vs upper-ring cameras (cam06..cam10)
    if cam_idx in (1, 2, 3, 4, 5):
        mask[int(0.12 * height) : int(0.88 * height), int(0.10 * width) : int(0.85 * width)] = 1.0
    else:
        mask[int(0.04 * height) : int(0.75 * height), int(0.10 * width) : int(0.90 * width)] = 1.0
    return mask


def generate_soft_roi_mask(
    height: int, width: int, cam_idx: int, blur_sigma: float = 35.0
) -> np.ndarray:
    """Fallback: generates a soft ROI rectangle around the turbine housing."""
    mask = np.zeros((height, width), dtype=np.float32)
    if cam_idx in (1, 2, 3, 4, 5):
        y0, y1 = int(0.18 * height), int(0.70 * height)
        x0, x1 = int(0.20 * width), int(0.82 * width)
    else:
        y0, y1 = int(0.08 * height), int(0.65 * height)
        x0, x1 = int(0.18 * width), int(0.85 * width)

    mask[y0:y1, x0:x1] = 1.0
    ksize = int(blur_sigma * 4) | 1  # ensure odd kernel size
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), blur_sigma)
    if blurred.max() > 0:
        blurred /= blurred.max()
    return blurred.astype(np.float32)


def compute_camera_bandpass_variance(
    frame_paths: list[Path],
    fps: float,
    f_low: float = 3.5,
    f_high: float = 7.5,
    chunk_rows: int = 300,
) -> np.ndarray:
    """Compute per-pixel temporal variance of band-passed frame differences with drift suppression."""
    N = len(frame_paths)
    if N < 4:
        raise ValueError(f"Need at least 4 frames to compute temporal bandpass, got {N}")

    sample_img = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    if sample_img is None:
        raise RuntimeError(f"Failed to read sample frame {frame_paths[0]}")
    h, w = sample_img.shape

    # Frequency bins for N-1 frame differences
    freqs = np.fft.rfftfreq(N - 1, d=1.0 / fps)
    band_mask = (freqs >= f_low) & (freqs <= f_high)
    band_idx = torch.tensor(np.where(band_mask)[0])

    if len(band_idx) == 0:
        # Fallback to nearest frequency bin if band is too narrow
        center_f = (f_low + f_high) / 2.0
        nearest = int(np.argmin(np.abs(freqs - center_f)))
        band_idx = torch.tensor([nearest])

    # Pre-allocate frames buffer in uint8
    frames = np.empty((N, h, w), dtype=np.uint8)
    for i, p in enumerate(frame_paths):
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"Failed to read image at {p}")
        frames[i] = im

    # Robust per-frame median luminance normalization to reject global auto-exposure drift
    medians = [float(np.median(frames[i])) for i in range(N)]
    mean_med = float(np.mean(medians))
    med_offsets = torch.tensor([m - mean_med for m in medians], dtype=torch.float32).view(N, 1, 1)

    variance_map = torch.zeros((h, w), dtype=torch.float32)

    # Process in vertical chunks to avoid high memory allocation
    for r_start in range(0, h, chunk_rows):
        r_end = min(r_start + chunk_rows, h)
        chunk = torch.from_numpy(frames[:, r_start:r_end, :]).float()
        # Suppress global frame luminance fluctuations
        chunk = chunk - med_offsets
        diff = torch.diff(chunk, dim=0)  # [N-1, chunk_rows, w]
        fft = torch.fft.rfft(diff, dim=0)
        # Parseval relation: sum of power in positive frequency bins
        power = torch.sum(torch.abs(fft[band_idx]) ** 2, dim=0) * (2.0 / ((N - 1) ** 2))
        variance_map[r_start:r_end, :] = power

    return variance_map.numpy()


def postprocess_motion_mask(
    vmap: np.ndarray,
    cam_idx: int,
    noise_percentile: float = 75.0,
    dilation_kernel: int = 31,
    dilation_iters: int = 2,
    smooth_sigma: float = 15.0,
) -> np.ndarray:
    """Post-process variance map: rig gating, thresholding, dilation, smoothing."""
    h, w = vmap.shape
    rig_roi = get_camera_rig_roi(h, w, cam_idx)
    vmap_gated = vmap * rig_roi

    pos_vals = vmap_gated[vmap_gated > 0]
    if len(pos_vals) == 0 or pos_vals.max() <= 1e-6:
        # Entirely static scene / empty variance
        return np.zeros((h, w), dtype=np.float32)

    # Threshold noise floor
    thresh = float(np.percentile(pos_vals, noise_percentile))
    signal = np.maximum(vmap_gated - thresh, 0.0)

    pos_signal = signal[signal > 0]
    if len(pos_signal) == 0 or pos_signal.max() <= 1e-6:
        return np.zeros((h, w), dtype=np.float32)

    # Robust quantile normalization (99th percentile) instead of single-pixel max
    p99 = float(np.percentile(pos_signal, 99.0))
    if p99 <= 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    signal_norm = np.clip(signal / p99, 0.0, 1.0).astype(np.float32)

    # Morphological dilation to solidly cover moving boundaries and wobble envelope
    if dilation_kernel > 1 and dilation_iters > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel)
        )
        dilated = cv2.dilate(signal_norm, kernel, iterations=dilation_iters)
    else:
        dilated = signal_norm

    # Smooth transition boundary with Gaussian blur
    if smooth_sigma > 0:
        ksize = int(smooth_sigma * 4) | 1
        smoothed = cv2.GaussianBlur(dilated, (ksize, ksize), smooth_sigma)
    else:
        smoothed = dilated

    if smoothed.max() > 0:
        smoothed /= smoothed.max()

    return smoothed.astype(np.float32)


def process_scene_camera(
    cam_dir: Path,
    cam_idx: int,
    fps: float,
    mode: str = "bandpass",
    f_low: float = 3.5,
    f_high: float = 7.5,
    dilation_kernel: int = 31,
    dilation_iters: int = 2,
    smooth_sigma: float = 15.0,
    noise_percentile: float = 75.0,
    chunk_rows: int = 300,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Process a single camera folder and return normalized motion mask [0.0, 1.0]."""
    frame_files = sorted(list(cam_dir.glob("*.jpg")))
    if not frame_files:
        frame_files = sorted(list(cam_dir.glob("*.png")))
    if not frame_files:
        raise FileNotFoundError(f"No image frames found in {cam_dir}")

    t0 = time.time()
    sample = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
    h, w = sample.shape

    if mode == "roi":
        mask = generate_soft_roi_mask(h, w, cam_idx)
        calc_time = time.time() - t0
    elif mode == "bandpass":
        vmap = compute_camera_bandpass_variance(
            frame_files, fps=fps, f_low=f_low, f_high=f_high, chunk_rows=chunk_rows
        )
        mask = postprocess_motion_mask(
            vmap,
            cam_idx=cam_idx,
            noise_percentile=noise_percentile,
            dilation_kernel=dilation_kernel,
            dilation_iters=dilation_iters,
            smooth_sigma=smooth_sigma,
        )
        calc_time = time.time() - t0
    elif mode == "combined":
        vmap = compute_camera_bandpass_variance(
            frame_files, fps=fps, f_low=f_low, f_high=f_high, chunk_rows=chunk_rows
        )
        bp_mask = postprocess_motion_mask(
            vmap,
            cam_idx=cam_idx,
            noise_percentile=noise_percentile,
            dilation_kernel=dilation_kernel,
            dilation_iters=dilation_iters,
            smooth_sigma=smooth_sigma,
        )
        roi_mask = generate_soft_roi_mask(h, w, cam_idx)
        mask = bp_mask * roi_mask
        if mask.max() > 0:
            mask /= mask.max()
        calc_time = time.time() - t0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    stats = {
        "n_frames": len(frame_files),
        "height": h,
        "width": w,
        "mode": mode,
        "runtime_s": round(calc_time, 2),
        "mask_mean": round(float(mask.mean()), 4),
        "mask_max": round(float(mask.max()), 4),
        "fraction_active": round(float((mask > 0.1).mean()), 4),
    }
    return mask, stats


def precompute_scene_motion_masks(
    scene_dir: Path,
    mode: str = "bandpass",
    f_low: float = 3.5,
    f_high: float = 7.5,
    fps: float | None = None,
    dilation_kernel: int = 31,
    dilation_iters: int = 2,
    smooth_sigma: float = 15.0,
    noise_percentile: float = 75.0,
    chunk_rows: int = 300,
    out_dir: Path | None = None,
    save_heatmaps: bool = True,
) -> dict[str, Any]:
    """Precompute and save motion masks for all cameras in a scene."""
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")

    # Determine FPS from offsets.json if not explicitly provided
    if fps is None:
        offsets_path = scene_dir / "offsets.json"
        if offsets_path.exists():
            try:
                meta = json.loads(offsets_path.read_text(encoding="utf-8"))
                fps = float(meta.get("fps", 20.0))
            except Exception:
                fps = 20.0
        else:
            fps = 20.0

    target_out = out_dir if out_dir is not None else (scene_dir / "motion_masks")
    target_out.mkdir(parents=True, exist_ok=True)

    cam_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("cam")])
    if not cam_dirs:
        raise RuntimeError(f"No cam* directories found in {scene_dir}")

    manifest: dict[str, Any] = {
        "scene": scene_dir.name,
        "mode": mode,
        "fps": round(fps, 4),
        "f_low_hz": f_low,
        "f_high_hz": f_high,
        "dilation_kernel": dilation_kernel,
        "dilation_iters": dilation_iters,
        "smooth_sigma": smooth_sigma,
        "noise_percentile": noise_percentile,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": {},
    }

    print(f"\n[{scene_dir.name}] Precomputing motion masks for {len(cam_dirs)} cameras (mode={mode}, fps={fps:.2f})...")
    for cam_d in cam_dirs:
        cam_name = cam_d.name
        try:
            cam_idx = int(cam_name.replace("cam", ""))
        except ValueError:
            cam_idx = 1

        mask, stats = process_scene_camera(
            cam_d,
            cam_idx=cam_idx,
            fps=fps,
            mode=mode,
            f_low=f_low,
            f_high=f_high,
            dilation_kernel=dilation_kernel,
            dilation_iters=dilation_iters,
            smooth_sigma=smooth_sigma,
            noise_percentile=noise_percentile,
            chunk_rows=chunk_rows,
        )

        # Save numpy float32 mask
        npy_path = target_out / f"{cam_name}.npy"
        np.save(str(npy_path), mask)

        # Save grayscale visualization
        vis_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        vis_path = target_out / f"{cam_name}_vis.png"
        cv2.imwrite(str(vis_path), vis_u8)

        # Save colormapped heatmap inspection artifact
        if save_heatmaps:
            heatmap = cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO)
            heatmap_path = target_out / f"{cam_name}_heatmap.png"
            cv2.imwrite(str(heatmap_path), heatmap)

        # Save visual overlay inspection artifact (blended over first frame)
        frame_files = sorted(list(cam_d.glob("*.jpg")) + list(cam_d.glob("*.png")))
        if frame_files:
            f0 = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
            if f0 is not None:
                color_mask = cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)
                base_bgr = cv2.cvtColor(f0, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(base_bgr, 0.6, color_mask, 0.4, 0)
                overlay_path = target_out / f"{cam_name}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)

        import hashlib
        stats["sha256"] = hashlib.sha256(npy_path.read_bytes()).hexdigest()
        stats["min"] = float(mask.min())
        stats["max"] = float(mask.max())
        stats["p50"] = float(np.percentile(mask, 50.0))
        stats["p95"] = float(np.percentile(mask, 95.0))
        stats["p99"] = float(np.percentile(mask, 99.0))

        manifest["cameras"][cam_name] = stats
        print(f"  {cam_name}: active={stats['fraction_active']*100:.1f}%, mean={stats['mask_mean']:.3f}, runtime={stats['runtime_s']}s -> {npy_path.name}")

    import hashlib
    h_all = hashlib.sha256()
    for cname in sorted(manifest["cameras"].keys()):
        h_all.update(cname.encode("utf-8"))
        npy_file = target_out / f"{cname}.npy"
        if npy_file.is_file():
            h_all.update(npy_file.read_bytes())
    manifest["weight_maps_hash"] = f"sha256:{h_all.hexdigest()}"

    # Save both weight_maps_meta.json (W3 spec) and manifest.json (legacy)
    weight_meta_path = target_out / "weight_maps_meta.json"
    weight_meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_path = target_out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Saved weight_maps_meta -> {weight_meta_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Task T36: Precompute motion-mask weighted loss maps (R2) for 4DGS reconstructions."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene directory name under data/multipleview/ (e.g. lego_turbine_2_missing)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Explicit absolute or relative path to scene directory",
    )
    parser.add_argument(
        "--all-turbines",
        action="store_true",
        help="Process all 4 known turbine wobble datasets",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bandpass", "roi", "combined"],
        default="bandpass",
        help="Mask generation mode: bandpass (default, 6-9 Hz temporal variance), roi (soft rectangle), or combined",
    )
    parser.add_argument(
        "--f-low",
        type=float,
        default=3.5,
        help="Low cutoff frequency for wobble band-pass in Hz (default: 3.5, covering 5.69 Hz fundamental)",
    )
    parser.add_argument(
        "--f-high",
        type=float,
        default=7.5,
        help="High cutoff frequency for wobble band-pass in Hz (default: 7.5)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate in Hz (default: read from scene offsets.json if present, else 20.0)",
    )
    parser.add_argument(
        "--dilation-kernel",
        type=int,
        default=31,
        help="Morphological dilation kernel size in pixels (default: 31)",
    )
    parser.add_argument(
        "--dilation-iters",
        type=int,
        default=2,
        help="Morphological dilation iterations (default: 2)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=15.0,
        help="Gaussian blur sigma for boundary smoothing (default: 10.0)",
    )
    parser.add_argument(
        "--noise-percentile",
        type=float,
        default=75.0,
        help="Percentile of variance to treat as background noise floor (default: 75.0)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=300,
        help="Number of rows per processing chunk to manage memory (default: 300)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Custom output directory (default: <scene>/motion_masks)",
    )

    args = parser.parse_args()

    targets: list[Path] = []
    if args.all_turbines:
        for s in TURBINE_SCENES:
            p = MULTIPLEVIEW_DIR / s
            if p.exists():
                targets.append(p)
            else:
                print(f"[Warning] Turbine scene {s} not found at {p}")
    elif args.data_dir:
        targets.append(Path(args.data_dir).resolve())
    elif args.scene:
        targets.append((MULTIPLEVIEW_DIR / args.scene).resolve())
    else:
        # Default to lego_turbine_2_missing
        targets.append(MULTIPLEVIEW_DIR / "lego_turbine_2_missing")

    out_p = Path(args.out_dir).resolve() if args.out_dir else None

    for t in targets:
        precompute_scene_motion_masks(
            scene_dir=t,
            mode=args.mode,
            f_low=args.f_low,
            f_high=args.f_high,
            fps=args.fps,
            dilation_kernel=args.dilation_kernel,
            dilation_iters=args.dilation_iters,
            smooth_sigma=args.smooth_sigma,
            noise_percentile=args.noise_percentile,
            chunk_rows=args.chunk_rows,
            out_dir=out_p,
            save_heatmaps=True,
        )

    print("\nPrecomputation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

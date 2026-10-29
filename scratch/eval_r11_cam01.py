"""Head-aware eval for the r11 real-scene retrain (turbine_r1_missing_6_r11-9d7aa4a4).

Tracks the same image patches in (a) the real captured cam01 frames and
(b) the head-aware rendered cam01 series, then compares per-patch FFT
amplitude/phase at the 5.691 Hz wobble line. Background patches must be
~static in the render (fourier_head_static_background=True).
"""
import json
import os
import sys

import cv2
import numpy as np

SCENE = "data/multipleview/turbine_r1_missing_6"
RENDER_DIR = "runs/turbine_r1_missing_6_r11-9d7aa4a4/cam01_series_head"
OUT_JSON = "runs/turbine_r1_missing_6_r11-9d7aa4a4/eval_cam01_head.json"
OUT_TSTD = "scratch/r11_eval_tstd.png"
FPS = 20.004025315610498
WOBBLE_HZ = 5.690745121784875
PATCH = 100  # patch size (px, in real-frame coordinates)


def load_series(paths):
    """Generator of grayscale float32 frames."""
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"cannot read {p}")
        yield img.astype(np.float32) / 255.0


def track_patch(series, cx, cy, size):
    """phaseCorrelate each frame's patch against frame 0's template."""
    h = size // 2
    win = cv2.createHanningWindow((size, size), cv2.CV_32F)
    disp = []
    tpl = None
    for fr in series:
        patch = fr[cy - h:cy + h, cx - h:cx + h]
        if patch.shape != (size, size):
            raise RuntimeError(f"patch out of bounds at ({cx},{cy})")
        if tpl is None:
            tpl = patch.copy()
            disp.append((0.0, 0.0))
            continue
        (dx, dy), _ = cv2.phaseCorrelate(tpl, patch, win)
        disp.append((dx, dy))
    return np.array(disp)  # (N, 2)


def analyze(disp):
    """Project on principal axis; FFT amplitude at wobble line + dominant peak."""
    n = len(disp)
    d = disp - disp.mean(axis=0, keepdims=True)
    cov = d.T @ d / n
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    proj = d @ axis  # (N,)
    spec = np.abs(np.fft.rfft(proj * np.hanning(n))) / (n / 2)
    freqs = np.fft.rfftfreq(n, d=1.0 / FPS)
    k_wob = int(np.argmin(np.abs(freqs - WOBBLE_HZ)))
    # dominant peak above 0.5 Hz (exclude DC/drift)
    lo = int(np.argmin(np.abs(freqs - 0.5)))
    k_dom = lo + int(np.argmax(spec[lo:]))
    phase = np.angle(np.fft.rfft(proj * np.hanning(n))[k_wob])
    return {
        "amp_at_wobble_px": float(spec[k_wob]),
        "dom_peak_hz": float(freqs[k_dom]),
        "dom_peak_amp_px": float(spec[k_dom]),
        "phase_rad": float(phase),
        "axis_angle_deg": float(np.degrees(np.arctan2(axis[1], axis[0]))),
        "rms_px": float(np.sqrt((proj ** 2).mean())),
    }


def main():
    mask = np.load(os.path.join(SCENE, "motion_masks", "cam01.npy"))
    print("mask shape:", mask.shape, "active frac:", float(mask.mean()))
    mh, mw = mask.shape

    real_paths = sorted(
        os.path.join(SCENE, "cam01", f)
        for f in os.listdir(os.path.join(SCENE, "cam01")) if f.endswith(".jpg")
    )
    render_paths = sorted(
        os.path.join(RENDER_DIR, f)
        for f in os.listdir(RENDER_DIR) if f.endswith(".png")
    )
    print("real frames:", len(real_paths), "rendered frames:", len(render_paths))

    real0 = cv2.imread(real_paths[0], cv2.IMREAD_GRAYSCALE)
    ren0 = cv2.imread(render_paths[0], cv2.IMREAD_GRAYSCALE)
    rh, rw = real0.shape
    nh, nw = ren0.shape
    print("real size:", real0.shape, "render size:", ren0.shape)
    sx, sy = nw / rw, nh / rh  # render px per real px

    # --- rotor patch: densest PATCH x PATCH cell of the mask (real coords) ---
    integral = cv2.integral(mask.astype(np.float64))
    h = PATCH // 2
    best, bpos = -1.0, None
    for cy in range(h, mh - h, 20):
        for cx in range(h, mw - h, 20):
            s = (integral[cy + h, cx + h] - integral[cy - h, cx + h]
                 - integral[cy + h, cx - h] + integral[cy - h, cx - h])
            if s > best:
                best, bpos = s, (cx, cy)
    rotor_real = bpos
    print("rotor patch (real coords):", rotor_real, "mask frac in patch:",
          best / PATCH / PATCH)

    # --- background patches: high texture, zero mask activity in/around cell ---
    dil = cv2.dilate(mask.astype(np.uint8), np.ones((151, 151), np.uint8))
    grad = cv2.Sobel(real0, cv2.CV_32F, 1, 1, ksize=3)
    gabs = cv2.GaussianBlur(np.abs(grad), (31, 31), 0)
    bg_patches = []
    cand = []
    for cy in range(h, mh - h, 40):
        for cx in range(h, mw - h, 40):
            if dil[cy, cx] == 0:
                cand.append((float(gabs[cy, cx]), (cx, cy)))
    cand.sort(reverse=True)
    for _, pos in cand:
        if all((pos[0] - p[0]) ** 2 + (pos[1] - p[1]) ** 2 > 400 ** 2 for p in bg_patches):
            bg_patches.append(pos)
        if len(bg_patches) == 3:
            break
    print("background patches (real coords):", bg_patches)

    results = {"render_size": [nw, nh], "real_size": [rw, rh]}

    # track rotor in real + rendered; backgrounds in rendered (and real as control)
    for name, (cx, cy), which in (
        [("rotor", rotor_real, "both")]
        + [(f"bg{i+1}", p, "both") for i, p in enumerate(bg_patches)]
    ):
        entry = {}
        for tag, paths, sc in (("real", real_paths, 1.0), ("render", render_paths, None)):
            if which == "both":
                if sc is None:
                    ccx, ccy, sz = int(round(cx * sx)), int(round(cy * sy)), int(round(PATCH * sx))
                else:
                    ccx, ccy, sz = cx, cy, PATCH
                disp = track_patch(load_series(paths), ccx, ccy, sz)
                a = analyze(disp)
                if sc is None:  # convert render-px amplitudes back to real px
                    for k in ("amp_at_wobble_px", "dom_peak_amp_px", "rms_px"):
                        a[k] /= sx
                entry[tag] = a
        results[name] = entry
        r, n_ = entry["real"], entry["render"]
        print(f"\n[{name}] real: amp@5.69={r['amp_at_wobble_px']:.3f}px "
              f"dom={r['dom_peak_hz']:.2f}Hz rms={r['rms_px']:.3f}px | "
              f"render: amp@5.69={n_['amp_at_wobble_px']:.3f}px "
              f"dom={n_['dom_peak_hz']:.2f}Hz rms={n_['rms_px']:.3f}px")

    # temporal-std map of rendered series (visual check of what moved)
    acc, acc2, cnt = None, None, 0
    for fr in load_series(render_paths):
        fr = cv2.resize(fr, (nw // 2, nh // 2))
        acc = fr if acc is None else acc + fr
        acc2 = fr * fr if acc2 is None else acc2 + fr * fr
        cnt += 1
    mean = acc / cnt
    tstd = np.sqrt(np.maximum(acc2 / cnt - mean * mean, 0))
    tstd_u8 = np.clip(tstd / max(tstd.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(OUT_TSTD, cv2.applyColorMap(tstd_u8, cv2.COLORMAP_JET))
    print("\ntemporal-std map ->", OUT_TSTD)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("results ->", OUT_JSON)


if __name__ == "__main__":
    main()

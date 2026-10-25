import sys
sys.path.insert(0, ".")
from pathlib import Path
import json
import numpy as np
import cv2

scene = "lego_turbine_2_missing"
scene_dir = Path(f"data/multipleview/{scene}")
offsets_json_path = scene_dir / "offsets.json"
offsets_sidecar = json.loads(offsets_json_path.read_text())["offsets_s"]

fps = 20.00393
n_frames = 164
t_arr = np.arange(n_frames) / fps

# Let's extract 1D motion signal for each camera
signals = {}

for c in range(1, 11):
    cam = f"cam{c:02d}"
    frames_dir = scene_dir / cam
    frame_files = sorted(list(frames_dir.glob("*.jpg")))
    
    f0 = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
    
    # We find 30 good corners on the turbine body
    mask = np.zeros_like(f0, dtype=np.uint8)
    if c in [1, 2, 3, 4, 5]:
        mask[300:1500, 400:2600] = 255
    else:
        mask[100:1200, 400:2800] = 255
        
    corners = cv2.goodFeaturesToTrack(f0, maxCorners=40, qualityLevel=0.03, minDistance=30, mask=mask)
    if corners is None or len(corners) == 0:
        continue
    corners = corners[:, 0, :]
    
    patch_r = 20
    search_r = 30
    patches = []
    valid_pts = []
    for pt in corners:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if x - patch_r - search_r >= 0 and x + patch_r + search_r < f0.shape[1] and \
           y - patch_r - search_r >= 0 and y + patch_r + search_r < f0.shape[0]:
            patch = f0[y - patch_r : y + patch_r + 1, x - patch_r : x + patch_r + 1]
            patches.append(patch)
            valid_pts.append((x, y))
            
    tracks_x = np.zeros((n_frames, len(patches)), dtype=np.float32)
    tracks_y = np.zeros((n_frames, len(patches)), dtype=np.float32)
    
    for i in range(n_frames):
        img = cv2.imread(str(frame_files[i]), cv2.IMREAD_GRAYSCALE)
        for p_idx, ((x0, y0), patch) in enumerate(zip(valid_pts, patches)):
            search_roi = img[y0 - search_r - patch_r : y0 + search_r + patch_r + 1,
                             x0 - search_r - patch_r : x0 + search_r + patch_r + 1]
            res = cv2.matchTemplate(search_roi, patch, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            mx, my = max_loc
            if 1 <= mx < res.shape[1] - 1 and 1 <= my < res.shape[0] - 1:
                dx = (res[my, mx + 1] - res[my, mx - 1]) / (2 * (2 * res[my, mx] - res[my, mx + 1] - res[my, mx - 1]) + 1e-6)
                dy = (res[my + 1, mx] - res[my - 1, mx]) / (2 * (2 * res[my, mx] - res[my + 1, mx] - res[my - 1, mx]) + 1e-6)
                dx = np.clip(dx, -1.0, 1.0)
                dy = np.clip(dy, -1.0, 1.0)
            else:
                dx, dy = 0.0, 0.0
            tracks_x[i, p_idx] = (x0 - search_r) + mx + dx - x0
            tracks_y[i, p_idx] = (y0 - search_r) + my + dy - y0
            
    # Compute median displacement across points (robust to outliers)
    med_x = np.median(tracks_x, axis=1)
    med_y = np.median(tracks_y, axis=1)
    
    # Detrend
    poly_x = np.polyfit(t_arr, med_x, 1)
    poly_y = np.polyfit(t_arr, med_y, 1)
    x_detrend = med_x - np.polyval(poly_x, t_arr)
    y_detrend = med_y - np.polyval(poly_y, t_arr)
    
    # Project onto principal axis of motion
    cov = np.cov(np.stack([x_detrend, y_detrend]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_vec = eigvecs[:, -1]
    
    sig_1d = x_detrend * principal_vec[0] + y_detrend * principal_vec[1]
    signals[cam] = sig_1d
    print(f"[{cam}] Processed {len(patches)} points. Principal amp std={sig_1d.std():.3f} px")

# Find optimal common frequency f0 in [5.0, 7.5] Hz
freq_candidates = np.linspace(5.0, 7.5, 501)
best_f0 = None
best_r2_sum = -1.0

for f in freq_candidates:
    omega = 2.0 * np.pi * f
    cos_t = np.cos(omega * t_arr)
    sin_t = np.sin(omega * t_arr)
    ones = np.ones_like(t_arr)
    A_mat = np.stack([ones, cos_t, sin_t], axis=1)
    
    r2_sum = 0.0
    for cam, sig in signals.items():
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, sig, rcond=None)
        pred = A_mat @ coeffs
        ss_tot = np.sum((sig - np.mean(sig))**2)
        ss_res = np.sum((sig - pred)**2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        r2_sum += max(0.0, r2)
        
    if r2_sum > best_r2_sum:
        best_r2_sum = r2_sum
        best_f0 = f

print(f"\n==========================================")
print(f"Best coherent frequency: f0 = {best_f0:.4f} Hz (Period = {1000.0/best_f0:.2f} ms)")
print(f"==========================================")

# Fit amplitude and phase at best_f0
T_period = 1.0 / best_f0
omega = 2.0 * np.pi * best_f0
cos_t = np.cos(omega * t_arr)
sin_t = np.sin(omega * t_arr)
ones = np.ones_like(t_arr)
A_mat = np.stack([ones, cos_t, sin_t], axis=1)

fits = {}
for cam, sig in sorted(signals.items()):
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, sig, rcond=None)
    c0, c1, c2 = coeffs
    # sig = c0 + c1*cos(w*t) + c2*sin(w*t) = c0 + Amp * sin(w*t + phi)
    # sin(w*t + phi) = sin(w*t)*cos(phi) + cos(w*t)*sin(phi)
    # => c2 = Amp*cos(phi), c1 = Amp*sin(phi)
    amp = np.sqrt(c1**2 + c2**2)
    phi = np.arctan2(c1, c2) # in [-pi, pi]
    fits[cam] = {"amp": amp, "phi": phi}

ref_phi = fits["cam01"]["phi"]

print(f"{'Camera':<8} | {'Amp (px)':<9} | {'Phase (rad)':<12} | {'Sidecar (ms)':<13} | {'Optical (ms)':<13} | {'Error (ms)':<10}")
print("-" * 75)

for cam, f_info in sorted(fits.items()):
    phi_diff = f_info["phi"] - ref_phi
    # delta_t = phi_diff / omega
    delta_s = phi_diff / omega
    sidecar_s = offsets_sidecar[cam]
    
    # Check modulo period and sign ambiguity (delta_s + k*T, or delta_s + (k+0.5)*T)
    # Find which alias is closest to sidecar_s
    best_candidate_s = delta_s
    min_diff = 1e9
    for k in [-1, 0, 1]:
        # Positive sign alias
        cand = delta_s + k * T_period
        if abs(cand - sidecar_s) < min_diff:
            min_diff = abs(cand - sidecar_s)
            best_candidate_s = cand
        # Negated sign alias (due to opposite viewing direction across rig)
        cand_neg = (delta_s + 0.5 * T_period) + k * T_period
        # wrap into [-T/2, T/2] relative to sidecar
        if abs(cand_neg - sidecar_s) < min_diff:
            min_diff = abs(cand_neg - sidecar_s)
            best_candidate_s = cand_neg
            
    err_ms = (best_candidate_s - sidecar_s) * 1000.0
    print(f"{cam:<8} | {f_info['amp']:9.3f} | {f_info['phi']:12.4f} | {sidecar_s*1000.0:13.2f} | {best_candidate_s*1000.0:13.2f} | {err_ms:10.2f}")

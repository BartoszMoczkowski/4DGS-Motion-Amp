import cv2
import numpy as np
from pathlib import Path

scene = "lego_turbine_2_missing"
fps = 20.00393

for c in range(1, 11):
    cam = f"cam{c:02d}"
    frames_dir = Path(f"data/multipleview/{scene}/{cam}")
    frame_files = sorted(list(frames_dir.glob("*.jpg")))
    n_frames = len(frame_files)
    
    first_img = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
    
    # Motion detection by frame difference
    f10 = cv2.imread(str(frame_files[10]), cv2.IMREAD_GRAYSCALE)
    diff = cv2.absdiff(first_img, f10)
    thresh = diff > 20
    ys, xs = np.where(thresh)
    if len(ys) == 0:
        print(f"{cam}: No motion detected in diff!")
        continue
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    
    mask = np.zeros_like(first_img, dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 255
    
    pts0 = cv2.goodFeaturesToTrack(first_img, maxCorners=100, qualityLevel=0.01, minDistance=15, mask=mask)
    if pts0 is None or len(pts0) < 5:
        print(f"{cam}: Too few corners ({0 if pts0 is None else len(pts0)})")
        continue
        
    curr_pts = pts0
    prev_img = first_img
    trajectories = [curr_pts.copy()]
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    valid_mask = np.ones(len(pts0), dtype=bool)

    for i in range(1, n_frames):
        img = cv2.imread(str(frame_files[i]), cv2.IMREAD_GRAYSCALE)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, img, curr_pts, None, **lk_params)
        back_pts, status_b, err_b = cv2.calcOpticalFlowPyrLK(img, prev_img, next_pts, None, **lk_params)
        d = abs(curr_pts - back_pts).reshape(-1, 2).max(-1)
        good = (status.flatten() == 1) & (status_b.flatten() == 1) & (d < 1.0)
        valid_mask = valid_mask & good
        trajectories.append(next_pts.copy())
        curr_pts = next_pts
        prev_img = img

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 3:
        print(f"{cam}: Only {len(valid_indices)} points tracked.")
        continue
        
    traj_arr = np.array(trajectories)[:, valid_indices, 0, :]
    t_arr = np.arange(n_frames) / fps
    detrended = np.zeros_like(traj_arr)
    for p in range(traj_arr.shape[1]):
        for dim in range(2):
            poly = np.polyfit(t_arr, traj_arr[:, p, dim], deg=1)
            detrended[:, p, dim] = traj_arr[:, p, dim] - np.polyval(poly, t_arr)
            
    rfftfreq = np.fft.rfftfreq(n_frames, d=1.0/fps)
    fft_x = np.abs(np.fft.rfft(detrended[..., 0], axis=0))
    fft_y = np.abs(np.fft.rfft(detrended[..., 1], axis=0))
    fft_mag = np.sqrt(fft_x**2 + fft_y**2)
    mean_fft = np.mean(fft_mag, axis=1)
    amp_px = (mean_fft / n_frames) * 2.0
    
    # Peak across full spectrum > 1 Hz
    idx_above_1hz = [i for i, f in enumerate(rfftfreq) if f >= 1.0]
    top_idx = idx_above_1hz[np.argmax(amp_px[idx_above_1hz])]
    
    # Peak in 5 - 9 Hz
    idx_5_9 = [i for i, f in enumerate(rfftfreq) if 5.0 <= f <= 9.0]
    top_5_9 = idx_5_9[np.argmax(amp_px[idx_5_9])]
    
    print(f"{cam} (pts={len(valid_indices)}): overall top={rfftfreq[top_idx]:.2f}Hz ({amp_px[top_idx]:.3f}px) | [5-9Hz] top={rfftfreq[top_5_9]:.2f}Hz ({amp_px[top_5_9]:.3f}px, raw={mean_fft[top_5_9]:.2f})")

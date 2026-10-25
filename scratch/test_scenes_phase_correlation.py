import cv2
import numpy as np
from pathlib import Path

scenes = ['lego_turbine_base', 'lego_turbine_2_missing', 'lego_turbine_5_missing', 'turbine_r1_missing_6']

for scene in scenes:
    cam = "cam01"
    frames_dir = Path(f"data/multipleview/{scene}/{cam}")
    frame_files = sorted(list(frames_dir.glob("*.jpg")))
    n_frames = len(frame_files)
    fps = 20.004

    # Load reference frame
    ref_img = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    roi_ref = ref_img[600:1400, 700:1600]
    hanning = cv2.createHanningWindow(roi_ref.shape[::-1], cv2.CV_32F)

    shifts = []
    # Sample every frame
    for i in range(min(n_frames, 100)):
        img = cv2.imread(str(frame_files[i]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        roi_curr = img[600:1400, 700:1600]
        shift, response = cv2.phaseCorrelate(roi_ref, roi_curr, hanning)
        shifts.append(shift)

    shifts = np.array(shifts)
    dx = shifts[:, 0]
    dy = shifts[:, 1]
    dx_p2p = dx.max() - dx.min()
    dy_p2p = dy.max() - dy.min()
    print(f"=== {scene} ===")
    print(f"  dx peak-to-peak: {dx_p2p:.3f} px (std={dx.std():.3f} px)")
    print(f"  dy peak-to-peak: {dy_p2p:.3f} px (std={dy.std():.3f} px)")

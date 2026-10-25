import cv2
import numpy as np
from pathlib import Path

scene = "lego_turbine_2_missing"
cam = "cam01"
frames_dir = Path(f"data/multipleview/{scene}/{cam}")
frame_files = sorted(list(frames_dir.glob("*.jpg")))
n_frames = len(frame_files)
fps = 20.00393

# Load reference frame
ref_img = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE).astype(np.float32)

# ROI of the turbine
# In cam01: X in [700, 1600], Y in [600, 1400]
roi_ref = ref_img[600:1400, 700:1600]

hanning = cv2.createHanningWindow(roi_ref.shape[::-1], cv2.CV_32F)

shifts = []
for i in range(n_frames):
    img = cv2.imread(str(frame_files[i]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    roi_curr = img[600:1400, 700:1600]
    shift, response = cv2.phaseCorrelate(roi_ref, roi_curr, hanning)
    shifts.append(shift)

shifts = np.array(shifts) # [N, 2] (dx, dy)
dx = shifts[:, 0]
dy = shifts[:, 1]

print(f"Phase correlate dx: min={dx.min():.3f}, max={dx.max():.3f}, std={dx.std():.3f}, peak-to-peak={dx.max()-dx.min():.3f} px")
print(f"Phase correlate dy: min={dy.min():.3f}, max={dy.max():.3f}, std={dy.std():.3f}, peak-to-peak={dy.max()-dy.min():.3f} px")

# FFT of dx, dy
dx_dt = dx - np.polyval(np.polyfit(np.arange(n_frames), dx, 1), np.arange(n_frames))
dy_dt = dy - np.polyval(np.polyfit(np.arange(n_frames), dy, 1), np.arange(n_frames))

rfftfreq = np.fft.rfftfreq(n_frames, d=1.0/fps)
fft_dx = np.abs(np.fft.rfft(dx_dt)) * 2.0 / n_frames
fft_dy = np.abs(np.fft.rfft(dy_dt)) * 2.0 / n_frames

# Check top peaks
print("\nTop 5 frequencies in dx:")
for idx in np.argsort(fft_dx)[::-1][:5]:
    print(f"  Freq: {rfftfreq[idx]:.3f} Hz -> Amp: {fft_dx[idx]:.4f} px")

print("\nTop 5 frequencies in dy:")
for idx in np.argsort(fft_dy)[::-1][:5]:
    print(f"  Freq: {rfftfreq[idx]:.3f} Hz -> Amp: {fft_dy[idx]:.4f} px")

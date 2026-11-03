import os
import shutil
import sys
from pathlib import Path

run_id = sys.argv[1] if len(sys.argv) > 1 else "turbine_r1_missing_6_w4_e3-625cdb58"
e3_dir = Path(f"runs/{run_id}/train_out/train/ours_15000")
renders_dir = e3_dir / "renders"
cameras_dir = e3_dir / "cameras"

camera_order = [
    "cam05",
    "cam01",
    "cam03",
    "cam04",
    "cam07",
    "cam08",
    "cam10",
    "cam06",
    "cam02",
    "cam09",
]

n_frames = 261

if not renders_dir.exists():
    print(f"Renders dir {renders_dir} does not exist yet.")
    sys.exit(1)

renders = sorted(list(renders_dir.glob("*.png")))
print(f"Found {len(renders)} renders in {renders_dir}")

cameras_dir.mkdir(parents=True, exist_ok=True)

for cam_idx, cam_name in enumerate(camera_order):
    cam_target_dir = cameras_dir / cam_name
    cam_target_dir.mkdir(parents=True, exist_ok=True)
    
    start_idx = cam_idx * n_frames
    end_idx = start_idx + n_frames
    cam_renders = renders[start_idx:end_idx]
    
    for f_idx, src_file in enumerate(cam_renders):
        dst_file = cam_target_dir / f"frame_{f_idx+1:05d}.png"
        if not dst_file.exists():
            try:
                os.link(src_file, dst_file)
            except Exception:
                shutil.copyfile(src_file, dst_file)

print(f"Successfully linked cameras into {cameras_dir}")

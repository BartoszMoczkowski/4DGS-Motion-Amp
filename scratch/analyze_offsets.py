import json
from pathlib import Path
import numpy as np

scenes = ['lego_turbine_base', 'lego_turbine_2_missing', 'lego_turbine_5_missing', 'turbine_r1_missing_6']
synced_root = Path('camera_sync/rtsp_capture/synced')

for s in scenes:
    print(f"\n==========================================")
    print(f"Scene: {s}")
    print(f"==========================================")
    manifest = json.loads((synced_root / s / 'sync_manifest.json').read_text())
    instants = manifest['instants']
    fps = manifest.get('fps', 20.0)
    print(f"Nominal FPS from manifest: {fps:.6f}")
    
    # Timeline
    t0 = instants[0]['ntp_unix']
    t_end = instants[-1]['ntp_unix']
    duration_s = t_end - t0
    n_frames = len(instants)
    print(f"Total instants: {n_frames}, Duration: {duration_s:.4f} s, effective FPS: {(n_frames - 1) / duration_s:.6f}")
    
    times_rel = np.array([inst['ntp_unix'] - t0 for inst in instants])
    
    cams = sorted(list(instants[0]['frames'].keys()))
    print(f"{'Camera':<8} | {'Mean (ms)':<10} | {'Std (ms)':<9} | {'Min (ms)':<9} | {'Max (ms)':<9} | {'Drift (ms/s)':<12} | {'Offset (ms)':<11}")
    print("-" * 80)
    
    for cam in cams:
        residuals_ms = np.array([inst['frames'][cam]['residual_ms'] for inst in instants])
        mean_res = np.mean(residuals_ms)
        std_res = np.std(residuals_ms)
        min_res = np.min(residuals_ms)
        max_res = np.max(residuals_ms)
        
        # Fit linear model: residual_ms = offset_ms + drift_ms_per_s * t_rel
        if duration_s > 0:
            poly = np.polyfit(times_rel, residuals_ms, 1)
            drift_ms_per_s = poly[0]
            offset_ms = poly[1]
        else:
            drift_ms_per_s = 0.0
            offset_ms = mean_res
            
        print(f"{cam:<8} | {mean_res:10.4f} | {std_res:9.4f} | {min_res:9.4f} | {max_res:9.4f} | {drift_ms_per_s:12.6f} | {offset_ms:11.4f}")

import json
from pathlib import Path
import numpy as np

scenes = ['lego_turbine_base', 'lego_turbine_2_missing', 'lego_turbine_5_missing', 'turbine_r1_missing_6']
synced_root = Path('camera_sync/rtsp_capture/synced')

for s in scenes:
    print(f"\n==========================================")
    print(f"Reproducibility Test: {s}")
    print(f"==========================================")
    manifest = json.loads((synced_root / s / 'sync_manifest.json').read_text())
    instants = manifest['instants']
    n = len(instants)
    half = n // 2
    
    instants_r1 = instants[:half]
    instants_r2 = instants[half:]
    
    cams = sorted(list(instants[0]['frames'].keys()))
    print(f"{'Camera':<8} | {'Range 1 (ms)':<14} | {'Range 2 (ms)':<14} | {'Diff (ms)':<10} | {'Status'}")
    print("-" * 65)
    
    for cam in cams:
        r1 = np.mean([inst['frames'][cam]['residual_ms'] for inst in instants_r1])
        r2 = np.mean([inst['frames'][cam]['residual_ms'] for inst in instants_r2])
        diff = abs(r2 - r1)
        status = "EXACT" if diff < 0.01 else ("MINOR DRIFT" if diff < 2.0 else "DRIFTING")
        print(f"{cam:<8} | {r1:14.4f} | {r2:14.4f} | {diff:10.4f} | {status}")

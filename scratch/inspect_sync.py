import json
from pathlib import Path

scenes = ['lego_turbine_base', 'lego_turbine_2_missing', 'lego_turbine_5_missing', 'turbine_r1_missing_6']
synced_root = Path('camera_sync/rtsp_capture/synced')
recs_root = Path('camera_sync/rtsp_capture/recordings')
mv_root = Path('data/multipleview')

for s in scenes:
    sm = synced_root / s / 'sync_manifest.json'
    rm = recs_root / s / 'recording_manifest.json'
    print(f'=== {s} ===')
    print('  synced_manifest exists:', sm.exists())
    print('  recording_manifest exists:', rm.exists())
    mv_dir = mv_root / s
    if mv_dir.exists():
        cam01_frames = list((mv_dir / 'cam01').glob('*.jpg')) if (mv_dir / 'cam01').exists() else []
        print(f'  data/multipleview/{s}/cam01 frame count: {len(cam01_frames)}')
    else:
        print(f'  data/multipleview/{s} does not exist!')
    if sm.exists():
        data = json.loads(sm.read_text())
        print(f'  fps: {data.get("fps")}')
        print(f'  instants_total: {data.get("instants_total")}, kept: {data.get("instants_kept")}')
        print(f'  ref_cam: {data.get("reference_camera")}')
        print('  residuals (mean_ms):')
        for cam, res in data.get('per_camera_residuals', {}).items():
            print(f'    {cam}: mean={res.get("mean_ms"):.2f} ms, max={res.get("max_ms"):.2f} ms')

import json
from pathlib import Path

scenes = ['lego_turbine_base', 'lego_turbine_2_missing', 'lego_turbine_5_missing', 'turbine_r1_missing_6']
synced_root = Path('camera_sync/rtsp_capture/synced')
mv_root = Path('data/multipleview')

for s in scenes:
    print(f"=== {s} ===")
    synced_frames_dir = synced_root / s / 'synced'
    synced_frame_dirs = sorted([d.name for d in synced_frames_dir.iterdir() if d.is_dir()])
    print(f"  synced frame dirs count: {len(synced_frame_dirs)} (first: {synced_frame_dirs[0]}, last: {synced_frame_dirs[-1]})")
    
    mv_dir = mv_root / s
    cam_dirs = sorted([d.name for d in mv_dir.iterdir() if d.is_dir() and d.name.startswith('cam')])
    print(f"  mv cam dirs: {cam_dirs}")
    for c in cam_dirs:
        c_frames = sorted([f.name for f in (mv_dir / c).glob('*.jpg')])
        assert len(c_frames) == len(synced_frame_dirs), f"{c} frame count {len(c_frames)} != {len(synced_frame_dirs)}"
    print(f"  ALL {len(cam_dirs)} cameras in multipleview have EXACTLY {len(synced_frame_dirs)} frames!")

    # Check sync_manifest instants count
    manifest = json.loads((synced_root / s / 'sync_manifest.json').read_text())
    assert len(manifest['instants']) == len(synced_frame_dirs), f"manifest instants {len(manifest['instants'])} != {len(synced_frame_dirs)}"
    print(f"  sync_manifest.json instants count matches EXACTLY: {len(manifest['instants'])}")

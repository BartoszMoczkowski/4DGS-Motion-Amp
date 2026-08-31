import os
import sys
import subprocess
from pathlib import Path

RUNS = [
    'grid-A20mm_M2',
    'grid-A20mm_M4',
    'grid-A40mm_M8',
    'sweep-A40mm_M8-g10000',
    'sweep-A40mm_M8-g25000',
    'sweep-A40mm_M8-g50000',
    'sweep-A40mm_M8-g100000',
]

for run_id in RUNS:
    run_dir = Path('runs') / run_id
    bridge_file = run_dir / 'render_default_arguments_bridge.py'
    if not bridge_file.exists():
        bridge_file = run_dir / 'train_default_arguments_bridge.py'
    
    print(f"\n=======================================================")
    print(f"Rendering showcase video for: {run_id}")
    print(f"=======================================================")
    
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{Path.cwd()}:/workspace",
        "-w", "/workspace",
        "-e", "PYTHONPATH=/workspace/core",
        "4dgs-motion-amp-cuda:latest",
        "/opt/build/.venv/bin/python",
        "orchestrator/pipeline/vendored/cuda/render.py",
        "--model_path", f"/workspace/runs/{run_id}/train_out",
        "--iteration", "-1",
        "--configs", f"/workspace/runs/{run_id}/{bridge_file.name}",
        "--skip_train",
        "--skip_test",
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Finished rendering {run_id}!")

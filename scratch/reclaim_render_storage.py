import os
import shutil
from pathlib import Path

target_runs = [
    "turbine_r1_missing_6_w4_e3-625cdb58",
    "turbine_r1_missing_6_w3_e2_residual_l2-10e6ad8b",
    "turbine_r1_missing_6_w3_e2-fda6282c",
    "turbine_r1_missing_6_w1_e1a-5c67daa2",
]

runs_dir = Path("runs")
total_reclaimed = 0

def get_dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())

for rname in target_runs:
    rdir = runs_dir / rname
    if not rdir.is_dir():
        continue
    
    initial_sz = get_dir_size(rdir)
    print(f"\nProcessing {rname} (Initial: {initial_sz / (1024**3):.2f} GB)...")
    
    # Target directories to safely remove inside train_out
    train_out = rdir / "train_out"
    if train_out.is_dir():
        for sub in train_out.glob("**/gt"):
            if sub.is_dir():
                sz = get_dir_size(sub)
                shutil.rmtree(sub)
                print(f"  Removed redundant GT directory: {sub.relative_to(rdir)} ({sz / (1024**3):.2f} GB)")
                total_reclaimed += sz
                
        for sub in train_out.glob("**/cameras"):
            if sub.is_dir():
                sz = get_dir_size(sub)
                shutil.rmtree(sub)
                print(f"  Removed redundant cameras directory: {sub.relative_to(rdir)} ({sz / (1024**3):.2f} GB)")
                total_reclaimed += sz
                
        for sub in train_out.glob("**/renders"):
            if sub.is_dir():
                sz = get_dir_size(sub)
                shutil.rmtree(sub)
                print(f"  Removed loose PNG renders directory: {sub.relative_to(rdir)} ({sz / (1024**3):.2f} GB)")
                total_reclaimed += sz

    final_sz = get_dir_size(rdir)
    print(f"  --> Final run footprint: {final_sz / (1024**3):.2f} GB (Kept checkpoints, trajectories, videos, & metrics)")

print(f"\n==========================================")
print(f"TOTAL STORAGE RECLAIMED: {total_reclaimed / (1024**3):.2f} GB")
print(f"==========================================")

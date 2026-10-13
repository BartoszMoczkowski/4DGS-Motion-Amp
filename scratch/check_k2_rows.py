from pathlib import Path
import csv

repo_root = Path(__file__).resolve().parent.parent
backends = ['rigid', 'rigid2', 'kabsch', 'rigid2_roi', 'mask_lift_oracle', 'mbs', 'multicut']

for b in backends:
    p = repo_root / "runs" / f"cubes_seg_{b}_results.csv"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            header = reader[0]
            for r in reader[1:]:
                if len(r) > 0 and r[0].strip() == "cubes-cubes_k2_ring":
                    print(f"[{b}] row: {r}")

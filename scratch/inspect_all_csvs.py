import glob
import csv

csv_files = sorted(glob.glob("runs/grid_seg_*results.csv"))
print(f"Found {len(csv_files)} CSV files:")

for f in csv_files:
    print("=" * 60)
    print(f"File: {f}")
    with open(f, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row)

import csv

path = 'runs/grid_seg_rigid2_roi_results.csv'
with open(path, 'r', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# Keep only rows that have ari_within_roi populated (the valid re-run)
valid = [r for r in rows if r.get('ari_within_roi', '') != '']

# Rewrite with header
with open(path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(valid[0]))
    w.writeheader()
    w.writerows(valid)

print(f'Kept {len(valid)} valid rows, removed {len(rows) - len(valid)} broken rows')

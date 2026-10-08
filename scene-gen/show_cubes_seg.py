import csv

def main():
    with open('runs/cubes_seg_summary.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_impl = {}
    for r in rows:
        impl = r['impl']
        k = int(r['k'])
        ari = float(r['ari']) if r['ari'] else 0.0
        iou = float(r['mean_iou']) if r['mean_iou'] else 0.0
        n_pred = int(r['n_pred']) if r['n_pred'] else 0
        n_gt = int(r['n_gt']) if r['n_gt'] else 0
        time_s = float(r['segment_s']) if r['segment_s'] else 0.0
        by_impl.setdefault(impl, []).append((k, ari, iou, n_pred, n_gt, time_s))

    print('\n================== SUMMARY PER ALGORITHM ==================')
    print('%-18s | %-10s | %-12s | %-10s' % ('Algorithm', 'Avg ARI', 'Avg Mean IoU', 'Avg Time'))
    print('-' * 60)
    for impl, data in by_impl.items():
        avg_ari = sum(x[1] for x in data) / len(data)
        avg_iou = sum(x[2] for x in data) / len(data)
        avg_time = sum(x[5] for x in data) / len(data)
        print('%-18s | %-10.4f | %-12.4f | %-8.1fs' % (impl, avg_ari, avg_iou, avg_time))

    print('\n================== DETAILED BREAKDOWN (ARI) ==================')
    header = 'k   | ' + ' | '.join('%-20s' % impl for impl in by_impl.keys())
    print(header)
    print('-' * len(header))
    for k in range(1, 9):
        line = 'k=%d | ' % k
        vals = []
        for impl in by_impl.keys():
            match = [x for x in by_impl[impl] if x[0] == k]
            if match:
                vals.append('%8.4f (K=%2d/%d)' % (match[0][1], match[0][3], match[0][4]))
            else:
                vals.append('                    ')
        line += ' | '.join(vals)
        print(line)

    print('\n================== DETAILED BREAKDOWN (Mean IoU) ==================')
    print(header)
    print('-' * len(header))
    for k in range(1, 9):
        line = 'k=%d | ' % k
        vals = []
        for impl in by_impl.keys():
            match = [x for x in by_impl[impl] if x[0] == k]
            if match:
                vals.append('%8.4f            ' % match[0][2])
            else:
                vals.append('                    ')
        line += ' | '.join(vals)
        print(line)


if __name__ == "__main__":
    main()

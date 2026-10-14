import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

for m_name in ["grid-A20mm_M2", "grid-A40mm_M8"]:
    run_dir = Path("runs") / m_name
    traj = np.load(run_dir / "trajectories.npz")
    xyz = traj["canonical_xyz"]
    gt = np.load(list(run_dir.glob("**/gt_segmentation.npz"))[0])
    _, nn = cKDTree(gt["points"]).query(xyz, k=1)
    gt_on_pred = gt["labels"][nn]
    roi = (gt_on_pred > 0)
    sub_xyz = xyz[roi]
    sub_gt = gt_on_pred[roi]

    ply_files = list((run_dir / "train_out").glob("**/point_cloud.ply"))
    ply_files.sort(key=lambda p: int(p.parent.name.split("_")[-1]) if "_" in p.parent.name else 0)
    ply_file = ply_files[-1]

    with open(ply_file, "rb") as f:
        header = b""
        while b"end_header\n" not in header: header += f.readline()
        h_len = f.tell()
    props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
             ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4")]
    for i in range(45): props.append((f"f_rest_{i}", "<f4"))
    props.extend([("opacity", "<f4"), ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
                  ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")])
    dt = np.dtype(props)
    with open(ply_file, "rb") as f:
        f.seek(h_len)
        data = np.fromfile(f, dtype=dt)

    raw_op = data["opacity"][roi]
    op = 1.0 / (1.0 + np.exp(-np.clip(raw_op, -20.0, 20.0)))
    scales = np.exp(np.stack([data["scale_0"][roi], data["scale_1"][roi], data["scale_2"][roi]], axis=1))
    vol = np.cbrt(scales.prod(axis=1))
    aniso = scales.max(axis=1) / np.maximum(scales.min(axis=1), 1e-8)

    pairs = cKDTree(sub_xyz).query_pairs(r=0.003, output_type="ndarray")
    same = (sub_gt[pairs[:, 0]] == sub_gt[pairs[:, 1]])
    cross = ~same

    print(f"\n[{m_name}] Pairs: {len(pairs)}, same: {same.sum()}, cross: {cross.sum()}")
    print(f"Same-part pairs: dist mean={np.linalg.norm(sub_xyz[pairs[same,0]] - sub_xyz[pairs[same,1]], axis=1).mean()*1000:.2f}mm")
    print(f"Cross-part pairs: dist mean={np.linalg.norm(sub_xyz[pairs[cross,0]] - sub_xyz[pairs[cross,1]], axis=1).mean()*1000:.2f}mm")
    print(f"Same-part op_prod: median={np.median(op[pairs[same,0]]*op[pairs[same,1]]):.5f}, 5%={np.percentile(op[pairs[same,0]]*op[pairs[same,1]], 5):.5f}")
    print(f"Cross-part op_prod: median={np.median(op[pairs[cross,0]]*op[pairs[cross,1]]):.5f}, 5%={np.percentile(op[pairs[cross,0]]*op[pairs[cross,1]], 5):.5f}")
    print(f"Same-part max_vol: median={np.median(np.maximum(vol[pairs[same,0]], vol[pairs[same,1]]))*1000:.2f}mm, 95%={np.percentile(np.maximum(vol[pairs[same,0]], vol[pairs[same,1]]), 95)*1000:.2f}mm")
    print(f"Cross-part max_vol: median={np.median(np.maximum(vol[pairs[cross,0]], vol[pairs[cross,1]]))*1000:.2f}mm, 95%={np.percentile(np.maximum(vol[pairs[cross,0]], vol[pairs[cross,1]]), 95)*1000:.2f}mm")
    print(f"Same-part max_aniso: median={np.median(np.maximum(aniso[pairs[same,0]], aniso[pairs[same,1]])):.1f}, 95%={np.percentile(np.maximum(aniso[pairs[same,0]], aniso[pairs[same,1]]), 95):.1f}")
    print(f"Cross-part max_aniso: median={np.median(np.maximum(aniso[pairs[cross,0]], aniso[pairs[cross,1]])):.1f}, 95%={np.percentile(np.maximum(aniso[pairs[cross,0]], aniso[pairs[cross,1]]), 95):.1f}")

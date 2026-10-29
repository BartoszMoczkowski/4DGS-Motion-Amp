"""Derive the turbine 3D ROI bbox for lego_turbine_2_missing by multi-view
mask lifting: project the baseline static point cloud into all 10 cameras,
vote points that land inside the motion masks, take the bbox of the voted set.

Outputs:
  - data/multipleview/lego_turbine_2_missing/roi_bbox.json
  - scratch/roi_validation_cam01.png (projection overlay)
  - prints the bbox in preset-ready YAML form
"""
import json
import struct
import sys
import numpy as np
from pathlib import Path

SCENE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/multipleview/lego_turbine_2_missing")
PCD = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("runs/lego_turbine_2_missing-b9c0d39f/train_out/point_cloud/iteration_15000/point_cloud.ply")
OUT_FIG = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("scratch/roi_validation_cam01.png")
MASK_DIR = SCENE / "motion_masks"
OUT_JSON = SCENE / "roi_bbox.json"

VOTE_MIN = int(sys.argv[4]) if len(sys.argv) > 4 else 2  # cameras a point must be masked in (suppresses single-view reflections)
MASK_THRESH = 0.3     # bandpass-energy threshold
OPACITY_MIN = 0.05    # sigmoid(opacity_logit) floor (drop transparent floaters)
PAD_FRAC = 0.15       # bbox padding
VOXEL_SIZE = 1.0      # cluster-finding voxel size (scene units)
CLUSTER_RADIUS = 15   # voxels around the densest voxel kept as the turbine cluster


def read_ply_xyz_opacity(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        n = int([l.split()[2] for l in header if l.startswith("element vertex")][0])
        props = [l.split()[2] for l in header if l.startswith("property float")]
        dt = np.dtype([(p, "<f4") for p in props])
        data = np.fromfile(f, dtype=dt, count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
    return xyz, op


def read_next(fid, n, fmt, e="<"):
    b = fid.read(n)
    return b, struct.unpack(e + fmt, b)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_cameras(scene):
    with open(scene / "sparse_" / "cameras.bin", "rb") as fid:
        _, (nc,) = read_next(fid, 8, "Q")
        _, (cid, mid, w, h) = read_next(fid, 24, "iiQQ")
        npr = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}[mid]
        _, params = read_next(fid, 8 * npr, "d" * npr)
    f, cx, cy = params[0], params[1], params[2]

    cams = {}
    with open(scene / "sparse_" / "images.bin", "rb") as fid:
        _, (n,) = read_next(fid, 8, "Q")
        for _ in range(n):
            _, vals = read_next(fid, 64, "idddddddi")
            q = np.array(vals[1:5]); t = np.array(vals[5:8])
            name = b""
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name += c
            _, (n2,) = read_next(fid, 8, "Q")
            fid.read(24 * n2)
            digits = "".join(ch for ch in name.decode() if ch.isdigit())
            cam_name = "cam" + digits.zfill(2)
            cams[cam_name] = (qvec2rotmat(q), t)
    return f, cx, cy, cams


def main():
    xyz, op = read_ply_xyz_opacity(PCD)
    keep = op > OPACITY_MIN
    pts = xyz[keep]
    print(f"points: {len(xyz)} total, {len(pts)} above opacity {OPACITY_MIN}")

    f, cx, cy, cams = load_cameras(SCENE)
    votes = np.zeros(len(pts), dtype=np.int32)

    for cam_name, (R, t) in sorted(cams.items()):
        mask_path = MASK_DIR / f"{cam_name}.npy"
        if not mask_path.exists():
            print(f"  {cam_name}: mask missing, skipped")
            continue
        mask = np.load(mask_path)
        H, W = mask.shape
        pc = pts @ R.T + t                      # camera coords (R|t world->cam)
        z = pc[:, 2]
        valid = z > 1e-6
        u = f * pc[:, 0] / np.maximum(z, 1e-6) + cx
        v = f * pc[:, 1] / np.maximum(z, 1e-6) + cy
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        inb = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        hit = np.zeros(len(pts), dtype=bool)
        hit[inb] = mask[vi[inb], ui[inb]] > MASK_THRESH
        votes += hit
        print(f"  {cam_name}: {hit.sum()} points inside mask")

    voted = votes >= VOTE_MIN
    print(f"\nvoted (>= {VOTE_MIN} cams): {voted.sum()} / {len(pts)} "
          f"({voted.mean() * 100:.2f}% of kept points)")

    # Keep only the dominant spatial cluster (the turbine); drop scattered outliers
    p = pts[voted]
    ij = np.floor(p / VOXEL_SIZE).astype(np.int64)
    keys, counts = np.unique(ij, axis=0, return_counts=True)
    center_vox = keys[np.argmax(counts)]
    in_cluster = np.linalg.norm(ij - center_vox, axis=1) < CLUSTER_RADIUS
    print(f"dominant cluster: {in_cluster.sum()} / {len(p)} voted points kept "
          f"({len(p) - in_cluster.sum()} outliers dropped)")
    voted[voted] = in_cluster

    roi_pts = pts[voted]
    lo = roi_pts.min(axis=0)
    hi = roi_pts.max(axis=0)
    pad = (hi - lo) * PAD_FRAC + 1e-3
    lo_p, hi_p = lo - pad, hi + pad

    # fraction of ALL points inside padded bbox
    inside = np.all((xyz >= lo_p) & (xyz <= hi_p), axis=1)
    frac = inside.mean()
    print(f"bbox lo: {np.round(lo_p, 3).tolist()}")
    print(f"bbox hi: {np.round(hi_p, 3).tolist()}")
    print(f"bbox covers {inside.sum()} / {len(xyz)} = {frac * 100:.2f}% of ALL Gaussians")

    result = {
        "scene": "lego_turbine_2_missing",
        "method": "multiview_motion_mask_lifting",
        "source_point_cloud": str(PCD),
        "vote_min_cameras": VOTE_MIN,
        "mask_thresh": MASK_THRESH,
        "opacity_min": OPACITY_MIN,
        "pad_frac": PAD_FRAC,
        "n_voted_points": int(voted.sum()),
        "frac_of_all_gaussians": round(float(frac), 5),
        "note": "bbox form covers 92% of Gaussians (table inside box) — NOT selective; "
                "the voxel grid in roi_voxels.npz is the authoritative ROI definition",
        "roi_bbox": [round(float(v), 3) for v in lo_p] + [round(float(v), 3) for v in hi_p],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # --- authoritative ROI: dilated voxel occupancy grid ---
    margin = 2.0
    grid_lo = np.floor((xyz.min(axis=0) - margin) / VOXEL_SIZE).astype(np.int64)
    grid_hi = np.ceil((xyz.max(axis=0) + margin) / VOXEL_SIZE).astype(np.int64)
    shape = tuple((grid_hi - grid_lo + 1).tolist())
    grid = np.zeros(shape, dtype=bool)
    vox_idx = np.floor(roi_pts / VOXEL_SIZE).astype(np.int64) - grid_lo
    grid[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] = True

    # 1-voxel dilation (6-connectivity, two passes = ~1 voxel shell)
    from scipy import ndimage  # noqa
    try:
        grid = ndimage.binary_dilation(grid, iterations=2)
    except Exception:
        # numpy fallback: shift-and-or along each axis, twice
        for _ in range(2):
            g = grid.copy()
            for ax in range(3):
                g |= np.roll(grid, 1, axis=ax) | np.roll(grid, -1, axis=ax)
            grid = g

    np.savez_compressed(
        SCENE / "roi_voxels.npz",
        grid=grid,
        origin=grid_lo.astype(np.float64) * VOXEL_SIZE,
        voxel_size=np.float64(VOXEL_SIZE),
    )
    n_roi_vox = int(grid.sum())
    print(f"wrote {SCENE / 'roi_voxels.npz'}: shape {shape}, "
          f"{n_roi_vox} occupied voxels ({n_roi_vox / grid.size * 100:.3f}% of grid)")

    # selectivity of the voxel ROI on the full cloud
    all_idx = np.floor(xyz / VOXEL_SIZE).astype(np.int64) - grid_lo
    inb = np.all((all_idx >= 0) & (all_idx < np.array(shape)), axis=1)
    sel = np.zeros(len(xyz), dtype=bool)
    sel[inb] = grid[all_idx[inb, 0], all_idx[inb, 1], all_idx[inb, 2]]
    print(f"voxel ROI selects {sel.sum()} / {len(xyz)} = {sel.mean() * 100:.2f}% of ALL Gaussians")

    # vote histogram for the record
    for k in range(1, 11):
        print(f"  votes=={k}: {(votes == k).sum()}")

    # --- validation figure: voxel-ROI-selected points projected over cam01 frame ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    cam_name = "cam01"
    R, t = cams[cam_name]
    img = np.array(Image.open(SCENE / cam_name / "frame_00001.jpg"))
    pc = xyz @ R.T + t
    z = pc[:, 2]
    u = f * pc[:, 0] / np.maximum(z, 1e-6) + cx
    v = f * pc[:, 1] / np.maximum(z, 1e-6) + cy
    ok = (z > 1e-6) & (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    ax.scatter(u[ok & ~sel], v[ok & ~sel], s=0.3, c="cyan", alpha=0.3, label="scene")
    ax.scatter(u[ok & sel], v[ok & sel], s=1.0, c="red", alpha=0.6, label="ROI voxels")
    ax.legend(markerscale=8)
    ax.set_title(f"ROI validation — {cam_name} (red = voxel-ROI-selected Gaussians)")
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")

    print("\nPreset-ready YAML:")
    bbox = result["roi_bbox"]
    print(f"  roi_bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {bbox[4]}, {bbox[5]}]")


if __name__ == "__main__":
    main()

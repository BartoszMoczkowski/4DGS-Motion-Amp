import torch
import numpy as np
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def test_sam():
    print("Loading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint="/workspace/weights/sam_vit_b_01ec64.pth").to("cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    # Load 1 image from cubes
    img_path = "/workspace/runs/cubes-cubes_k2_both/convert_out/data/multipleview/cubes_k2_both/cam01/frame_00001.jpg"
    im = cv2.imread(img_path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print(f"Generating masks for {img_path} ({im_rgb.shape})...")
    
    masks = mask_generator.generate(im_rgb)
    print(f"Generated {len(masks)} masks!")
    for i, m in enumerate(masks[:5]):
        print(f"  Mask {i}: area={m['area']}, bbox={m['bbox']}, predicted_iou={m['predicted_iou']:.3f}, stability={m['stability_score']:.3f}")

if __name__ == "__main__":
    test_sam()

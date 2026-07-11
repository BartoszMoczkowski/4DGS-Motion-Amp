# 4DGS config for the synthetic animated pump (data/multipleview/pump01), produced by
# omniverse_pipeline (Omniverse capture -> omni_to_4dgs). Based on multipleview/default.py.
#
# Scene notes: rigid machine, subtle periodic per-part motion (few-mm), static cameras.
# Tunables worth trying for this regime are called out inline.
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     # last entry is the TIME resolution of the deformation grid. Capture is 60 frames;
     # 150 gives headroom. Raise if fast/high-freq motion looks temporally blurred.
     'resolution': [64, 64, 64, 150]
    },
    multires = [1, 2],
    defor_depth = 0,
    net_width = 128,
    # Regularization: keep motion smooth/periodic without erasing the subtle displacements
    # motion amplification needs. If tiny motions get smoothed away, lower these.
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes = 0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False,
)
OptimizationParams = dict(
    dataloader=True,
    iterations = 15000,
    batch_size=1,
    coarse_iterations = 3000,
    densify_until_iter = 10_000,
    # Disabled (> total iterations): the base default (3000) collides exactly with
    # coarse_iterations, resetting opacity to ~0 right as the fine-stage deformation
    # network engages for the first time -> reliably nan's the loss. multipleview/default.py
    # disables this the same way (opacity_reset_interval = 60000, commented out there).
    opacity_reset_interval = 60000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
)

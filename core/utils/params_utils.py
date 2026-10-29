def merge_hparams(args, config):
    params = ["ModelParams", "PipelineParams", "ModelHiddenParams", "OptimizationParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)

    # Reconcile shared flags across param groups when only one group was customized
    opt_cfg = config.get("OptimizationParams", {}) if "OptimizationParams" in config else {}
    pipe_cfg = config.get("PipelineParams", {}) if "PipelineParams" in config else {}
    model_cfg = config.get("ModelParams", {}) if "ModelParams" in config else {}
    hidden_cfg = config.get("ModelHiddenParams", {}) if "ModelHiddenParams" in config else {}

    if hasattr(args, "gated_coarse"):
        if pipe_cfg.get("gated_coarse", False) and not opt_cfg.get("gated_coarse", False):
            args.gated_coarse = True
    if hasattr(args, "roi_bbox"):
        if opt_cfg.get("roi_bbox", None) is None and pipe_cfg.get("roi_bbox", None) is not None:
            args.roi_bbox = pipe_cfg["roi_bbox"]
    if hasattr(args, "use_camera_offsets"):
        if "use_camera_offsets" in model_cfg and "use_camera_offsets" not in opt_cfg:
            args.use_camera_offsets = model_cfg["use_camera_offsets"]
        elif model_cfg.get("use_camera_offsets", True) is False and opt_cfg.get("use_camera_offsets", True) is True:
            args.use_camera_offsets = False
    if hasattr(args, "render_process"):
        if model_cfg.get("render_process", False) or hidden_cfg.get("render_process", False):
            args.render_process = True

    return args

from argparse import Namespace


def merge_hparams(args, config):
    """Merge hierarchical config dictionaries (`ModelParams`, `PipelineParams`,
    `ModelHiddenParams`, `OptimizationParams`) onto the flat `args` Namespace.

    Wait-order note (W0 audit fix):
    `bridge.py` emits all four parameter dicts with every field present. Because `args`
    is a single flat Namespace before `lp.extract(args)` / `op.extract(args)` / `pp.extract(args)`
    are called, shared field names (`gated_coarse`, `roi_bbox`, `use_camera_offsets`) must not
    have their `OptimizationParams` settings clobbered by `ModelParams` or `PipelineParams`
    defaults. We therefore apply `OptimizationParams` last and explicitly reconcile shared
    flags whenever `OptimizationParams` overrides the default.
    """
    params = ["ModelParams", "PipelineParams", "ModelHiddenParams", "OptimizationParams"]
    args_dict = vars(args)

    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if key in args_dict:
                    args_dict[key] = value

    # Reconcile shared flags across groups so setting a flag in either OptimizationParams
    # or its companion group (PipelineParams / ModelParams) works consistently.
    opt_cfg = config["OptimizationParams"] if "OptimizationParams" in config.keys() else {}
    pipe_cfg = config["PipelineParams"] if "PipelineParams" in config.keys() else {}
    model_cfg = config["ModelParams"] if "ModelParams" in config.keys() else {}

    if "gated_coarse" in opt_cfg or "gated_coarse" in pipe_cfg:
        args_dict["gated_coarse"] = bool(
            opt_cfg.get("gated_coarse", False) or pipe_cfg.get("gated_coarse", False)
        )
    if "roi_bbox" in opt_cfg or "roi_bbox" in pipe_cfg:
        opt_bbox = opt_cfg.get("roi_bbox", None)
        pipe_bbox = pipe_cfg.get("roi_bbox", None)
        args_dict["roi_bbox"] = opt_bbox if opt_bbox is not None else pipe_bbox
    if "use_camera_offsets" in opt_cfg and "use_camera_offsets" in model_cfg:
        # Default is True; if either explicitly disables it, honor False
        args_dict["use_camera_offsets"] = bool(
            opt_cfg["use_camera_offsets"] and model_cfg["use_camera_offsets"]
        )
    if "strict_camera_offsets" in opt_cfg or "strict_camera_offsets" in model_cfg:
        args_dict["strict_camera_offsets"] = bool(
            opt_cfg.get("strict_camera_offsets", False)
            or model_cfg.get("strict_camera_offsets", False)
        )
    if "offsets_file" in opt_cfg or "offsets_file" in model_cfg:
        args_dict["offsets_file"] = str(
            opt_cfg.get("offsets_file", "") or model_cfg.get("offsets_file", "")
        )
    if "learnable_camera_offsets" in opt_cfg or "learnable_camera_offsets" in model_cfg:
        args_dict["learnable_camera_offsets"] = bool(
            opt_cfg.get("learnable_camera_offsets", False)
            or model_cfg.get("learnable_camera_offsets", False)
        )

    return Namespace(**args_dict)

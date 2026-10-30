#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""Vendored, verbatim copy of the repo-root ``train.py`` (T09 copy-in, per
``planning/INSTRUCTIONS.md``'s "copy the logic in, don't call the original script" rule; see
``pipeline.vendored.cuda``'s package docstring). Body is byte-for-byte the reference script's,
including its own argparse ``if __name__ == "__main__":`` entry point — this file is executed as
a separate process inside the ``cuda`` container (``pipeline/stages/train.py`` builds the CLI
invocation), never imported by the orchestrator's own host process.

**Fixed on T11's real-hardware run (2026-07-18):** the original script did
``args.save_iterations.append(args.iterations)`` *before* applying ``--configs``'s
``merge_hparams`` override -- harmless for the reference script's own CLI-only workflow (where
``--iterations`` is passed directly on the command line and already final at parse time), but
wrong for this project's config-bridge-driven one, where ``OptimizationParams.iterations`` is
routinely overridden *after* parsing via the bridge file (e.g. a fast smoke-test run setting
``iterations: 100`` in config while argparse's own default is ``30_000`` --
``arguments/__init__.py``). The append picked up the stale pre-merge default, so the actual final
iteration was never in ``save_iterations`` and no checkpoint/``point_cloud`` ever got written --
``amp.default``/``render.default`` then failed downstream with ``FileNotFoundError`` on the
missing ``point_cloud`` directory, with no error at all from ``train.default`` itself (exit code
0). Moved the append to *after* the ``--configs`` merge so it always uses the true final
``iterations`` value.
"""
import copy
import json
import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from time import time

import lpips
import mmengine
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from scene.camera_offsets import CameraOffsets, resolve_camera_time
from scene.fourier_rigid import FourierRigidHead
from utils.general_utils import get_regularization_weights, safe_state
from utils.image_utils import psnr
from utils.loader_utils import FineSampler, get_stamp_list
from utils.loss_utils import l1_loss, l2_loss, lpips_loss, ssim
from utils.scene_utils import render_training_image
from utils.timer import Timer
from utils.trajectory_loss import TrajectoryReprojectionLoss

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(
    dataset,
    opt,
    hyper,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    gaussians,
    scene,
    stage,
    tb_writer,
    train_iter,
    timer,
    camera_offsets_module=None,
    offsets_optimizer=None,
    trajectory_loss_fn=None,
    fourier_head=None,
    fourier_optimizer=None,
    motion_components=None,
    motion_components_optimizer=None,
    split_merge_controller=None,
):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    #
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                collate_fn=list,
            )
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=list,
            )
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack, 0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False
        #
    count = 0
    for iteration in range(first_iter, final_iter + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    count += 1
                    viewpoint_index = (count) % len(video_cams)
                    if (count // (len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifer,
                        stage=stage,
                        cam_type=scene.dataset_type,
                    )["render"]

                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(
                        viewpoint_stack,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=list,
                    )
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size:
                viewpoint_cam = viewpoint_stack.pop(
                    randint(0, len(viewpoint_stack) - 1)
                )
                if not viewpoint_stack:
                    viewpoint_stack = temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))
        # breakpoint()
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        masks = []
        has_mask = False
        render_pkg_list = []
        for viewpoint_cam in viewpoint_cams:
            custom_time = resolve_camera_time(viewpoint_cam, camera_offsets_module)
            if custom_time is not None:
                freeze_iter = getattr(opt, "learnable_offsets_freeze_iter", 10000)
                if stage != "fine" or iteration < freeze_iter:
                    custom_time = custom_time.detach()

            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                stage=stage,
                cam_type=scene.dataset_type,
                custom_time=custom_time,
                fourier_head=fourier_head,
                motion_components=motion_components,
                iteration=iteration,
            )
            render_pkg_list.append(render_pkg)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            images.append(image.unsqueeze(0))
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam["image"].cuda()

            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

            cam_mask = getattr(viewpoint_cam, "mask", None)
            if getattr(opt, "use_motion_mask", False) and cam_mask is not None:
                has_mask = True
                if not isinstance(cam_mask, torch.Tensor):
                    cam_mask = torch.as_tensor(cam_mask, dtype=torch.float32)
                cam_mask = cam_mask.to(device=gt_image.device, dtype=torch.float32)
                if cam_mask.dim() == 2:
                    cam_mask = cam_mask.unsqueeze(0)
                if cam_mask.shape[-2:] != (gt_image.shape[1], gt_image.shape[2]):
                    cam_mask = torch.nn.functional.interpolate(
                        cam_mask.unsqueeze(0),
                        size=(gt_image.shape[1], gt_image.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                mask_max = float(getattr(opt, "motion_mask_max", 50.0) or 50.0)
                from utils.loss_utils import normalize_loss_weights
                normalize_mean = bool(getattr(opt, "normalize_motion_mask", True))
                cam_mask = normalize_loss_weights(
                    cam_mask,
                    mask_max=mask_max,
                    floor=1.0,
                    normalize_mean=normalize_mean,
                )

                masks.append(cam_mask.unsqueeze(0))
            else:
                masks.append(None)

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)

        mask_tensor = None
        if getattr(opt, "use_motion_mask", False) and has_mask:
            full_masks = []
            for i, m in enumerate(masks):
                if m is not None:
                    full_masks.append(m)
                else:
                    full_masks.append(torch.ones((1, 1, gt_images[i].shape[2], gt_images[i].shape[3]), device=gt_images[i].device))
            mask_tensor = torch.cat(full_masks, 0)

        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :], mask=mask_tensor)

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm

        loss = Ll1
        if stage == "fine":
            bypass_hexplane = (
                fourier_head is not None
                and getattr(fourier_head, "pure_rigid", False)
                and getattr(fourier_head, "static_background", False)
            )
            time_smoothness_weight, l1_time_planes_weight, plane_tv_weight = (
                get_regularization_weights(iteration, hyper)
            )
            if not bypass_hexplane and (
                time_smoothness_weight != 0
                or l1_time_planes_weight != 0
                or plane_tv_weight != 0
            ):
                tv_loss = gaussians.compute_regulation(
                    time_smoothness_weight,
                    l1_time_planes_weight,
                    plane_tv_weight,
                )
                loss += tv_loss

            if camera_offsets_module is not None:
                freeze_iter = getattr(opt, "learnable_offsets_freeze_iter", 10000)
                if iteration >= freeze_iter and getattr(opt, "lambda_offset_prior", 0.01) > 0.0:
                    offset_prior_loss = camera_offsets_module.compute_prior_loss(
                        sigma_s=getattr(opt, "offset_prior_sigma_s", 0.005),
                        weight=getattr(opt, "lambda_offset_prior", 0.01),
                    )
                    loss = loss + offset_prior_loss

            if motion_components is not None and getattr(motion_components, "is_active", True):
                t_val = custom_time if custom_time is not None else viewpoint_cams[0].time
                t_tensor = torch.as_tensor(t_val, dtype=torch.float32, device=gaussians.get_xyz.device).view(1, 1).repeat(gaussians.get_xyz.shape[0], 1)
                m_hex, _, _, _, _ = gaussians._deformation(gaussians.get_xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features, t_tensor)
                h_res = m_hex - gaussians.get_xyz
                loss_res = motion_components.compute_residual_loss(h_res, gaussians, iteration)
                loss = loss + loss_res

                # W5 (T53): Outer-loop split/merge hypothesis evaluation and proposal
                if split_merge_controller is not None:
                    if split_merge_controller.active_hypothesis is not None:
                        decided, status = split_merge_controller.evaluate_and_decide(
                            current_iter=iteration,
                            current_metric=psnr_.item(),
                            pc=gaussians,
                            mc=motion_components,
                            opt=gaussians.optimizer,
                            mc_opt=motion_components_optimizer,
                        )
                        if decided:
                            print(f"[W5 ITER {iteration}] Structural hypothesis {status}: resulting K={motion_components.num_components}")

                    if split_merge_controller.is_proposal_due(iteration):
                        for k in range(motion_components.num_components):
                            proposed = split_merge_controller.propose_split(
                                pc=gaussians,
                                mc=motion_components,
                                target_k=k,
                                iteration=iteration,
                                baseline_metric=psnr_.item(),
                                hex_residuals=h_res,
                                opt=gaussians.optimizer,
                                mc_opt=motion_components_optimizer,
                                lr=getattr(opt, "motion_components_lr", 0.001),
                                resp_lr=getattr(opt, "motion_components_responsibilities_lr", 0.01),
                            )
                            if proposed:
                                print(f"[W5 ITER {iteration}] Proposed split for component {k} -> K={motion_components.num_components}")
                                break

            if trajectory_loss_fn is not None and getattr(opt, "lambda_trajectory", 0.0) > 0.0:
                traj_losses = []
                for v_idx, viewpoint_cam in enumerate(viewpoint_cams):
                    m_final = render_pkg_list[v_idx]["means3D_final"]
                    roi_mask_t = None
                    if getattr(gaussians, "_roi_grid", None) is not None:
                        from scene.roi_grid import voxel_roi_mask
                        roi_mask_t = voxel_roi_mask(gaussians.get_xyz, gaussians._roi_grid)
                    t_loss = trajectory_loss_fn(
                        viewpoint_cam,
                        gaussians.get_xyz,
                        m_final,
                        roi_mask=roi_mask_t,
                    )
                    traj_losses.append(t_loss)
                if traj_losses:
                    loss += getattr(opt, "lambda_trajectory", 0.0) * (sum(traj_losses) / len(traj_losses))
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            loss += opt.lambda_dssim * (1.0 - ssim_loss)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss

        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            # Close the GUI listener socket before re-exec so the port is freed even if
            # a client had connected (otherwise it can be left in TIME_WAIT and the
            # restarted process's network_gui.init() fails with "Address already in use").
            try:
                network_gui.listener.close()
            except Exception:
                pass
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = (
                viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            )
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "psnr": f"{psnr_:.{2}f}",
                        "point": f"{total_point}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                [pipe, background],
                stage,
                scene.dataset_type,
                camera_offsets_module=camera_offsets_module,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
                if camera_offsets_module is not None:
                    try:
                        offsets_file = os.path.join(scene.model_path, "learned_offsets.json")
                        with open(offsets_file, "w", encoding="utf-8") as f:
                            json.dump(camera_offsets_module.get_telemetry_dict(), f, indent=2)
                    except Exception as e:
                        print(f"Warning: failed to save learned offsets at iter {iteration}: {e}")
                if motion_components is not None:
                    try:
                        torch.save(motion_components.state_dict(), os.path.join(scene.model_path, "motion_components.pth"))
                        with open(os.path.join(scene.model_path, "motion_components_state.json"), "w", encoding="utf-8") as f:
                            json.dump(motion_components.get_summary(), f, indent=2)
                        if gaussians._motion_component_logits is not None:
                            torch.save(gaussians._motion_component_logits, os.path.join(scene.model_path, "motion_component_logits.pth"))
                    except Exception as e:
                        print(f"Warning: failed to save motion components at iter {iteration}: {e}")
                if split_merge_controller is not None:
                    try:
                        with open(os.path.join(scene.model_path, "split_merge_state.json"), "w", encoding="utf-8") as f:
                            json.dump(split_merge_controller.get_summary(), f, indent=2)
                    except Exception as e:
                        print(f"Warning: failed to save split_merge_state at iter {iteration}: {e}")
                if fourier_head is not None:
                    try:
                        torch.save(fourier_head.state_dict(), os.path.join(scene.model_path, "fourier_head.pth"))
                        with open(os.path.join(scene.model_path, "fourier_head_state.json"), "w", encoding="utf-8") as f:
                            json.dump(fourier_head.get_summary(), f, indent=2)
                    except Exception as e:
                        print(f"Warning: failed to save fourier head at iter {iteration}: {e}")
                if motion_components is not None:
                    try:
                        torch.save(motion_components.state_dict(), os.path.join(scene.model_path, "motion_components.pth"))
                        with open(os.path.join(scene.model_path, "motion_components_state.json"), "w", encoding="utf-8") as f:
                            json.dump(motion_components.get_summary(), f, indent=2)
                        if gaussians._motion_component_logits is not None:
                            torch.save(gaussians._motion_component_logits, os.path.join(scene.model_path, "motion_component_logits.pth"))
                    except Exception as e:
                        print(f"Warning: failed to save motion components at iter {iteration}: {e}")
            if dataset.render_process:
                if (
                    (iteration < 1000 and iteration % 10 == 9)
                    or (iteration < 3000 and iteration % 50 == 49)
                    or (iteration < 60000 and iteration % 100 == 99)
                ):
                    # breakpoint()
                    render_training_image(
                        scene,
                        gaussians,
                        [test_cams[iteration % len(test_cams)]],
                        render,
                        pipe,
                        background,
                        stage + "test",
                        iteration,
                        timer.get_elapsed_time(),
                        scene.dataset_type,
                    )
                    render_training_image(
                        scene,
                        gaussians,
                        [train_cams[iteration % len(train_cams)]],
                        render,
                        pipe,
                        background,
                        stage + "train",
                        iteration,
                        timer.get_elapsed_time(),
                        scene.dataset_type,
                    )
                    # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor_grad, visibility_filter
                )

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (
                        opt.opacity_threshold_fine_init
                        - opt.opacity_threshold_fine_after
                    ) / (opt.densify_until_iter)
                    densify_threshold = (
                        opt.densify_grad_threshold_fine_init
                        - iteration
                        * (
                            opt.densify_grad_threshold_fine_init
                            - opt.densify_grad_threshold_after
                        )
                        / (opt.densify_until_iter)
                    )
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and gaussians.get_xyz.shape[0] < 360000
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    gaussians.densify(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        5,
                        5,
                        scene.model_path,
                        iteration,
                        stage,
                    )
                if (
                    iteration > opt.pruning_from_iter
                    and iteration % opt.pruning_interval == 0
                    and gaussians.get_xyz.shape[0] > 200000
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    gaussians.prune(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                    )

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if (
                    iteration % opt.densification_interval == 0
                    and gaussians.get_xyz.shape[0] < 360000
                    and opt.add_point
                ):
                    gaussians.grow(5, 5, scene.model_path, iteration, stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if offsets_optimizer is not None:
                freeze_iter = getattr(opt, "learnable_offsets_freeze_iter", 10000)
                if iteration >= freeze_iter and iteration < opt.iterations:
                    offsets_optimizer.step()
                    camera_offsets_module.clamp_and_anchor()
                offsets_optimizer.zero_grad(set_to_none=True)

            if fourier_optimizer is not None:
                freeze_iter = getattr(opt, "fourier_head_freeze_iter", 500)
                if iteration >= freeze_iter and iteration < opt.iterations:
                    fourier_optimizer.step()
                fourier_optimizer.zero_grad(set_to_none=True)

            if motion_components_optimizer is not None:
                if iteration < opt.iterations:
                    motion_components_optimizer.step()
                motion_components_optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path
                    + "/chkpnt"
                    + f"_{stage}_"
                    + str(iteration)
                    + ".pth",
                )
                if camera_offsets_module is not None:
                    try:
                        offsets_file = os.path.join(scene.model_path, "learned_offsets.json")
                        with open(offsets_file, "w", encoding="utf-8") as f:
                            json.dump(camera_offsets_module.get_effective_offsets(), f, indent=2)
                    except Exception:
                        pass
                if fourier_head is not None:
                    try:
                        torch.save(fourier_head.state_dict(), os.path.join(scene.model_path, f"fourier_head_{stage}_{iteration}.pth"))
                    except Exception:
                        pass


_TRAINING_CURVES = []


def _flush_training_curves(model_path):
    global _TRAINING_CURVES
    if not model_path:
        return
    try:
        curves_path = os.path.join(model_path, "training_curves.json")
        with open(curves_path, "w", encoding="utf-8") as f:
            json.dump({"curves": _TRAINING_CURVES}, f, indent=2)
    except Exception as e:
        print(f"Warning: could not write training curves to {curves_path}: {e}")


def training(
    dataset,
    hyper,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    expname,
):
    global _TRAINING_CURVES
    _TRAINING_CURVES = []
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()

    # Propagate gated coarse settings to pipe if set on opt
    if getattr(opt, "gated_coarse", False):
        pipe.gated_coarse = True
    if getattr(opt, "roi_bbox", None) is not None and getattr(pipe, "roi_bbox", None) is None:
        pipe.roi_bbox = opt.roi_bbox

    # Voxel ROI (T41): authoritative, densification-safe ROI for gated coarse /
    # Fourier head / trajectory loss. Positional lookup, survives densification.
    if getattr(opt, "use_roi_voxels", False):
        roi_npz = os.path.join(dataset.source_path, "roi_voxels.npz")
        if os.path.exists(roi_npz):
            from scene.roi_grid import load_roi_grid
            gaussians._roi_grid = load_roi_grid(roi_npz, device="cuda")
            print(f"[ROI] Loaded voxel ROI from {roi_npz}: "
                  f"{int(gaussians._roi_grid['grid'].sum())} occupied voxels")
        else:
            print(f"[ROI] WARNING: use_roi_voxels set but {roi_npz} missing — ROI gating disabled")

    camera_offsets_module = None
    offsets_optimizer = None
    if getattr(opt, "learnable_camera_offsets", False):
        cam_names = []
        init_offsets = {}
        drift = {}
        fps = 20.0
        img_len = 60
        ref_cam = "cam01"
        offsets_sha256 = None
        if hasattr(scene, "train_camera") and scene.train_camera is not None:
            cam_names = scene.train_camera.get_camera_names()
            init_offsets = scene.train_camera.offsets_s
            drift = scene.train_camera.drift_s_per_s
            fps = scene.train_camera.fps
            img_len = getattr(scene.train_camera.dataset, "image_length", 60)
            ref_cam = getattr(scene.train_camera.dataset, "reference_camera", "cam01")
            offsets_sha256 = getattr(scene.train_camera, "offsets_file_sha256", None)
        if not cam_names and hasattr(dataset, "source_path"):
            offsets_path = os.path.join(dataset.source_path, "offsets.json")
            if os.path.exists(offsets_path):
                with open(offsets_path, "r", encoding="utf-8") as f:
                    off_data = json.load(f)
                init_offsets = off_data.get("offsets_s", {})
                cam_names = list(init_offsets.keys())
                fps = float(off_data.get("fps", 20.0))
                ref_cam = str(off_data.get("reference_camera", "cam01"))
        if not cam_names:
            cam_names = [f"cam{i:02d}" for i in range(1, 11)]

        camera_offsets_module = CameraOffsets(
            cam_names=cam_names,
            init_offsets_s=init_offsets,
            reference_camera=ref_cam,
            max_clamp_s=getattr(opt, "max_offset_clamp_s", 0.040),
            fps=fps,
            image_length=img_len,
            drift_s_per_s=drift,
            offset_prior_sigma_s=getattr(opt, "offset_prior_sigma_s", 0.005),
            lambda_offset_prior=getattr(opt, "lambda_offset_prior", 0.01),
            offsets_file_sha256=offsets_sha256,
        ).cuda()
        offsets_optimizer = torch.optim.Adam(
            camera_offsets_module.parameters(),
            lr=getattr(opt, "learnable_offsets_lr", 0.0001),
        )
        print(f"[R7] Initialized CameraOffsets for {len(cam_names)} cameras, lr={getattr(opt, 'learnable_offsets_lr', 0.0001)}")

    trajectory_loss_fn = None
    if getattr(opt, "lambda_trajectory", 0.0) > 0.0:
        roi = getattr(opt, "trajectory_roi_bbox", None) or getattr(opt, "roi_bbox", None)
        track_file = getattr(opt, "trajectory_track_file", "") or None
        trajectory_loss_fn = TrajectoryReprojectionLoss(
            scene_path=dataset.source_path,
            track_file=track_file,
            roi_bbox=roi,
        )
        print(f"[R8] Initialized TrajectoryReprojectionLoss with lambda={opt.lambda_trajectory}")

    fourier_head = None
    fourier_optimizer = None
    if getattr(opt, "use_fourier_rigid_head", False):
        roi_bbox = getattr(opt, "fourier_head_roi_bbox", None) or getattr(opt, "roi_bbox", None)
        fourier_head = FourierRigidHead(
            base_frequency_hz=getattr(opt, "fourier_head_frequency_hz", 7.112),
            learn_frequency=getattr(opt, "fourier_head_learn_frequency", False),
            num_harmonics=getattr(opt, "fourier_head_num_harmonics", 3),
            fps=scene.train_camera.fps if hasattr(scene, "train_camera") else 20.0,
            image_length=getattr(scene.train_camera.dataset, "image_length", 60) if hasattr(scene, "train_camera") else 60,
            roi_bbox=roi_bbox,
            pure_rigid=getattr(opt, "fourier_head_pure_rigid", False),
            static_background=getattr(opt, "fourier_head_static_background", False),
        ).cuda()

        tracks_path = os.path.join(dataset.source_path, "trajectory_tracks.json")
        if not os.path.exists(tracks_path):
            tracks_path = os.path.join(dataset.source_path, "optical_offsets.json")
        if os.path.exists(tracks_path):
            try:
                fourier_head.init_from_optical_tracks(tracks_path)
            except Exception as e:
                print(f"Warning: could not init fourier head from {tracks_path}: {e}")

        fourier_optimizer = torch.optim.Adam(
            fourier_head.parameters(),
            lr=getattr(opt, "fourier_head_lr", 0.001),
        )
        gaussians.fourier_head = fourier_head
        print(f"[R9] Initialized FourierRigidHead at {fourier_head.effective_frequency_hz.item():.3f} Hz with {fourier_head.num_harmonics} harmonics, lr={getattr(opt, 'fourier_head_lr', 0.001)}")

    motion_components = None
    motion_components_optimizer = None
    if getattr(opt, "use_motion_components", False):
        from scene.motion_components import (
            MotionComponents,
            FourierComponentTrajectory,
            ChebyshevComponentTrajectory,
        )
        K = getattr(opt, "num_motion_components", 1)
        comp_type = getattr(opt, "motion_components_type", "fourier")
        fps = scene.train_camera.fps if hasattr(scene, "train_camera") else 20.004
        comps = []
        for _ in range(K):
            if comp_type == "fourier":
                comps.append(
                    FourierComponentTrajectory(
                        base_frequency_hz=getattr(opt, "motion_components_frequency_hz", 5.691),
                        fps=fps,
                        num_harmonics=getattr(opt, "motion_components_num_harmonics", 1),
                    )
                )
            elif comp_type == "chebyshev":
                comps.append(
                    ChebyshevComponentTrajectory(
                        degree=getattr(opt, "motion_components_degree", 4),
                    )
                )
            else:
                raise ValueError(f"Unknown motion_components_type: {comp_type}")

        motion_components = MotionComponents(
            num_components=K,
            components=comps,
            ramp_start_iter=getattr(opt, "motion_components_ramp_start_iter", 1000),
            ramp_end_iter=getattr(opt, "motion_components_ramp_end_iter", 3000),
            lambda_res_max=getattr(opt, "motion_components_lambda_res_max", 1.0),
        ).cuda()

        if gaussians._motion_component_logits is None or gaussians._motion_component_logits.shape[-1] != K:
            init_logits = torch.full((gaussians.get_xyz.shape[0], K), -5.0, device="cuda", dtype=torch.float32)
            gaussians._motion_component_logits = nn.Parameter(init_logits.requires_grad_(True))

        params = [
            {"params": list(motion_components.parameters()), "lr": getattr(opt, "motion_components_lr", 0.001), "name": "mc_trajectories"},
            {"params": [gaussians._motion_component_logits], "lr": getattr(opt, "motion_components_responsibilities_lr", 0.01), "name": "mc_logits"},
        ]
        motion_components_optimizer = torch.optim.Adam(params)
        gaussians.motion_components = motion_components
        print(f"[W4] Initialized MotionComponents with K={K} ({comp_type}), ramp=[{motion_components.ramp_start_iter}, {motion_components.ramp_end_iter}], lambda_res={motion_components.lambda_res_max}")

    split_merge_controller = None
    if getattr(opt, "use_split_merge", False) and motion_components is not None:
        from scene.component_split_merge import ComponentSplitMergeController
        split_merge_controller = ComponentSplitMergeController(
            enabled=True,
            start_iter=getattr(opt, "split_merge_start_iter", 1500),
            interval=getattr(opt, "split_merge_interval", 1000),
            max_components=getattr(opt, "split_merge_max_components", 4),
            min_component_support=getattr(opt, "split_merge_min_support", 10.0),
            eval_window=getattr(opt, "split_merge_eval_window", 300),
            min_metric_improvement=getattr(opt, "split_merge_min_improvement", 0.05),
            cooldown_iters=getattr(opt, "split_merge_cooldown_iters", 2000),
        )
        print(f"[W5] Initialized ComponentSplitMergeController (start={split_merge_controller.start_iter}, interval={split_merge_controller.interval}, K_max={split_merge_controller.max_components})")

    if opt.coarse_iterations > 0:
        scene_reconstruction(
            dataset,
            opt,
            hyper,
            pipe,
            testing_iterations,
            saving_iterations,
            checkpoint_iterations,
            checkpoint,
            debug_from,
            gaussians,
            scene,
            "coarse",
            tb_writer,
            opt.coarse_iterations,
            timer,
            camera_offsets_module=camera_offsets_module,
            offsets_optimizer=offsets_optimizer,
            trajectory_loss_fn=trajectory_loss_fn,
            fourier_head=fourier_head,
            fourier_optimizer=fourier_optimizer,
            motion_components=motion_components,
            motion_components_optimizer=motion_components_optimizer,
            split_merge_controller=split_merge_controller,
        )
    else:
        print("coarse_iterations <= 0: skipping coarse stage entirely.")
    scene_reconstruction(
        dataset,
        opt,
        hyper,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        gaussians,
        scene,
        "fine",
        tb_writer,
        opt.iterations,
        timer,
        camera_offsets_module=camera_offsets_module,
        offsets_optimizer=offsets_optimizer,
        trajectory_loss_fn=trajectory_loss_fn,
        fourier_head=fourier_head,
        fourier_optimizer=fourier_optimizer,
    )
    if camera_offsets_module is not None:
        try:
            offsets_file = os.path.join(args.model_path, "learned_offsets.json")
            with open(offsets_file, "w", encoding="utf-8") as f:
                json.dump(camera_offsets_module.get_telemetry_dict(), f, indent=2)
            print(f"[W1] Saved final learned offsets telemetry to {offsets_file}")
        except Exception as e:
            print(f"Warning: failed to save final learned offsets: {e}")
    if motion_components is not None:
        try:
            torch.save(motion_components.state_dict(), os.path.join(args.model_path, "motion_components.pth"))
            with open(os.path.join(args.model_path, "motion_components_state.json"), "w", encoding="utf-8") as f:
                json.dump(motion_components.get_summary(), f, indent=2)
            if gaussians._motion_component_logits is not None:
                torch.save(gaussians._motion_component_logits, os.path.join(args.model_path, "motion_component_logits.pth"))
            print(f"[W4] Saved final MotionComponents to {args.model_path}")
        except Exception as e:
            print(f"Warning: could not save final motion components: {e}")

    if split_merge_controller is not None:
        try:
            with open(os.path.join(args.model_path, "split_merge_state.json"), "w", encoding="utf-8") as f:
                json.dump(split_merge_controller.get_summary(), f, indent=2)
            print(f"[W5] Saved final split/merge ledger to {args.model_path}")
        except Exception as e:
            print(f"Warning: could not save final split_merge_state: {e}")

    if fourier_head is not None:
        try:
            torch.save(fourier_head.state_dict(), os.path.join(args.model_path, "fourier_head.pth"))
            with open(os.path.join(args.model_path, "fourier_head_state.json"), "w", encoding="utf-8") as f:
                json.dump(fourier_head.get_summary(), f, indent=2)
            summary = fourier_head.get_summary()
            print(f"[R9] Saved final FourierRigidHead: freq={summary['frequency_hz']:.3f} Hz, axis={summary['dominant_axis']}, amps={summary['amplitudes_m']}")
        except Exception as e:
            print(f"Warning: failed to save final fourier head: {e}")
    _flush_training_curves(args.model_path)


def prepare_output_and_logger(expname):
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    stage,
    dataset_type,
    camera_offsets_module=None,
):
    global _TRAINING_CURVES
    record_step = (iteration % 20 == 0) or (iteration in testing_iterations)
    entry = None
    if record_step:
        entry = {
            "stage": stage,
            "iteration": iteration,
            "loss": float(loss.item()),
            "l1": float(Ll1.item()),
            "elapsed": float(elapsed),
        }

    if tb_writer:
        tb_writer.add_scalar(
            f"{stage}/train_loss_patches/l1_loss", Ll1.item(), iteration
        )
        tb_writer.add_scalar(
            f"{stage}/train_loss_patchestotal_loss", loss.item(), iteration
        )
        tb_writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        #
        validation_configs = (
            {
                "name": "test",
                "cameras": [
                    scene.getTestCameras()[idx % len(scene.getTestCameras())]
                    for idx in range(10, 5000, 299)
                ],
            },
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(10, 5000, 299)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    eval_custom_time = resolve_camera_time(viewpoint, camera_offsets_module)
                    if eval_custom_time is not None:
                        eval_custom_time = eval_custom_time.detach()
                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            stage=stage,
                            cam_type=dataset_type,
                            custom_time=eval_custom_time,
                            *renderArgs,
                        )["render"],
                        0.0,
                        1.0,
                    )
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(
                            viewpoint.original_image.to("cuda"), 0.0, 1.0
                        )
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                stage
                                + "/"
                                + config["name"]
                                + "_view_{}/render".format(viewpoint.image_name),
                                image[None],
                                global_step=iteration,
                            )
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    stage
                                    + "/"
                                    + config["name"]
                                    + "_view_{}/ground_truth".format(
                                        viewpoint.image_name
                                    ),
                                    gt_image[None],
                                    global_step=iteration,
                                )
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask

                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                if entry is not None:
                    if config["name"] == "test":
                        entry["test_psnr"] = float(psnr_test)
                    elif config["name"] == "train":
                        entry["train_psnr"] = float(psnr_test)
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(
                        stage + "/" + config["name"] + "/loss_viewpoint - l1_loss",
                        l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        stage + "/" + config["name"] + "/loss_viewpoint - psnr",
                        psnr_test,
                        iteration,
                    )

        if tb_writer:
            tb_writer.add_histogram(
                f"{stage}/scene/opacity_histogram",
                scene.gaussians.get_opacity,
                iteration,
            )

            tb_writer.add_scalar(
                f"{stage}/total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
            tb_writer.add_scalar(
                f"{stage}/deformation_rate",
                scene.gaussians._deformation_table.sum()
                / scene.gaussians.get_xyz.shape[0],
                iteration,
            )
            tb_writer.add_histogram(
                f"{stage}/scene/motion_histogram",
                scene.gaussians._deformation_accum.mean(dim=-1) / 100,
                iteration,
                max_bins=500,
            )

        torch.cuda.empty_cache()

    if entry is not None:
        if camera_offsets_module is not None and iteration in testing_iterations:
            try:
                off_tel = camera_offsets_module.get_telemetry_dict()
                entry["camera_offsets"] = off_tel
                print(
                    f"[W1] Offsets telemetry at iter {iteration}: "
                    f"prior_loss={off_tel['prior_loss']:.6f}, "
                    f"delta_offsets_s={off_tel['delta_offsets_s']}"
                )
            except Exception:
                pass
        if scene and hasattr(scene, "gaussians"):
            try:
                entry["points"] = int(scene.gaussians.get_xyz.shape[0])
            except Exception:
                pass
        _TRAINING_CURVES.append(entry)
        if iteration % 500 == 0 or iteration in testing_iterations:
            _flush_training_curves(args.model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[3000, 7000, 14000]
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[14000, 20000, 30_000, 45000, 60000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmcv

        from utils.params_utils import merge_hparams

        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Appended *after* any --configs merge (T11 fix, 2026-07-18 -- see module docstring): must use
    # the true final `iterations`, not the pre-merge argparse default, or the actual last training
    # iteration is never in `save_iterations` and no checkpoint ever gets written.
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
    )

    # All done
    print("\nTraining complete.")

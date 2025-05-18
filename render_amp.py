#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

from utils.sh_utils import eval_sh
import math

import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from motion_amp.renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
import sys
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from gaussian_renderer import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
import time as time_m
import threading
import concurrent.futures


def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def generate_frame_data(views, gaussians, pipeline, background, cam_type):
    
    means3D_list = None
    means2D_list = None
    scales_list = None
    rotations_list = None
    opacity_list = None
    shs_list = None
    colors_list = None
    cov3D_list = None
    rasterizer_settings_list = []

    values_list = [
        means3D_list,
        means2D_list,
        scales_list,
        rotations_list,
        opacity_list,
        shs_list,
        colors_list,
        cov3D_list,
    ]
    for _idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        means3D_final,means2D, scales_final, rotations_final, opacity_final, shs_final,colors_precomp,cov3D_precomp,rasterizer_settings = render(view, gaussians, pipeline, background,cam_type=cam_type)

        values = [means3D_final,means2D, scales_final, rotations_final, opacity_final, shs_final,colors_precomp,cov3D_precomp]

        rasterizer_settings_list.append(rasterizer_settings)

        for i in range(len(values_list)):
            value = values[i]
            value_list = values_list[i]
            if value_list == None:
                values_list[i] = [value]
            else:
                values_list[i].append(value)

    return values_list, rasterizer_settings_list


def amplify_frame_data_phase(values_list, amp_factors, freq_cutoffs):

    for i, zipped in enumerate(zip(values_list, amp_factors,freq_cutoffs)):
        values, a, freq_cutoff = zipped
        if a == -1:
            continue
        if any(list(map(lambda x : x == None, values))):
            continue
        lower_bound, upper_bound = freq_cutoff
        values_unsqueezed = list(map(lambda x : x.unsqueeze(-1),values))
        values_tensor = torch.cat(values_unsqueezed, dim=-1)

        fft : torch.Tensor = torch.fft.rfft(values_tensor,dim=-1,norm="ortho")
        n_frames = len(values)
        frequencies = torch.fft.rfftfreq(n_frames,1/20)
        lower_bound = lower_bound * frequencies.min()
        upper_bound = upper_bound * frequencies.max()

        filtered_frequencies = (frequencies > lower_bound) & (frequencies < upper_bound) 
        fft_time_mean : torch.Tensor = fft.mean(axis=-1).unsqueeze(-1).repeat(*([1]*len(fft.shape)),frequencies.shape[0])
        fft_amp = fft + (fft_time_mean + a * (fft - fft_time_mean)) * filtered_frequencies.cuda() 
        del fft_time_mean, fft
        torch.cuda.empty_cache()
        amped_values = torch.fft.irfft(fft_amp,dim=-1,norm="ortho")
        # amped_values = amped_values.roll(1,-1)
        # amped_values[:,:,0] = values_tensor[:,:,0]
        values_list[i] = list(map(lambda x : x.squeeze(),torch.split(amped_values,1,dim=-1)))
        del  amped_values,filtered_frequencies, frequencies, values_unsqueezed,values_tensor,fft_amp
        torch.cuda.empty_cache()

    return values_list

def amplify_frame_data_phase_abs(values_list, amp_factors, freq_cutoffs):

    for i, zipped in enumerate(zip(values_list, amp_factors,freq_cutoffs)):
        values, a, freq_cutoff = zipped
        if a == -1:
            continue
        if any(list(map(lambda x : x == None, values))):
            continue
        lower_bound, upper_bound = freq_cutoff
        values_unsqueezed = list(map(lambda x : x.unsqueeze(-1),values))
        values_tensor = torch.cat(values_unsqueezed, dim=-1)

        fft : torch.Tensor = torch.fft.rfft(values_tensor,dim=-1,norm="ortho")
        n_frames = len(values)
        frequencies = torch.fft.rfftfreq(n_frames,1/20)
        lower_bound = lower_bound * frequencies.min()
        upper_bound = upper_bound * frequencies.max()

        filtered_frequencies = (frequencies > lower_bound) & (frequencies < upper_bound) 
        fft_amp = fft +  a *  filtered_frequencies.cuda() 
        del  fft
        torch.cuda.empty_cache()
        amped_values = torch.fft.irfft(fft_amp,dim=-1,norm="ortho")
        # amped_values = amped_values.roll(1,-1)
        # amped_values[:,:,0] = values_tensor[:,:,0]
        values_list[i] = list(map(lambda x : x.squeeze(),torch.split(amped_values,1,dim=-1)))
        del  amped_values,filtered_frequencies, frequencies, values_unsqueezed,values_tensor,fft_amp
        torch.cuda.empty_cache()

    return values_list

def amplify_frame_data_eulerian(values_list, amp_factors, freq_cutoffs):

    for i, zipped in enumerate(zip(values_list, amp_factors,freq_cutoffs)):
        values, a, freq_cutoff = zipped
        if a == -1:
            continue
        if any(list(map(lambda x : x == None, values))):
            continue
        lower_bound, upper_bound = freq_cutoff
        values_unsqueezed = list(map(lambda x : x.unsqueeze(-1),values))
        values_tensor = torch.cat(values_unsqueezed, dim=-1)
        values_delta = values_tensor.roll(-1,-1) - values_tensor

        fft_delta = torch.fft.rfft(values_delta,dim=-1,norm="ortho")
        n_frames = len(values)
        frequencies = torch.fft.rfftfreq(n_frames,1/20)
        lower_bound = lower_bound * frequencies.min()
        upper_bound = upper_bound * frequencies.max()

        filtered_frequencies = (frequencies > lower_bound) & (frequencies < upper_bound) 
        fft_delta = fft_delta * filtered_frequencies.cuda() 
    
        values_delta_filtered = torch.fft.irfft(fft_delta,dim=-1,norm="ortho")
        amped_values = values_tensor + a * values_delta_filtered
        amped_values = amped_values.roll(1,-1)
        amped_values[:,:,0] = values_tensor[:,:,0]
        values_list[i] = list(map(lambda x : x.squeeze(),torch.split(amped_values,1,dim=-1)))
        del values_delta, amped_values,fft_delta, filtered_frequencies, frequencies, values_delta_filtered, values_unsqueezed,values_tensor
        torch.cuda.empty_cache()

    return values_list

def amplify_frame_data_eulerian_abs(values_list, amp_factors, freq_cutoffs):

    for i, zipped in enumerate(zip(values_list, amp_factors,freq_cutoffs)):
        values, a, freq_cutoff = zipped
        if a == -1:
            continue
        if any(list(map(lambda x : x == None, values))):
            continue
        lower_bound, upper_bound = freq_cutoff
        values_unsqueezed = list(map(lambda x : x.unsqueeze(-1),values))
        n_frames = len(values)
        values_tensor = torch.cat(values_unsqueezed, dim=-1)
        values_delta = values_tensor - values_tensor[:,:,0].unsqueeze(-1).repeat(*([1]*len(values_tensor.shape)),values_tensor.shape[-1])

        fft_delta = torch.fft.rfft(values_delta,dim=-1,norm="ortho")
        frequencies = torch.fft.rfftfreq(n_frames,1/20)
        lower_bound = lower_bound * frequencies.min()
        upper_bound = upper_bound * frequencies.max()

        filtered_frequencies = (frequencies > lower_bound) & (frequencies < upper_bound) 
        fft_delta = fft_delta * filtered_frequencies.cuda() 
    
        values_delta_filtered = torch.fft.irfft(fft_delta,dim=-1,norm="ortho")
        amped_values = values_tensor + a * values_delta_filtered
        amped_values = amped_values.roll(1,-1)
        amped_values[:,:,0] = values_tensor[:,:,0]
        values_list[i] = list(map(lambda x : x.squeeze(),torch.split(amped_values,1,dim=-1)))
        del values_delta, amped_values,fft_delta, filtered_frequencies, frequencies, values_delta_filtered, values_unsqueezed,values_tensor
        torch.cuda.empty_cache()

    return values_list

def render_data(values_list, rasterizer_settings_list, views, name, cam_type):
    render_images = []
    gt_list = []
    render_list = []
    for i in range(len(rasterizer_settings_list)):

        rasterizer_settings = rasterizer_settings_list[i]
        rasterizer = GaussianRasterizer(raster_settings=rasterizer_settings)

        rendered_image, radii, depth = rasterizer(
        means3D = values_list[0][i],
        means2D = values_list[1][i],
        shs = values_list[5][i],
        colors_precomp = values_list[6][i],
        opacities = values_list[4][i],
        scales = values_list[2][i],
        rotations = values_list[3][i],
        cov3D_precomp = values_list[7][i])

        
        rendering = rendered_image
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)

        view = views[i]
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    return render_images, gt_list, render_list
            
def render_set_amp(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, amp_factors, freq_cutoffs, method = "eulerian"):
    time1 = time_m.time()

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    print("point nums:",gaussians._xyz.shape[0])

    values_list, rasterizer_settings_list = generate_frame_data(views,gaussians,pipeline,background,cam_type)
    try:
        if method == "eulerian":
            values_list = amplify_frame_data_eulerian(values_list,amp_factors,freq_cutoffs)
        else:
            values_list = amplify_frame_data_phase(values_list,amp_factors,freq_cutoffs)
    finally:
        torch.cuda.empty_cache()
    render_images, gt_list, render_list = render_data(values_list,rasterizer_settings_list,views,name,cam_type)

    time2=time_m.time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)
    imageio.mimwrite(os.path.join(model_path, name, f"amp", f'video_rgb_amp.mp4'), render_images, fps=20)
    del values_list, render_list, gt_list
    torch.cuda.empty_cache()

class AmpConfig():
    def __init__(self, model, hyperparam, iteration, pipeline, amp_factors, freq_list):
        self.model = model
        self.hyperparam = hyperparam
        self.iteration = iteration
        self.pipeline = pipeline
        self.amp_factors = amp_factors
        self.freq_list = freq_list
        with torch.no_grad():
            self.gaussians = GaussianModel(self.model.sh_degree, hyperparam)
            self.scene : Scene = Scene(self.model, self.gaussians, load_iteration=iteration, shuffle=False)
            self.cam_type = self.scene.dataset_type
            self.bg_color = [1,1,1] if self.model.white_background else [0, 0, 0]
            self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            generate_frame_data(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        # if not skip_test:
        #     render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
        if not skip_video:
            render_set_amp(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type,[5],[(0,100)])

def load_data(amp_config : AmpConfig):
    values_list, rasterizer_settings_list = generate_frame_data(amp_config.scene.getVideoCameras(),amp_config.gaussians,pipeline,amp_config.background,amp_config.cam_type)
    return values_list,rasterizer_settings_list

def render_app(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, amp_factors : list, freq_cutoffs : list, method="eulerian"):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        render_set_amp(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type,amp_factors,freq_cutoffs, method=method)


def get_combined_args(parser : ArgumentParser, model_path : str,config_path : str):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    # WHY WOULD YOU ADD ONLY ALLOW TYPE ERRORS ????
    except:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    merged_dict["model_path"] = model_path
    merged_dict["configs"] = config_path


    return Namespace(**merged_dict)



def load_config(model_path, config_path, amp_factors, freq_list):
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser, model_path, config_path)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    # render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
    return AmpConfig(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), amp_factors, freq_list)


def main(model_path, config_path, amp_factors, freq_list, method):
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser, model_path, config_path)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    # render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
    render_app(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), amp_factors, freq_list, method)
    # print(args)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
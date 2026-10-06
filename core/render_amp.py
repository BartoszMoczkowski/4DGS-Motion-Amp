import gc
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
import sys
import time as time_m
import concurrent.futures


from arguments import ModelParams, PipelineParams, ModelHiddenParams
from gaussian_renderer import GaussianModel
from motion_amp.renderer import render_mod
from diff_gaussian_rasterization import GaussianRasterizer

# As this program relies on 4DGS, a lot of function are modified versions
# from that work. https://github.com/hustvl/4DGaussians

def generate_frame_data(views, gaussians, pipeline, background, cam_type, low_vram_mode=False):
    """
    Generate the propoerties of Gaussians for each timestamp, given the camera information and model.
    
    :param views: The list of camera information.
    :param gaussians: The model to be rendered.
    :param pipeline: The rendering pipeline.    
    :param background: The background color for the scene.
    :param cam_type: The type of camera used in
    :param low_vram_mode: If True, the data is moved out of the GPU aggresivly. Default is False.
    This saves VRAM at the cost of performance.

    Based on the rendering pipeline from 4DGS.    
    """

    # Create lists to store the parameters
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

    # 
    for _idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        # For each view (timestamp) we get the Gaussians  parameters for the scene.
        # In the actual software additional parmeters are present but they are not relevant to the amplification process.
        # They are kept since the rendering pipeline expects them.
        gaussian_parameters = render_mod(view, gaussians, pipeline, background,cam_type=cam_type)

        means3D_final = gaussian_parameters[0]
        means2D = gaussian_parameters[1]
        scales_final = gaussian_parameters[2]
        rotations_final = gaussian_parameters[3]
        opacity_final = gaussian_parameters[4]
        shs_final = gaussian_parameters[5]
        colors_precomp = gaussian_parameters[6]
        cov3D_precomp = gaussian_parameters[7]
        rasterizer_settings = gaussian_parameters[8]

        # In low VRAM mode we move all the data out of the GPU
        if low_vram_mode:
            values = [means3D_final.cpu(),means2D.cpu(), scales_final.cpu(), rotations_final.cpu(), opacity_final.cpu(), shs_final.cpu(),colors_precomp,cov3D_precomp]
        else:
            values = [means3D_final,means2D, scales_final, rotations_final, opacity_final, shs_final,colors_precomp,cov3D_precomp]

        # we store all the data in the lists

        rasterizer_settings_list.append(rasterizer_settings)

        for i in range(len(values_list)):
            value = values[i]
            value_list = values_list[i]
            if value_list is None:
                values_list[i] = [value]
            else:
                values_list[i].append(value)

        # We delete the temporary variables to free up memory
        del values,means3D_final,means2D, scales_final, rotations_final, opacity_final, shs_final,colors_precomp,cov3D_precomp
    # The data is returned for later processing
    return values_list, rasterizer_settings_list

# Number of parameter slots produced by generate_frame_data():
# [means3D, means2D, scales, rotations, opacity, shs, colors_precomp, cov3D_precomp]
N_AMP_PARAMS = 8

# Parameter slots living on a constrained domain, repaired after amplification (bug 11)
_IDX_SCALES = 2
_IDX_ROTATIONS = 3
_IDX_OPACITY = 4


def _sanitize_amplified_values(amped, param_idx):
    """Clamp/normalize an amplified parameter tensor back into its valid domain (bug 11).

    Amplification happens AFTER the activations in motion_amp/renderer.py, so amplified
    values can leave the parameter domain: opacity must stay in [0, 1], scales must stay
    positive, and rotation quaternions must be unit-norm. The CUDA rasterizer fork does
    NOT normalize quaternions (submodules/depth-diff-gaussian-rasterization/
    cuda_rasterizer/forward.cu:127), so re-normalization must happen here.
    """
    if param_idx == _IDX_OPACITY:
        return torch.clamp(amped, 0.0, 1.0)
    if param_idx == _IDX_SCALES:
        return torch.clamp_min(amped, 1e-8)
    if param_idx == _IDX_ROTATIONS:
        # quaternions are [N, 4, ..., T]; normalize along the feature axis (dim=-2)
        return amped / amped.norm(dim=-2, keepdim=True).clamp_min(1e-12)
    return amped


def validate_amp_args(amp_factors, freq_cutoffs):
    """Validate and broadcast amplification arguments (bug 14).

    ``amp_factors`` must have exactly ``N_AMP_PARAMS`` entries; -1 is the skip sentinel
    and the only allowed negative value. ``freq_cutoffs`` must have either 1 entry
    (broadcast to all parameter slots) or ``N_AMP_PARAMS`` entries. Returns the validated
    (and possibly broadcast) lists. Raises ValueError on any mismatch instead of silently
    truncating via zip().
    """
    amp_factors = list(amp_factors)
    freq_cutoffs = list(freq_cutoffs)
    if len(amp_factors) != N_AMP_PARAMS:
        raise ValueError(
            f"amp_factors must have exactly {N_AMP_PARAMS} entries (one per parameter slot), "
            f"got {len(amp_factors)}.")
    for a in amp_factors:
        if a < 0 and a != -1:
            raise ValueError(
                f"Negative amplification factor {a} is not allowed: -1 is the skip sentinel "
                "and the only permitted negative value.")
    if len(freq_cutoffs) == 1:
        freq_cutoffs = freq_cutoffs * N_AMP_PARAMS
    elif len(freq_cutoffs) != N_AMP_PARAMS:
        raise ValueError(
            f"freq_cutoffs must have either 1 entry (broadcast to all {N_AMP_PARAMS} parameter "
            f"slots) or {N_AMP_PARAMS} entries, got {len(freq_cutoffs)}.")
    return amp_factors, freq_cutoffs


def build_freq_cutoffs(freq_low, freq_high):
    """Pair --freq_low/--freq_high into per-parameter cutoff tuples, with validation (bug 14).

    Both lists must have equal length, and that length must be either 1 (broadcast to all
    parameter slots) or N_AMP_PARAMS. Returns a list of (lower, upper) tuples of length
    N_AMP_PARAMS.
    """
    if freq_low is None or freq_high is None:
        raise ValueError("Both --freq_low and --freq_high must be provided.")
    if len(freq_low) != len(freq_high):
        raise ValueError(
            f"--freq_low and --freq_high must have equal length, got {len(freq_low)} and "
            f"{len(freq_high)}.")
    if len(freq_low) == 1:
        freq_low = list(freq_low) * N_AMP_PARAMS
        freq_high = list(freq_high) * N_AMP_PARAMS
    elif len(freq_low) != N_AMP_PARAMS:
        raise ValueError(
            f"--freq_low/--freq_high must have length 1 (broadcast) or {N_AMP_PARAMS}, "
            f"got {len(freq_low)}.")
    return list(zip(freq_low, freq_high))


def amplify_frame_data_eulerian(values_list, amp_factors, freq_cutoffs, low_vram_mode=False):
    """Implementation of the eulerian amplification algorithm

    SEMANTICS CHANGED (2026-10-05 correctness review, bugs 1, 2, 7, 13): this method now
    performs textbook Eulerian *displacement* amplification — the deviation of each parameter
    from its per-Gaussian temporal mean is band-pass filtered and scaled by ``a``:
    ``out[t] = mean + a * filter(v[t] - mean)``. The previous implementation amplified the
    first temporal difference (velocity), which for slow motion gave an effective displacement
    gain far below ``a`` (bug 2), and its rolled difference contaminated the FFT with a
    wrap-around sample (bug 1). The mean-anchored formulation removes both issues, makes
    ``a=1`` an exact identity, and needs no frame-0 reset (bug 7). All four ``eulerian*``
    variants now share these semantics; the ``*_mod`` variants only chunk Gaussians to save
    memory, and the ``*_abs*`` names are kept for CLI backward compatibility.
    
    Args:
        values_list (list): List of lists of parameters containing the gaussian data to be amplified.
        amp_factors (list): List of amplification factors for parameter, a=-1 means to do nothing.
        freq_cutoffs (list): List of tuples containing the relative (0.0 to 1.0) lower and upper bounds for each data parameter. Bounds are inclusive on both ends (bug 12).
        low_vram_mode (bool): Whether to use low VRAM mode or not.
    
    values lists should be taken as the output from generate_frame_data(). amp_factors and freq_cutoffs
    are validated/broadcast by validate_amp_args().
    """

    amp_factors, freq_cutoffs = validate_amp_args(amp_factors, freq_cutoffs)

    # Loop over each parameter
    for i, zipped in enumerate(zip(values_list, amp_factors,freq_cutoffs)):
        values, a, freq_cutoff = zipped

        # Check if the given parameter should be amplified
        if a == -1:
            continue
        if any(list(map(lambda x : x == None, values))):
            continue
       
        lower_bound, upper_bound = freq_cutoff # extract the frequency bounds for the current parameter
        
        values_unsqueezed = list(map(lambda x : x.unsqueeze(-1),values))
        values_tensor = torch.cat(values_unsqueezed, dim=-1) # create the combined tensor
        
        # move the tensor to the GPU if low VRAM mode is on as it might not be loaded
        if low_vram_mode and torch.cuda.is_available():
            values_tensor = values_tensor.cuda() 
        
        # deviation from the per-Gaussian temporal mean (the Eulerian reference frame, bugs 2/13)
        temporal_mean = values_tensor.mean(dim=-1, keepdim=True)
        values_delta = values_tensor - temporal_mean

        fft_delta = torch.fft.rfft(values_delta,dim=-1,norm="ortho") # calculate the real valued FFT for the time dimension

        # caluculate the frequencies present, since the bound are relative we do not scale the frequencies by 1/fps 
        n_frames = len(values)
        frequencies = torch.fft.rfftfreq(n_frames)
        lower_bound = lower_bound * frequencies.max()
        upper_bound = upper_bound * frequencies.max()

        # filter fequencies based on bounds (inclusive on both ends, bug 12) and use them as a mask for the difference tensor
        filtered_frequencies = (frequencies >= lower_bound) & (frequencies <= upper_bound) 
        fft_delta_filtered = fft_delta * filtered_frequencies.to(fft_delta.device) 

        # revert the difference to time space; n=n_frames so odd frame counts don't crash (bug 10)
        values_delta_filtered = torch.fft.irfft(fft_delta_filtered,dim=-1,norm="ortho",n=n_frames)
        
        del fft_delta_filtered, lower_bound, upper_bound 

        # calculuate the amplified value: mean + a * filtered deviation
        amped_values = temporal_mean + a * values_delta_filtered
        # repair the parameter domain after amplification (bug 11)
        amped_values = _sanitize_amplified_values(amped_values, i)

        # if low VRAM mode is on, move the data back out of the GPU
        if low_vram_mode:
            amped_values = amped_values.cpu()

        # split the data back into individual elements and save it to the list
        values_list[i] = list(map(lambda x : x.squeeze(),torch.split(amped_values,1,dim=-1)))
        del values_delta,values, amped_values,fft_delta, filtered_frequencies, frequencies, values_delta_filtered, values_unsqueezed,values_tensor, temporal_mean
        torch.cuda.empty_cache() # clear the GPU cache

    return values_list

def amplify_frame_data_eulerian_mod(values_list, amp_factors, freq_cutoffs,low_vram_mode=False):
    """Implementation of the eulerian segmented (chunked) amplification algorithm

    Same mean-anchored displacement-amplification semantics as amplify_frame_data_eulerian()
    (see its docstring — semantics changed 2026-10-05 per the correctness review, bugs 1, 2,
    7, 13); only the per-Gaussian chunking for memory savings differs.
    
    Args:
        values_list (list): List of lists of parameters containing the gaussian data to be amplified.
        amp_factors (list): List of amplification factors for parameter, a=-1 means to do nothing.
        freq_cutoffs (list): List of tuples containing the relative (0.0 to 1.0) lower and upper bounds for each data parameter. Bounds are inclusive on both ends (bug 12).
        low_vram_mode (bool): Whether to use low VRAM mode or not.
    
    values lists should be taken as the output from generate_frame_data(). amp_factors and freq_cutoffs
    are validated/broadcast by validate_amp_args().
    """

    amp_factors, freq_cutoffs = validate_amp_args(amp_factors, freq_cutoffs)

    # Loop over each parameter
    for i, zipped in enumerate(zip(values_list, amp_factors,freq_cutoffs)):
        values, a, freq_cutoff = zipped
       
        # Check if the given parameter should be amplified
        if a == -1:
            continue
        if any(list(map(lambda x : x == None, values))):
            continue
        
        lower_bound, upper_bound = freq_cutoff # extract the frequency bounds for the current parameter
       
        values_unsqueezed = list(map(lambda x : x.unsqueeze(-1),values))
        values_tensor_full = torch.cat(values_unsqueezed, dim=-1) # create the combined tensor
      
        temp = [] # temporary list to store results of the splitting
        # caluculate the frequencies present, since the bound are relative we do not scale the frequencies by 1/fps 
        n_frames = len(values)
        frequencies = torch.fft.rfftfreq(n_frames)
        lower_bound = lower_bound * frequencies.max()
        upper_bound = upper_bound * frequencies.max()

        # filter fequencies based on bounds (inclusive on both ends, bug 12)
        filtered_frequencies = (frequencies >= lower_bound) & (frequencies <= upper_bound)
        del values_unsqueezed

        # split the tensor into chunks of 1024 Gaussians (chunking the Gaussian axis is exact:
        # the FFT runs along the time axis and the temporal mean is per Gaussian)
        for values_tensor in values_tensor_full.split(1024,dim=0):

            # move the tensor to the GPU if low VRAM mode is on as it might not be loaded
            if low_vram_mode and torch.cuda.is_available():
                values_tensor = values_tensor.cuda()

            # deviation from the per-Gaussian temporal mean (the Eulerian reference frame, bugs 2/13)
            temporal_mean = values_tensor.mean(dim=-1, keepdim=True)
            values_delta = values_tensor - temporal_mean

            fft_delta = torch.fft.rfft(values_delta,dim=-1,norm="ortho") # calculate the real valued FFT for the time dimension

            fft_delta_filtered = fft_delta * filtered_frequencies.to(fft_delta.device)

            # revert the difference to time space; n=n_frames so odd frame counts don't crash (bug 10)
            values_delta_filtered = torch.fft.irfft(fft_delta_filtered,dim=-1,norm="ortho",n=n_frames)

            # calculuate the amplified value: mean + a * filtered deviation
            amped_values = temporal_mean + a * values_delta_filtered

            # if low VRAM mode is on, move the data back out of the GPU
            if low_vram_mode:
                amped_values = amped_values.cpu()

            # store the partial result
            temp.append(amped_values)
            del fft_delta_filtered,fft_delta, values_delta_filtered, values_tensor, temporal_mean, values_delta, amped_values
            torch.cuda.empty_cache() # clear the GPU cache

        catted = torch.cat(temp, dim=0) # combine the partial result

        # repair the parameter domain after amplification (bug 11)
        catted = _sanitize_amplified_values(catted, i)

        # split the data back into individual elements and save it to the list
        values_list[i] = list(map(lambda x : x.squeeze(),torch.split(catted,1,dim=-1)))
        del values, catted, frequencies, filtered_frequencies, values_tensor_full, temp
        torch.cuda.empty_cache() # clear the GPU cache
    return values_list

def amplify_frame_data_eulerian_abs(values_list, amp_factors, freq_cutoffs,low_vram_mode=False):
    """Implementation of the eulerian absolute amplification algorithm

    SEMANTICS CHANGED (2026-10-05 correctness review, bugs 2, 13): the reference frame is now
    the per-Gaussian temporal mean (textbook Eulerian reference), not the LAST frame that
    ``torch.narrow(values_tensor, -1, -1, 1)`` used to select despite the "initial element"
    comment. With that change this method is mathematically identical to
    amplify_frame_data_eulerian() — see its docstring; the name is kept for CLI backward
    compatibility. A side benefit: ``a=1`` full-band now reproduces the input exactly.
    
    Args:
        values_list (list): List of lists of parameters containing the gaussian data to be amplified.
        amp_factors (list): List of amplification factors for parameter, a=-1 means to do nothing.
        freq_cutoffs (list): List of tuples containing the relative (0.0 to 1.0) lower and upper bounds for each data parameter. Bounds are inclusive on both ends (bug 12).
        low_vram_mode (bool): Whether to use low VRAM mode or not.
    
    values lists should be taken as the output from generate_frame_data(). amp_factors and freq_cutoffs
    are validated/broadcast by validate_amp_args().
    """

    amp_factors, freq_cutoffs = validate_amp_args(amp_factors, freq_cutoffs)

    # Loop over each parameter
    for i, zipped in enumerate(zip(values_list, amp_factors,freq_cutoffs)):
        values, a, freq_cutoff = zipped

        # Check if the given parameter should be amplified
        if a == -1:
            continue
        if any(list(map(lambda x : x == None, values))):
            continue

        lower_bound, upper_bound = freq_cutoff # extract the frequency bounds for the current parameter
      
        values_unsqueezed = list(map(lambda x : x.unsqueeze(-1),values))
        values_tensor = torch.cat(values_unsqueezed, dim=-1) # create the combined tensor
        
        # move the tensor to the GPU if low VRAM mode is on as it might not be loaded
        if low_vram_mode and torch.cuda.is_available():
            values_tensor = values_tensor.cuda()
        
        # deviation from the per-Gaussian temporal mean (the Eulerian reference frame, bugs 2/13)
        temporal_mean = values_tensor.mean(dim=-1, keepdim=True)
        values_delta = values_tensor - temporal_mean

        # calculate the real valued FFT for the time dimension
        fft_delta = torch.fft.rfft(values_delta,dim=-1,norm="ortho")
        
        # caluculate the frequencies present, since the bound are relative we do not scale the frequencies by 1/fps 
        n_frames = len(values)
        frequencies = torch.fft.rfftfreq(n_frames)
        lower_bound = lower_bound * frequencies.max()
        upper_bound = upper_bound * frequencies.max()

        # filter fequencies based on bounds (inclusive on both ends, bug 12) and use them as a mask for the difference tensor
        filtered_frequencies = (frequencies >= lower_bound) & (frequencies <= upper_bound) 
        fft_delta = fft_delta * filtered_frequencies.to(fft_delta.device) 

        # revert the difference to time space; n=n_frames so odd frame counts don't crash (bug 10)
        values_delta_filtered = torch.fft.irfft(fft_delta,dim=-1,norm="ortho",n=n_frames)

        del lower_bound, upper_bound

        # calculuate the amplified value: mean + a * filtered deviation
        amped_values = temporal_mean + a * values_delta_filtered
        # repair the parameter domain after amplification (bug 11)
        amped_values = _sanitize_amplified_values(amped_values, i)
  
        # if low VRAM mode is on, move the data back out of the GPU
        if low_vram_mode:
            amped_values = amped_values.cpu()
     
        # split the data back into individual elements and save it to the list
        values_list[i] = list(map(lambda x : x.squeeze(),torch.split(amped_values,1,dim=-1)))
        del values_delta, amped_values,fft_delta, filtered_frequencies, frequencies, values_delta_filtered, values_unsqueezed,values_tensor, temporal_mean
        torch.cuda.empty_cache() # clear the GPU cache

    return values_list

def amplify_frame_data_eulerian_abs_mod(values_list, amp_factors, freq_cutoffs,low_vram_mode=False):
    """Implementation of the eulerian absolute segmented (chunked) amplification algorithm

    Same mean-anchored displacement-amplification semantics as amplify_frame_data_eulerian()
    (see its docstring — semantics changed 2026-10-05 per the correctness review, bugs 2, 13);
    only the per-Gaussian chunking for memory savings differs. The name is kept for CLI
    backward compatibility.
    
    Args:
        values_list (list): List of lists of parameters containing the gaussian data to be amplified.
        amp_factors (list): List of amplification factors for parameter, a=-1 means to do nothing.
        freq_cutoffs (list): List of tuples containing the relative (0.0 to 1.0) lower and upper bounds for each data parameter. Bounds are inclusive on both ends (bug 12).
        low_vram_mode (bool): Whether to use low VRAM mode or not.
    
    values lists should be taken as the output from generate_frame_data(). amp_factors and freq_cutoffs
    are validated/broadcast by validate_amp_args().
    """

    amp_factors, freq_cutoffs = validate_amp_args(amp_factors, freq_cutoffs)

    # Loop over each parameter
    for i, zipped in enumerate(zip(values_list, amp_factors,freq_cutoffs)):
        values, a, freq_cutoff = zipped
        
        # Check if the given parameter should be amplified
        if a == -1:
            continue
        if any(list(map(lambda x : x == None, values))):
            continue

        lower_bound, upper_bound = freq_cutoff # extract the frequency bounds for the current parameter
        
        values_unsqueezed = list(map(lambda x : x.unsqueeze(-1),values))
        values_tensor_full = torch.cat(values_unsqueezed, dim=-1) # create the combined tensor
        
        # move the tensor to the GPU if low VRAM mode is off (kept from the original implementation)
        if not low_vram_mode and torch.cuda.is_available():
            values_tensor_full = values_tensor_full.cuda()

        # deviation from the per-Gaussian temporal mean (the Eulerian reference frame, bugs 2/13)
        temporal_mean_full = values_tensor_full.mean(dim=-1, keepdim=True)
        values_delta_full = values_tensor_full - temporal_mean_full
        temp = [] # temporary list to store results of the splitting
     
        # caluculate the frequencies present, since the bound are relative we do not scale the frequencies by 1/fps 
        n_frames = len(values)
        frequencies = torch.fft.rfftfreq(n_frames)
        lower_bound = lower_bound * frequencies.max()
        upper_bound = upper_bound * frequencies.max()

        # filter fequencies based on bounds (inclusive on both ends, bug 12)
        filtered_frequencies = (frequencies >= lower_bound) & (frequencies <= upper_bound)
        
        del values_unsqueezed

        # split the tensor into chunks of 1024 Gaussians (chunking the Gaussian axis is exact:
        # the FFT runs along the time axis and the temporal mean is per Gaussian)
        for values_delta, temporal_mean in zip(values_delta_full.split(1024,dim=0), temporal_mean_full.split(1024,dim=0)):
            
            # move the tensor to the GPU if low VRAM mode is on as it might not be loaded
            if low_vram_mode and torch.cuda.is_available():
                values_delta = values_delta.cuda()
                temporal_mean = temporal_mean.cuda()

            fft_delta = torch.fft.rfft(values_delta,dim=-1,norm="ortho") # calculate the real valued FFT for the time dimension

            fft_delta_filtered = fft_delta * filtered_frequencies.to(fft_delta.device)  #mask the difference tensor
            # revert the difference to time space; n=n_frames so odd frame counts don't crash (bug 10)
            values_delta_filtered = torch.fft.irfft(fft_delta_filtered,dim=-1,norm="ortho",n=n_frames)

            # calculuate the amplified value: mean + a * filtered deviation
            amped_values = temporal_mean + a * values_delta_filtered

            # if low VRAM mode is on, move the data back out of the GPU
            if low_vram_mode:
                amped_values = amped_values.cpu()

            # store the partial result
            temp.append(amped_values)
            del fft_delta_filtered,fft_delta, values_delta_filtered
            torch.cuda.empty_cache() # clear the GPU cache
        # combine the partial result and repair the parameter domain after amplification (bug 11)
        catted = _sanitize_amplified_values(torch.cat(temp, dim=0), i)

        # split the data back into individual elements and save it to the list
        values_list[i] = list(map(lambda x : x.squeeze(),torch.split(catted,1,dim=-1)))
        del values_delta,values, amped_values,values_tensor_full,values_delta_full, temporal_mean_full, frequencies, filtered_frequencies, temp, catted
        torch.cuda.empty_cache() # clear the GPU cache
    return values_list

def render_data(values_list, rasterizer_settings_list, views, name, cam_type, low_vram_mode=False, frozen_cam=False):
    """
    Render the scene base on a list of the Gaussian parameters 

    Args:
        values_list (list): list of the Gaussian parameters 
        rasterizer_settings_list (list): list of the Gaussian parameters 
        views (list): list of the views 
        name (str): the name of the scene 
        cam_type (str): the type of camera 
        low_vram_mode(bool) : is true run on low VRAM mode
        frozen_cam(bool) : is true run on freeze the camera in place

    Adapted from the pipeline from 4DGS
    """
    # create lists for storing outputs
    render_images = []
    gt_list = []
    render_list = []

    # loop over each element (each representing a timestamp)
    for i in range(len(rasterizer_settings_list)):

        # if the camera is frozen always use the setting for first frame
        if frozen_cam:
            rasterizer_settings = rasterizer_settings_list[0]
        else:
            rasterizer_settings = rasterizer_settings_list[i]
        # Initialize the rasterizer from 3DGS
        rasterizer = GaussianRasterizer(raster_settings=rasterizer_settings)

        # Extract the data and move it to the GPU is needed
        if low_vram_mode:
            means3D = values_list[0].pop(0).cuda()
            means2D = values_list[1].pop(0).cuda()
            shs = values_list[5].pop(0).cuda()
            opacities = values_list[4].pop(0).cuda()
            scales = values_list[2].pop(0).cuda()
            rotations = values_list[3].pop(0).cuda()
            colors_precomp = values_list[6].pop(0)
            cov3D_precomp = values_list[7].pop(0)
        else:
            means3D = values_list[0][i]
            means2D = values_list[1][i]
            shs = values_list[5][i]
            opacities = values_list[4][i]
            scales = values_list[2][i]
            rotations = values_list[3][i]
            colors_precomp = values_list[6].pop(0)
            cov3D_precomp = values_list[7].pop(0)

        # render the image
        rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

        # clear out unneeded data
        if low_vram_mode:
            means3D.cpu()
            means2D.cpu()
            shs.cpu()
            opacities.cpu()
            scales.cpu()
            rotations.cpu()

        del means3D, means2D, rotations, scales, shs, opacities, colors_precomp, cov3D_precomp, rasterizer, radii, depth

        rendering = rendered_image.cpu()
        
        # transform the image and save it to the list
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

        del rendering, rendered_image

        view = views[i]
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    return render_images, gt_list, render_list


class AmpConfig():
    # Helper class used when interfacing with the program from the UI. Stores the data about the scene
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

            
def multithread_write(image_list, path):
    # Taken from 4DGS to run the render pipeline correctly
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
    for index, future in enumerate(tasks):
        # bug 19: `status` was a Future, never == False, so failed writes were silently
        # swallowed; call .result() so failures are observed, retry once, then raise
        count, ok = future.result()
        if not ok:
            count, ok = write_image(image_list[count], count, path)
            if not ok:
                raise RuntimeError(f"Failed to write frame {count} to {path}")
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8) # Taken from 4DGS to run the render pipeline correctly


def render_set_amp(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, amp_factors, freq_cutoffs, method = "eulerian", low_vram_mode=False, path="render.mp4",fps=20, frozen_cam=False):
    """
    Modified rendering function from 4DGS with amplification.
    Args:
        model_path(str): path to the model folder,
        name(str): name of the scene,
        iteration(int): interation number,
        views: camera positions in time,
        gaussians: the Gaussian scene,
        pipeline: the rendering pipeline,
        background: the background color,
        cam_type(str): camera type 
        amp_factors(list): list of amplification factors (exactly 8 entries, -1 skips a parameter),
        freq_cutoffs(list): list of tuples of frequency bounds (lower,upper); 1 entry broadcasts to all 8 parameters,
        method(str): method to use for amplification [eulerian, eulerian_abs, eulerian_mod, eulerian_abs_mod].
            All methods perform mean-anchored displacement amplification (semantics changed
            2026-10-05; see amplify_frame_data_eulerian's docstring),
        low_vram_mode(bool): if True, use low VRAM mode,
        path(str): path for the video output,
        fps(int): frame rate of the output video,
        frozen_cam(bool): if true freeze the camera
    """
    time1 = time_m.time()
    # Prepere folders
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    print("point nums:",gaussians._xyz.shape[0])

    # Validate the amplification arguments before doing any work (bug 14)
    amp_factors, freq_cutoffs = validate_amp_args(amp_factors, freq_cutoffs)

    # Step 1: extract the gaussian data (bug 17: forward low_vram_mode so the
    # frame data is kept off the GPU during extraction in low VRAM mode)
    values_list, rasterizer_settings_list = generate_frame_data(views,gaussians,pipeline,background,cam_type, low_vram_mode)
    
    # Step 2: amplify the motion
    try:
        if method == "eulerian":
            values_list = amplify_frame_data_eulerian(values_list,amp_factors,freq_cutoffs, low_vram_mode)
        elif method == "eulerian_abs":
            values_list = amplify_frame_data_eulerian_abs(values_list,amp_factors,freq_cutoffs, low_vram_mode)
        elif method == "eulerian_mod":
            values_list = amplify_frame_data_eulerian_mod(values_list,amp_factors,freq_cutoffs, low_vram_mode)
        elif method == "eulerian_abs_mod":
            values_list = amplify_frame_data_eulerian_abs_mod(values_list,amp_factors,freq_cutoffs, low_vram_mode)
        else:
            # bug 18: an unknown method used to silently render an unamplified video
            raise ValueError(f"Unknown amplification method '{method}'. Valid methods: "
                             "eulerian, eulerian_abs, eulerian_mod, eulerian_abs_mod")
    finally:
        torch.cuda.empty_cache()

    # Step 3: use the amplified values to render
    render_images, gt_list, render_list = render_data(values_list,rasterizer_settings_list,views,name,cam_type,low_vram_mode,frozen_cam)


    # Write the output images and video and clear out memory
    time2=time_m.time()
    print("FPS:",len(views)/(time2-time1)) # bug 21: was len(views)-1

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)
    imageio.mimwrite(os.path.join(model_path, name, path), render_images, fps=fps)
    del values_list, render_list, gt_list
    torch.cuda.empty_cache()

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, amp_factors : list, freq_cutoffs : list, method = "eulerian", low_vram_mode=False, path="render.mp4",fps=20, frozen_cam=False):
    # helper function to load the model before running the pipeline 
    # Based on the same function from 4DGS

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set_amp(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type,amp_factors,freq_cutoffs, method=method, low_vram_mode=low_vram_mode, path=path,fps=fps, frozen_cam=frozen_cam)


def get_combined_args(parser : ArgumentParser, model_path = None,config_path = None):
    # helper funtion for loading program arguments, taken from 4DGS modified to accept the model and 
    # config path as function parameters 

    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(model_path if model_path is not None else args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    if model_path is not None:
        merged_dict["model_path"] = model_path
    if config_path is not None:
        merged_dict["configs"] = config_path


    return Namespace(**merged_dict)



def load_config(model_path, config_path, amp_factors, freq_list):
    # Helper function used to load the scene without command line arguments.
    # Modified from 4DGS
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
        # bug 8: the pinned mmcv 2.2.0 moved Config to mmengine; try mmengine first
        # (same approach as the __main__ path below)
        try:
            from mmengine.config import Config
        except ImportError:
            from mmcv import Config
        from utils.params_utils import merge_hparams
        config = Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    safe_state(args.quiet)
    return AmpConfig(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), amp_factors, freq_list)

if __name__ == "__main__":
    # Run the rendering based on commandline arguments
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
    parser.add_argument("--amp_factors", type=int, nargs="+",
                        help="Exactly 8 amplification factors, one per parameter slot "
                             "[means3D, means2D, scales, rotations, opacity, SHs, color, cov3D]; "
                             "-1 skips a parameter and is the only allowed negative value")
    parser.add_argument("--freq_high", type=float, nargs="+",
                        help="Upper relative frequency bound(s): 1 value broadcasts to all 8 parameter slots, or 8 values")
    parser.add_argument("--freq_low", type=float, nargs="+",
                        help="Lower relative frequency bound(s): 1 value broadcasts to all 8 parameter slots, or 8 values")
    
    parser.add_argument("--video_path", type=str, default="render.mp4")
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--method", type=str, default="eulerian",
                        help="Amplification method [eulerian, eulerian_abs, eulerian_mod, eulerian_abs_mod]. "
                             "All methods perform mean-anchored displacement amplification "
                             "(semantics changed 2026-10-05; see amplify_frame_data_eulerian's docstring)")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--frozen_cam", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        try:
            from mmengine.config import Config
        except ImportError:
            from mmcv import Config
        from utils.params_utils import merge_hparams
        config = Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    safe_state(args.quiet)

    # bug 14: validate lengths and broadcast instead of silently truncating with zip()
    freq_cutoffs = build_freq_cutoffs(args.freq_low, args.freq_high)
    amp_factors, freq_cutoffs = validate_amp_args(args.amp_factors, freq_cutoffs)
    render_sets(
        model.extract(args), 
        hyperparam.extract(args), 
        args.iteration, 
        pipeline.extract(args), 
        amp_factors, 
        freq_cutoffs, 
        method=args.method, 
        low_vram_mode=args.low_vram, 
        frozen_cam=args.frozen_cam,
        path=args.video_path,
        fps=args.video_fps
        )

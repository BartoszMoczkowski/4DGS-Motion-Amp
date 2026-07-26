import pandas as pd
import torch
import os 
import torch
import time

from render_amp import load_config, AmpConfig, generate_frame_data, render_data
from render_amp import amplify_frame_data_eulerian,amplify_frame_data_eulerian_mod,amplify_frame_data_eulerian_abs,amplify_frame_data_eulerian_abs_mod

class AMPUI():
    # Helper class for running the modified rendering pipeline
    config = None
    low_vram_mode = False
    def __init__(self):
        torch.cuda.memory._record_memory_history(enabled=True)
        print("AMPUI initialized")

    def load_config(self, model_path, config_path, amp_factors, freq_cutoffs):
        # load the scene from the given paths and store the data for the amplification
        with torch.no_grad():
            try:
                del self.config
                torch.cuda.empty_cache()
            except:
                pass
            self.config : AmpConfig = load_config(model_path, config_path, amp_factors, freq_cutoffs)
            try:
                del self.values
            except:
                pass
            try:
                del self.ras_settings
            except:
                pass
            torch.cuda.empty_cache()
            
            values, ras_settings = generate_frame_data(self.config.scene.getVideoCameras(),
                                                             self.config.gaussians,
                                                             self.config.pipeline,
                                                             self.config.background,
                                                             self.config.cam_type,
                                                             self.low_vram_mode
                                                             )
            self.values : torch.Tensor = values
            self.ras_settings = ras_settings

    def render(self, method):
        # Run the amplification and rendering and return the resulting images, as well as 
        # the time needed to run the amplification step
        with torch.no_grad():

            start_time = time.time_ns()

            if method == "base":
                amped_values = amplify_frame_data_eulerian(self.values, self.config.amp_factors, self.config.freq_list,self.low_vram_mode)
            elif method == "base segmented":
                amped_values = amplify_frame_data_eulerian_mod(self.values, self.config.amp_factors, self.config.freq_list,self.low_vram_mode)
            elif method == "abs":
                amped_values = amplify_frame_data_eulerian_abs(self.values, self.config.amp_factors, self.config.freq_list,self.low_vram_mode)
            elif method == "abs segmented":
                amped_values = amplify_frame_data_eulerian_abs_mod(self.values, self.config.amp_factors, self.config.freq_list,self.low_vram_mode)


            execution_time = time.time_ns() - start_time
            images, _,_ = render_data(amped_values, self.ras_settings, self.config.scene.getVideoCameras(), "video", self.config.cam_type,self.low_vram_mode, frozen_cam=True)
            del amped_values
            torch.cuda.empty_cache()
            return images, execution_time


# prepare configs for running automated tests
data = pd.DataFrame()

a_s = [2] + [-1.0] * 7 # amplfication factors
freqs = [(0.0,1.0)] * 8 # frequency ranges

# methods to test
methods = [
    'base', 
    'base segmented', 
    'abs', 
    'abs segmented'
    ]
# VRAM modes to test (True = low_vram_mode is on)
vram_modes = [
    False, 
    True
    ]
# Models to test (model_path, config_path)
models = [
    ('multipleview\\test_hand_2','multipleview\\default.py'),
    ('dnerf\\bouncingballs','dnerf\\bouncingballs.py'),
    ('dnerf\\lego','dnerf\\lego.py'),
    ]

# iteration for combination
repeats = 5
results = []


for method in methods:
    for vram_mode in vram_modes:
        for model,config in models:
            for _ in range(repeats):

                # For each method, model and vram mode run the given number of iterations
                # For each iteration load the scene and run the render pipeline, then save the performance metrics 
                torch.cuda.empty_cache()
                try:
                    AI = AMPUI()
                    AI.low_vram_mode = vram_mode
                    AI.load_config(os.path.join("./output",model),os.path.join("./arguments",config),a_s,freqs)

                    torch.cuda.memory.reset_accumulated_memory_stats()
                    torch.cuda.memory.reset_max_memory_cached()
                    torch.cuda.memory.reset_max_memory_allocated()

                    frames, execution_time = AI.render(method)

                    peak_memory_allocated = torch.cuda.max_memory_allocated()
                    peak_memory_cached = torch.cuda.max_memory_cached()

                    # Results are saved in Mb for VRAM and ms for runtime to avoid using numbers with large orders  of magnitude
                    results.append([model,method,vram_mode,peak_memory_allocated/1e6,peak_memory_cached/1e6,execution_time/1e6,""])

                    del frames, AI
                except Exception as e:
                    results.append([model,method,vram_mode,'-','-','-',e])
                    

df = pd.DataFrame(results)
df.to_csv("results.csv")
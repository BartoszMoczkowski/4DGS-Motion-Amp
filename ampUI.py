import streamlit as st
import os 
import itertools as it
from render_amp import load_config, AmpConfig, generate_frame_data, render_data
from render_amp import amplify_frame_data_eulerian,amplify_frame_data_eulerian_mod,amplify_frame_data_eulerian_abs,amplify_frame_data_eulerian_abs_mod
import torch
import numpy as np
import io 
import av
import time
from PIL import Image

# Because there are a lot of input parameters for the modified pipeline, A simple GUI was created 
# in streamlit to make testing easier.
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


# store the AMPUI object in the streamlit session_statet to prevent it from being unloaded on restarts
if "AI" not in st.session_state:
    st.session_state["AI"] = AMPUI()   
AI = st.session_state["AI"]

st.title("AMP UI")

# list all models as a dropdown menu
model_folders = [folder for folder in os.listdir("./output")]
secondary_model_folders = [list(map(lambda x : os.path.join(folder,x),os.listdir(os.path.join("./output", folder)))) for folder in model_folders] 
secondary_model_folders = list(it.chain.from_iterable(secondary_model_folders))
selected_model = st.selectbox("Select Folder", secondary_model_folders)

# list of configs as a dropdown menu
config_folders = [folder for folder in os.listdir("./arguments") if os.path.isdir(os.path.join("./arguments", folder))]
secondary_config_folders = [list(map(lambda x : os.path.join(folder,x),os.listdir(os.path.join("./arguments", folder)))) for folder in config_folders] 
secondary_config_folders = list(it.chain.from_iterable(secondary_config_folders))
selected_config = st.selectbox("Select Folder", secondary_config_folders)

# Create fields in which the user can set the amplification factors and frequency filtering ranges
chanels_list = ["pos3d","pos2d","rotation","scale","opacity","SHs","color","cov3D"]
a_s = [-1.0] * len(chanels_list)
freq_low_list = [0.0]*len(chanels_list)
freq_high_list = [1.0]*len(chanels_list)

amp_factors, freq = st.columns(2)
freq_low, freq_high = freq.columns(2)

for i in range(8):
    a_s[i] = amp_factors.number_input(f"{chanels_list[i]} Amplification Factor", min_value=-1.0, max_value=100.0, value=a_s[i],step=0.01)
    freq_low_list[i] = freq_low.number_input(f"{chanels_list[i]} Low Frequency Cutoff", min_value=0.0, max_value=100.0, value=freq_low_list[i])
    freq_high_list[i] = freq_high.number_input(f"{chanels_list[i]} High Frequency Cutoff", min_value=0.0, max_value=100.0, value=freq_high_list[i])

# Create checkbox for low VRAM modoe an a dropdown for the different algorithms
AI.low_vram_mode = st.checkbox("Low VRAM mode")
method = st.selectbox("Select Method", ["base","base segmented","abs", "abs segmented"])

# Button for loading the scene
if st.button("Load Config", on_click=lambda : AI.load_config(os.path.join("./output", selected_model), os.path.join("./arguments", selected_config),a_s,list(zip(freq_low_list,freq_high_list)))):
    print(
        os.path.join("./output", selected_model), 
        os.path.join("./arguments", selected_config),
        a_s,
        list(zip(freq_low_list,freq_high_list)))
    st.write(AI.config  )

# Button to start rendering
if st.button("render"):

    torch.cuda.memory.reset_accumulated_memory_stats()
    torch.cuda.memory.reset_max_memory_cached()
    torch.cuda.memory.reset_max_memory_allocated()

    frames, execution_time = AI.render(method)

    peak_memory_allocated = torch.cuda.max_memory_allocated()
    peak_memory_cached = torch.cuda.max_memory_cached()

    height, width = frames[0].shape[:2]
    fps = 20 # For testing this is set manualy in the code
    output_buffer = io.BytesIO()

    container = av.open(output_buffer, mode='w', format='mp4')

    stream = container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    space_time = []

    for frame in frames:
        # To create the crosssection of the video we take a slice of it along the 1st or 2nd axis,
        # one againg set manual as it is rarly changed
        space_time.append(np.expand_dims(frame[:,200,:],axis=1))
        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(video_frame):
            container.mux(packet)
    image = Image.fromarray(np.hstack(space_time))
    del frames
    st.image(image.resize((width,height)),use_column_width=True)
    for packet in stream.encode():
        container.mux(packet)

    container.close()

    video_bytes = output_buffer.getvalue()
    video_stream = io.BytesIO(video_bytes)

    # Show the video and the performance metrics
    st.video(video_stream)

    st.write(f"Algorithm run time: {execution_time/1e6}ms")
    st.write(f"Max memory allocated: {peak_memory_allocated/1e6}Mb")
    st.write(f"Max memory cached: {peak_memory_cached/1e6}Mb")
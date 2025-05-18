#Create a skeleton streamlit app...
import streamlit as st
import os 
import itertools as it
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from render_amp import main, load_config, AmpConfig, generate_frame_data, render_data
from render_amp import amplify_frame_data_eulerian,amplify_frame_data_eulerian_abs,amplify_frame_data_phase,amplify_frame_data_phase_abs
import torch
import numpy as np
import io 
import av
torch.cuda.empty_cache()
def run_amp(model_path, config_path, amp_list, freq_list, method):
    main(model_path, config_path, amp_list, freq_list, method)

class AMPUI():
    config = None
    def __init__(self):
        print("AMPUI initialized")

    def load_config(self, model_path, config_path, amp_factors, freq_cutoffs):
        with torch.no_grad():

            self.config : AmpConfig = load_config(model_path, config_path, amp_factors, freq_cutoffs)
            values, ras_settings = generate_frame_data(self.config.scene.getVideoCameras(),
                                                             self.config.gaussians,
                                                             self.config.pipeline,
                                                             self.config.background,
                                                             self.config.cam_type
                                                             )
            self.values = values
            self.ras_settings = ras_settings

    def render(self):
        with torch.no_grad():

            amped_values = amplify_frame_data_eulerian(self.values, self.config.amp_factors, self.config.freq_list)
            images, _,_ = render_data(amped_values, self.ras_settings, self.config.scene.getVideoCameras(), "video", self.config.cam_type)
            del amped_values
            torch.cuda.empty_cache()
            return images

if "AI" not in st.session_state:
    st.session_state["AI"] = AMPUI()   
AI = st.session_state["AI"]

st.title("AMP UI")


#Create a list of all folders in the avialable under ./output folder and generate a dropdown menu for user to select from
model_folders = [folder for folder in os.listdir("./output")]
secondary_model_folders = [list(map(lambda x : os.path.join(folder,x),os.listdir(os.path.join("./output", folder)))) for folder in model_folders] 
secondary_model_folders = list(it.chain.from_iterable(secondary_model_folders))
#Create a dropdown menu for user to select from the list of secondary folders
selected_model = st.selectbox("Select Folder", secondary_model_folders)

#filter only folders from conifg_folders...
config_folders = [folder for folder in os.listdir("./arguments") if os.path.isdir(os.path.join("./arguments", folder))]
secondary_config_folders = [list(map(lambda x : os.path.join(folder,x),os.listdir(os.path.join("./arguments", folder)))) for folder in config_folders] 
secondary_config_folders = list(it.chain.from_iterable(secondary_config_folders))
#Create a dropdown menu for user to select from the list of secondary folders
selected_config = st.selectbox("Select Folder", secondary_config_folders)

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

method = st.selectbox("Select Method", ["eulerian","phase"])

if st.button("Load Config", on_click=lambda : AI.load_config(os.path.join("./output", selected_model), os.path.join("./arguments", selected_config),a_s,list(zip(freq_low_list,freq_high_list)))):

    st.write(AI.config  )



if st.button("render"):
    frames = AI.render()
    height, width = frames[0].shape[:2]
    fps = 20  # frames per second

    # Create an in-memory buffer.
    output_buffer = io.BytesIO()

    # Open an output container in 'write' mode with format 'mp4'
    container = av.open(output_buffer, mode='w', format='mp4')

    # Add a video stream. Here we use 'libx264' (H.264 codec).
    stream = container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    # Convert each frame to an AV VideoFrame and encode it.
    for frame in frames:
        # Convert the NumPy array (assumed to be in RGB format) to a PyAV VideoFrame.
        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        # Encode the frame and mux the resulting packets to the container.
        for packet in stream.encode(video_frame):
            container.mux(packet)

    # Flush any remaining packets.
    for packet in stream.encode():
        container.mux(packet)

    container.close()

    # Retrieve the video bytes from the in-memory buffer.
    video_bytes = output_buffer.getvalue()
    video_stream = io.BytesIO(video_bytes)
    print(frames[0].shape)
    #st.image(frames[0])
    st.video(video_stream)
    
### Motion Amplification for 4D Gaussian Splatting
This is the project for the Thesis in Technology Univeristy of Lodz by Bartosz Moczkowski.

The project is based on previous work of 4DGS https://github.com/hustvl/4DGaussians and as such only a small number of files were created by me. 
These files are:
* render_amp.py
* run_renders_auto.py
* cameras.py
* ampUI.py
* motion_amp/render.py

Additional files that contain results of the work include:
* results_visual
* result.csv
* data/multipleview
* output

the remaining data in ./data is synthetic data from DNerf https://github.com/albertpumarola/D-NeRF.

Due to the size limitations the ./output folder containing the generated trained models is not included in the repository.

### Documentation

Compiled project documentation lives in [`docs/`](docs/README.md): project overview, the motion-segmentation work (`motion_seg/`), the Omniverse→4DGS synthetic-data pipeline (`omniverse_pipeline/`), and the pipeline orchestrator (`orchestrator/`). Detailed chronological working notes are in `.claude_notes/`.
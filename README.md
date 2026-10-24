### Motion Amplification for 4D Gaussian Splatting
This is the project for the Thesis in Technology Univeristy of Lodz by Bartosz Moczkowski.

The project is based on previous work of 4DGS https://github.com/hustvl/4DGaussians and as such only a small number of files were created by me. 
These files are:
* core/render_amp.py
* amp-ui/amp_ui/run_renders_auto.py
* amp-ui/amp_ui/cameras.py
* amp-ui/amp_ui/ampUI.py
* core/motion_amp/renderer.py

Additional files that contain results of the work include:
* results_visual
* result.csv
* data/multipleview
* output

the remaining data in ./data is synthetic data from DNerf https://github.com/albertpumarola/D-NeRF.

Due to the size limitations the ./output folder containing the generated trained models is not included in the repository.

### Documentation & Quick Start

Compiled project documentation lives in [`docs/`](docs/README.md) and [`AGENTS.md`](AGENTS.md).

#### Quick Start (from clone to running experiment)

```bash
# 1. Install entire workspace environment (Python 3.12 + PyTorch cu126)
uv sync

# 2. Inspect available commands across pipeline phases
uv run amp --help

# 3. Dry-run an experiment spec (touches nothing, validates plan)
uv run amp exp run experiments/pump_grid_v2.yaml --dry-run

# 4. Start local MLflow dashboard (SQLite store at runs/mlflow/mlflow.db)
uv run amp mlflow serve    # UI at http://127.0.0.1:5000

# 5. Explore runs, configs, metrics, and artifacts
uv run amp explore
```

For detailed architecture, pipeline stages, and command catalogs, see [`AGENTS.md`](AGENTS.md) and [`docs/overview.md`](docs/overview.md).
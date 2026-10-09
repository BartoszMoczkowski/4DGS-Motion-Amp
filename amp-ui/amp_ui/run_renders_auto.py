import pandas as pd
import os
import sys
import time

# Resolve paths relative to the repo root (this file lives at amp-ui/amp_ui/,
# so the repo root is two levels up). Model outputs live in <root>/output and
# configs in <root>/core/arguments; render_amp.py lives in <root>/core.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORE_DIR = os.path.join(REPO_ROOT, "core")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
ARGUMENTS_DIR = os.path.join(CORE_DIR, "arguments")

import torch

from render_amp import load_config, AmpConfig, generate_frame_data, render_data
from render_amp import amplify_frame_data_eulerian,amplify_frame_data_eulerian_mod,amplify_frame_data_eulerian_abs,amplify_frame_data_eulerian_abs_mod


def _clone_values(values):
    # Deep-copy the per-frame parameter structure before amplification: the
    # amplify_frame_data_* functions mutate the list in place.
    return [[(v.clone() if isinstance(v, torch.Tensor) else v) for v in channel] for channel in values]


def _find_config_rel(dataset, name, arguments_dir=ARGUMENTS_DIR):
    # Pair a model with a config: prefer <dataset>/<name>.py, then fall back to
    # <dataset>/default.py or <dataset>/<dataset>_default.py.
    candidates = [
        os.path.join(dataset, f"{name}.py"),
        os.path.join(dataset, "default.py"),
        os.path.join(dataset, f"{dataset}_default.py"),
    ]
    return next(
        (c for c in candidates if os.path.isfile(os.path.join(arguments_dir, c))),
        None,
    )


def resolve_model_specs(models=None):
    """Resolve model names to (model_rel, config_rel) pairs.

    ``models=None`` auto-discovers every trained model under OUTPUT_DIR (the historical
    default); an explicit list of ``dataset/name`` strings (what `amp bench renders
    --models` passes) is validated the same way and raises SystemExit on a miss.
    """
    if models is None:
        specs = discover_models(OUTPUT_DIR, ARGUMENTS_DIR)
        if not specs:
            available = []
            for dataset in sorted(os.listdir(OUTPUT_DIR)) if os.path.isdir(OUTPUT_DIR) else []:
                dataset_dir = os.path.join(OUTPUT_DIR, dataset)
                if os.path.isdir(dataset_dir):
                    available.extend(os.path.join(dataset, d) for d in sorted(os.listdir(dataset_dir)))
            raise SystemExit(
                "No trained models with matching configs found.\n"
                f"Looked for trained models (dirs containing cfg_args) under: {OUTPUT_DIR}\n"
                f"and configs under: {ARGUMENTS_DIR}\n"
                f"Model dirs currently on disk: {available or 'none'}"
            )
        return specs
    specs = []
    for model in models:
        parts = model.replace("\\", "/").strip("/").split("/")
        if len(parts) != 2:
            raise SystemExit(
                f"invalid model spec {model!r}: expected '<dataset>/<name>' relative to {OUTPUT_DIR}"
            )
        dataset, name = parts
        model_dir = os.path.join(OUTPUT_DIR, dataset, name)
        if not os.path.isfile(os.path.join(model_dir, "cfg_args")):
            raise SystemExit(f"model {model!r} not found (no cfg_args in {model_dir})")
        config_rel = _find_config_rel(dataset, name)
        if config_rel is None:
            raise SystemExit(f"no config found for model {model!r} under {ARGUMENTS_DIR}")
        specs.append((model, config_rel))
    return specs


def discover_models(output_dir, arguments_dir):
    # Find trained models on disk (a model dir must contain cfg_args) and pair
    # each with a config: prefer <dataset>/<name>.py, then fall back to
    # <dataset>/default.py or <dataset>/<dataset>_default.py.
    models = []
    if not os.path.isdir(output_dir):
        return models
    for dataset in sorted(os.listdir(output_dir)):
        dataset_dir = os.path.join(output_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for name in sorted(os.listdir(dataset_dir)):
            model_dir = os.path.join(dataset_dir, name)
            if not os.path.isdir(model_dir):
                continue
            if not os.path.isfile(os.path.join(model_dir, "cfg_args")):
                continue
            config_rel = _find_config_rel(dataset, name, arguments_dir)
            if config_rel is None:
                print(f"WARNING: no config found for model {os.path.join(dataset, name)}; skipped.")
                continue
            models.append((os.path.join(dataset, name), config_rel))
    return models


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

            # Work on a deep copy so self.values always retains the raw
            # extracted data (the amplify functions mutate in place).
            raw_values = _clone_values(self.values)

            # The amplify functions are async CUDA work; synchronize before
            # starting and before stopping the timer so both VRAM modes are
            # timed the same way.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time_ns()

            if method == "base":
                amped_values = amplify_frame_data_eulerian(raw_values, self.config.amp_factors, self.config.freq_list,self.low_vram_mode)
            elif method == "base segmented":
                amped_values = amplify_frame_data_eulerian_mod(raw_values, self.config.amp_factors, self.config.freq_list,self.low_vram_mode)
            elif method == "abs":
                amped_values = amplify_frame_data_eulerian_abs(raw_values, self.config.amp_factors, self.config.freq_list,self.low_vram_mode)
            elif method == "abs segmented":
                amped_values = amplify_frame_data_eulerian_abs_mod(raw_values, self.config.amp_factors, self.config.freq_list,self.low_vram_mode)


            if torch.cuda.is_available():
                torch.cuda.synchronize()
            execution_time = time.time_ns() - start_time
            images, _,_ = render_data(amped_values, self.ras_settings, self.config.scene.getVideoCameras(), "video", self.config.cam_type,self.low_vram_mode, frozen_cam=True)
            del amped_values
            torch.cuda.empty_cache()
            return images, execution_time


# results.csv schema (named headers matching the archived CSV at repo root;
# no index column)
COLUMNS = ["model","method","low_vram","mem_alloc","mem_seg","time","error_msg"]
RESULTS_PATH = os.path.join(REPO_ROOT, "results.csv")

# Benchmark defaults (historical module-scope constants, now the defaults of
# run_benchmark(); `amp bench renders` overrides them per-flag).
DEFAULT_AMP_FACTORS = [2] + [-1.0] * 7  # amplification factors
DEFAULT_FREQS = [(0.0, 1.0)] * 8        # frequency ranges
DEFAULT_METHODS = ['base', 'base segmented', 'abs', 'abs segmented']
DEFAULT_VRAM_MODES = [False, True]      # True = low_vram_mode is on
DEFAULT_REPEATS = 5


def save_results(results, path):
    # Persist the accumulated results atomically (temp file + os.replace) so a
    # crash mid-run preserves every completed row.
    df = pd.DataFrame(results, columns=COLUMNS)
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def run_benchmark(models=None, methods=None, vram_modes=None, repeats=None, out_csv=None,
                  amp_factors=None, freqs=None):
    """Run the methods x vram_modes x models x repeats render benchmark.

    Every parameter defaults to the historical module-scope constant (DEFAULT_* /
    RESULTS_PATH); calling with no arguments reproduces the original script behavior exactly,
    including the atomic per-combo CSV write (a crash mid-run keeps completed rows).
    Returns the accumulated results rows (COLUMNS order).
    """
    a_s = amp_factors if amp_factors is not None else DEFAULT_AMP_FACTORS
    freq_cutoffs = freqs if freqs is not None else DEFAULT_FREQS
    methods = methods if methods is not None else DEFAULT_METHODS
    vram_modes = vram_modes if vram_modes is not None else DEFAULT_VRAM_MODES
    repeats = repeats if repeats is not None else DEFAULT_REPEATS
    out_csv = out_csv if out_csv is not None else RESULTS_PATH

    model_specs = resolve_model_specs(models)
    print(f"Benchmarking {len(model_specs)} model(s): {model_specs}")

    results = []
    for method in methods:
        for vram_mode in vram_modes:
            for model, config in model_specs:
                for _ in range(repeats):

                    # For each method, model and vram mode run the given number of iterations
                    # For each iteration load the scene and run the render pipeline, then save the performance metrics
                    torch.cuda.empty_cache()
                    try:
                        AI = AMPUI()
                        AI.low_vram_mode = vram_mode
                        AI.load_config(os.path.join(OUTPUT_DIR, model), os.path.join(ARGUMENTS_DIR, config), a_s, freq_cutoffs)

                        torch.cuda.memory.reset_accumulated_memory_stats()
                        torch.cuda.memory.reset_max_memory_cached()
                        torch.cuda.memory.reset_max_memory_allocated()

                        frames, execution_time = AI.render(method)

                        peak_memory_allocated = torch.cuda.max_memory_allocated()
                        peak_memory_cached = torch.cuda.max_memory_cached()

                        # Results are saved in Mb for VRAM and ms for runtime to avoid using numbers with large orders  of magnitude
                        results.append([model, method, vram_mode, peak_memory_allocated/1e6, peak_memory_cached/1e6, execution_time/1e6, ""])

                        del frames, AI
                    except Exception as e:
                        results.append([model, method, vram_mode, '-', '-', '-', str(e)])

                    # Persist after every benchmark combo so a crash mid-run keeps
                    # the rows already completed.
                    save_results(results, out_csv)

    print(f"Benchmark finished; results written to {out_csv}")
    return results


if __name__ == "__main__":
    run_benchmark()

import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import calendar
from utils import general, vis
import math
from matplotlib import colormaps

### Set up and parameter selection ########

# read configuration
config_path = Path(__file__).parent / "0.config.yaml"
config = general.read_config(config_path)

# get models and parameters from config
models = config["models"]

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
unperturbed_e3sm_path = Path(config["upert_e3sm_path"])
perturbed_e3sm_path = Path(config["pert_e3sm_path"])
plot_dir = exp_dir / "plots"  # where to save plots
if not plot_dir.exists():
    plot_dir.mkdir(parents=False, exist_ok=True)

ic_dates = [
    dt.datetime.strptime(str_date, "%Y-%m-%dT%Hz") for str_date in config["ic_dates"]
]
all_lead_times = np.arange(0, config["n_timesteps"] + 1) * 6  # in hours
n_ics = len(ic_dates)
n_timesteps = config["n_timesteps"]
# open all datasets, concat on model dim, and sort latitudes, which should instead by done in the model output script
model_ds = (
    xr.open_mfdataset(
        str(exp_dir / "*_output.nc"),
        combine="nested",
        concat_dim="model",
        preprocess=lambda x: general.sort_latitudes(x, "BLOOG", input=False),
    )
    / 100
)
# open E3SM dataset
upert_e3sm_ds = xr.open_dataset(unperturbed_e3sm_path)
pert_e3sm_ds = xr.open_dataset(perturbed_e3sm_path)
breakpoint()
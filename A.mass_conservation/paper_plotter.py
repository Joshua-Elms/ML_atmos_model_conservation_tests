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
plot_var = "sp" # choose either "msl" or "sp", can only use "sp" if models = ["sfno"] (other models don't output SP)
dry = True # whether to plot dry mass conservation (i.e., exclude moisture contribution to surface pressure)
display_var = r"$p_s^{dry}$" if dry and plot_var=="sp" else "p_s" if plot_var=="sp" else "MSLP"
# vis options
cmap_str = "nipy_spectral" # options here: matplotlib.org/stable/tutorials/colors/colormaps.html
weird_color_const = 3 # unsure how this works, but it spaces out colors better when there are many models and inits
day_interval_x_ticks = 15 # how many days between x-ticks on the plot
standardized_ylims = (975, 986) # y-limits for the plot, set to None to use the model output min/max, normally (1010, 1014)
conservation_half_range = 3 # hPa
show_legend = True
legend_ncols = 3
legend_loc = "lower left" # options: 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = exp_dir / "plots" # where to save plots
if not plot_dir.exists():
    plot_dir.mkdir(parents=False, exist_ok=True)

ic_dates = [dt.datetime.strptime(str_date, '%Y-%m-%dT%Hz') for str_date in config["ic_dates"]]
lead_times = np.arange(0, config["n_timesteps"] + 1) * 6  # in hours
n_ics = len(ic_dates)
n_timesteps = config["n_timesteps"]
# open all datasets, concat on model dim, and sort latitudes, which should instead by done in the model output script
ds = xr.open_mfdataset(str(exp_dir / "*_output.nc"), combine="nested", concat_dim="model", preprocess=lambda x:general.sort_latitudes(x, "BLOOG", input=False)) / 100 # convert to hPa
# change model names
rename_dict = {
    "FCNv2_SM": "A_ckpt", # from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_fcnv2_sm/version?version=v0.2
    "SFNO": "B_ckpt", # from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/sfno_73ch_small/version?version=0.1.0
}
models = [rename_dict.get(m, m) for m in models]
ds = ds.assign_coords(model=[rename_dict.get(m, m) for m in ds.model.values])
# n_timesteps = 5 ###DEBUG
# lead_times = lead_times[:n_timesteps+1]###DEBUG
##############################################

fig, ax = plt.subplots(figsize=(12.5, 6.5))
title = f"{'Dry ' if dry else ''}Mass Conservation in Two SFNO Checkpoints\nA (2023/09/07) vs. B (2024/04/08)" # generic is "Mass Conservation in Various ML Models"
save_title = f"{plot_var.lower()}_trends_dry={'T' if dry else 'F'}.png"
ylab = fr"Global {display_var} (hPa)" if plot_var == "sp" else "Global MSLP (hPa)"
n_ics = len(ic_dates)
n_models = len(models)
dx = 1/(n_models-1) if n_models > 1 else 1
base_color_indices = np.linspace(0, 1-dx/n_ics, n_models)[np.newaxis, :]
color_indices = np.concatenate((base_color_indices, base_color_indices + dx/(weird_color_const*n_ics))).T
qual_colors = colormaps.get_cmap(cmap_str)(color_indices)
linewidth = 2
fontsize = 24
smallsize = 20
fcst_linestyle = "solid" # see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
bound_linestyle = "dashed"
e5_linestyle = (0, (3, 1, 1, 1)) # densely dotted line

if dry:
    ds[f"MEAN_{plot_var}"] = ds[f"MEAN_{plot_var}"] - ds["MEAN_sp_moist"]

# plot initial condition and approximate conservation bounds
initial_value = np.full_like(lead_times, ds[f"MEAN_{plot_var}"].isel(lead_time=0).values.mean().round(1), dtype="float64")
ax.plot(lead_times, initial_value, alpha=0.5, color="black", linewidth=1.5*linewidth, label=f"Initial global mean = {initial_value[0]} hPa", linestyle=bound_linestyle)
ax.plot(lead_times, initial_value - conservation_half_range, color="red", linewidth=1.5*linewidth, linestyle=bound_linestyle, label=f"+/- {conservation_half_range} hPa")
ax.plot(lead_times, initial_value + conservation_half_range, color="red", linewidth=1.5*linewidth, linestyle=bound_linestyle)

### Loop through each model and plot the results ###
for m, model in enumerate(models):
    ### Load model output data ###
    data_path = exp_dir / f"{model}_output.nc" # where output data is stored
    print(f"Loading data from {data_path}")
    ##############################################

    ### Plot the results ####################

    for i, ic in enumerate(ic_dates):
        model_linedat = ds[f"MEAN_{plot_var}"].sel(model=model).isel(init_time=i).squeeze()
        color = qual_colors[m, i]
        ax.plot(lead_times[:model_linedat.size], model_linedat, color=color, linewidth=linewidth, label=f"{model} init {ic.strftime('%Y-%m-%d %Hz')}", linestyle=fcst_linestyle)
        
    # ax.plot(lead_times, ens_mean, color="red", linewidth=2*linewidth, label="Mean of forecast lines", linestyle=fcst_linestyle)

ax.set_xticks(lead_times[::4*day_interval_x_ticks], (lead_times[::4*day_interval_x_ticks]//(24)), fontsize=smallsize)
if standardized_ylims:
    val_range = standardized_ylims
    print(f"Using standardized y-limits: {val_range}")
else:
    try: 
        val_range = (math.floor(ds[f"MEAN_{plot_var}"].min()), math.ceil(ds[f"MEAN_{plot_var}"].max()))
    except OverflowError: # in case of NaNs
        val_range = (1013-7, 1013+7) # wide range to cover non-conservative models that would cause a blowup here
    print(f"Using model output min/max for y-limits ({model}): {val_range}")
val_diff = val_range[1] - val_range[0]
if val_diff < 2.0:
    ytick_interval = 0.5
elif val_diff < 10.0:
    ytick_interval = 2.0
elif val_diff < 20.0:
    ytick_interval = 3.0
else:
    ytick_interval = 5.0
yticks = np.arange(val_range[0], val_range[1]+ytick_interval, ytick_interval)
if ytick_interval % 1 == 0:
    yticks = yticks.astype(int)
ax.set_yticks(yticks, yticks, fontsize=smallsize)
ax.set_xlabel("Simulation Time (days)", fontsize=fontsize)
ax.set_ylabel(ylab, fontsize=fontsize)
ax.set_xlim(xmin=-3, xmax=lead_times[-1]+3)
ax.set_ylim(val_range[0]-0.1*val_diff, val_range[1]+0.1*val_diff)
fig.suptitle(title, fontsize=28)
ax.grid()
ax.set_facecolor("#FFFFFF")
fig.tight_layout()
if show_legend:
    plt.legend(fontsize=10, loc=legend_loc, ncols=legend_ncols)
plt.savefig(plot_dir / save_title, dpi=300, bbox_inches="tight")
print(f"Saved figure to {plot_dir / save_title}")
###########################################

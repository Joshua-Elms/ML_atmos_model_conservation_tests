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
# vis options
cmap_str = "Dark2" # options here: matplotlib.org/stable/tutorials/colors/colormaps.html, if "single:" is included, only one color will be used
day_interval_x_ticks = 1 # how many days between x-ticks on the plot
spec_int = 5e-4
individual_standardized_ylims = (3.08e9, 3.15e9) # y-limits for the plot, set to None to use the model output min/max, normally (1010, 1014)
show_legend = True
plot_base_fields = True # whether to plot the base fields (pointwise data) for each model
relabel_Pangu6 = True # whether to relabel "Pangu6" to "Pangu24" and only use every fourth lead time

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = exp_dir / "plots" # where to save plots

ic_dates = [dt.datetime.strptime(str_date, "%Y-%m-%dT%Hz") for str_date in config["ic_dates"]]
all_lead_times = np.arange(0, config["n_timesteps"] + 1) * 6  # in hours
n_ics = len(ic_dates)
n_timesteps = config["n_timesteps"]
##############################################

### Loop through each model and plot the individual values ###
for model in models:
    ### Load model output data ###
    data_path = exp_dir / f"{model}_output.nc" # where output data is stored
    print(f"Loading data from {data_path}")
    ds = xr.open_dataset(data_path)
    ds = ds.assign_attrs({"time units": "hours since start"})
    ds = ds.sortby("lat", ascending=True)
    ##############################################

    ### Plot the results ######################
    if model == "Pangu24":
        lead_times = all_lead_times[::4]
    elif relabel_Pangu6 and model == "Pangu6":
        model = "Pangu24"
        lead_times = all_lead_times[::4]
    else:
        lead_times = all_lead_times.copy()  # in hours
    title = f"Total Energy Trends\n{model}"
    save_title = f"total_energy_trends_{model}.png"
    ylab = "LW-TE (J/m^2)"
    linewidth = 2
    fontsize = 24
    smallsize = 20
    fcst_linestyle = "solid" # see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    e5_linestyle = (0, (3, 1, 1, 1)) # densely dotted line
    cmap = colormaps.get_cmap(cmap_str) if "single:" not in cmap_str else None
    qual_colors = cmap(np.linspace(0, 1, n_ics)) if cmap is not None else [cmap_str.split(":")[1]] * n_ics

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    for i, ic in enumerate(ic_dates):
        model_linedat = ds["lw_te"].isel(init_time=i).sel(lead_time=lead_times).squeeze()
        color = qual_colors[i]
        ax.plot(lead_times, model_linedat, color=color, alpha=0.5, linewidth=linewidth, label=f"Forecast init   {ic.strftime('%Y-%m-%d %Hz')}", linestyle=fcst_linestyle)

    ens_mean = ds["lw_te"].mean(dim="init_time").squeeze()
    # ax.plot(lead_times, ens_mean, color="red", linewidth=2*linewidth, label="Mean of forecast lines", linestyle=fcst_linestyle)
       
    if model == "Pangu24": # only 24 hour timestep
        ax.set_xticks(lead_times[::day_interval_x_ticks], (lead_times[::day_interval_x_ticks]//(24)), fontsize=smallsize)
    else:
        ax.set_xticks(lead_times[::4*day_interval_x_ticks], (lead_times[::4*day_interval_x_ticks]//(24)), fontsize=smallsize)
    if individual_standardized_ylims:
        val_range = individual_standardized_ylims
        print(f"Using standardized y-limits: {val_range}")
    else:
        try: 
            val_range = (math.floor(ens_mean.min()), math.ceil(ens_mean.max()))
            if max(abs(val_range[0]), val_range[1]) > 4e9:
                raise OverflowError
        except OverflowError: # in case of NaNs
            val_range = (2.5e9, 3.5e9) # wide range to cover non-conservative models that would cause a blowup here
        print(f"Using model output min/max for y-limits ({model}): {val_range}")
    val_diff = val_range[1] - val_range[0]
    if val_diff <= 0.2e9:
        ytick_interval = 0.02e9
    elif val_diff <= 0.5e9:
        ytick_interval = 0.2e9
    elif val_diff <= 1.0e9:
        ytick_interval = 0.4e9
    else:
        ytick_interval = 1.0e9
    yticks = np.arange(val_range[0], val_range[1]+ytick_interval, ytick_interval)
    ax.set_yticks(yticks, yticks, fontsize=smallsize)
    ax.set_xlabel("Simulation Time (days)", fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)
    ax.set_xlim(xmin=0, xmax=lead_times[-1])
    ax.set_ylim(val_range[0]-0.1*val_diff, val_range[1]+0.1*val_diff)
    fig.suptitle(title, fontsize=28)
    ax.grid()
    ax.set_facecolor("#FFFFFF")
    fig.tight_layout()
    if show_legend:
        plt.legend(fontsize=12, loc="upper left", ncols=2)
    plt.savefig(plot_dir / save_title, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {plot_dir / save_title}")
    ###########################################
    
    
if plot_base_fields:
    ### plot TE as a sanity check ###
    titles = [f"{model}: TE init {ic_dates[0].strftime('%d-%m-%Y %Hz')} @ {(t/4):.2f} days lead time" for t in np.arange(0, n_timesteps+1)]
    data = ds["te"].isel(init_time=0).squeeze()  # select final init time
    gif_plot_var = f"TE_{model}"
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=gif_plot_var,
        iter_var="lead_time",
        iter_vals=np.arange(0, n_timesteps+1),
        plot_dir=plot_dir,
        units="J/m^2",
        cmap="PRGn",
        titles=titles,
        keep_images=False,
        dpi=300,
        fps=1,
        fig_size=(8, 4),
        vlims=(2.8e9, 3.2e9),  # Set vlims for better visualization
        central_longitude=180.0,
        adjust = {
        "top": 0.93,
        "bottom": 0.03,
        "left": 0.09,
        "right": 0.87,
        "hspace": 0.0,
        "wspace": 0.0,
        },
        cbar_kwargs={
            "rotation": "horizontal",
            "y": -0.015,
            "horizontalalignment": "right",
            "labelpad": -29,
            "fontsize": 9
        },
    )
    print(f"Made {gif_plot_var}.gif.")

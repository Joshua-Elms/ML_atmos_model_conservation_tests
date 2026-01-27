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
perturbed = config["temp_perturbation_degC"] != 0

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
unperturbed_processed_e3sm_path = exp_dir / "upert_processed_e3sm.nc"
perturbed_processed_e3sm_path = exp_dir / "pert_processed_e3sm.nc"
plot_dir = exp_dir / "plots"  # where to save plots
if not plot_dir.exists():
    plot_dir.mkdir(parents=False, exist_ok=True)

ic_dates = [
    dt.datetime.strptime(str_date, "%Y-%m-%dT%Hz") for str_date in config["ic_dates"]
]
all_lead_times_h = np.arange(0, config["n_timesteps"] + 1) * 6  # in hours
all_lead_times_d = all_lead_times_h / 24  # in days
n_ics = len(ic_dates)
n_timesteps = config["n_timesteps"]
# open all datasets, concat on model dim, and sort latitudes, which should instead by done in the model output script
model_ds = xr.open_mfdataset(
    str(exp_dir / "*_output.nc"),
    combine="nested",
    concat_dim="model",
    preprocess=lambda x: general.sort_latitudes(x, "BLOOG", input=False),
)
# DELETE THIS AFTER FIXED IN EXPERIMENT SCRIPT
if config["experiment_name"] == "pert=0C_nt=236_sai_m_all":
    model_ds = model_ds.assign_coords(
        lead_time=model_ds.lead_time / np.timedelta64(1, "h")
    )  # convert lead_time to hours
# open E3SM dataset
upert_e3sm_ds = xr.open_dataset(unperturbed_processed_e3sm_path)
pert_e3sm_ds = xr.open_dataset(perturbed_processed_e3sm_path)

# output table of TE and components at initial and final times
energy_term_names = [
    "sensible_heat",
    "geopotential",
    "kinetic",
    "latent_heat",
    "total",
]
pointwise_names = [f"{name}_energy" for name in energy_term_names]
column_names = [f"{name}_energy_column" for name in energy_term_names]
area_weighted_names = [f"AW_{name}_energy" for name in energy_term_names]
if perturbed:
    energy_ylims = {
        "sensible_heat": (246.5, 254.5),
        "latent_heat": (5.2, 8.5),
        "geopotential": (60.5, 63.5),
        "kinetic": (0.1, 0.2),
        "total": (313, 324),
    }
else:
    energy_ylims = {
        "sensible_heat": (244, 250),
        "latent_heat": (4.6, 6.25),
        "geopotential": (60, 62.5),
        "kinetic": (0.08, 0.2),
        "total": (310, 318),
    }

fmt_numbers = lambda numbers: [f"{(num.item())/1e7:.4f}" for num in numbers]

print(f"Area-weighted energy terms at initial and final times, 10^7 J/m^2:")
for i, name in enumerate(energy_term_names):
    print(f"\n\t{name}:")
    print(
        f"\t\tE3SM unperturbed (t={{0,-1}}): {fmt_numbers(upert_e3sm_ds[f"AW_{name}_energy"].isel(lead_time=[0, -1]).squeeze().values)}"
    )
    print(
        f"\t\tE3SM perturbed (t={{0,-1}})  : {fmt_numbers(pert_e3sm_ds[f"AW_{name}_energy"].isel(lead_time=[0, -1]).squeeze().values)}"
    )
    for model in models:
        print(
            f"\t\t{model} (t={{0,-1}}){' ' * (16 - len(model))}: {fmt_numbers(model_ds[f"AW_{name}_energy"].sel(model=model, lead_time=all_lead_times_h[[0, -1]]).squeeze().values)}"
        )

    # Plot 1: Unperturbed energy trend lineplots for each energy term
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        all_lead_times_h,
        (
            upert_e3sm_ds[f"AW_{name}_energy"]
            .sel(lead_time=all_lead_times_h)
            .squeeze()
            .values
            / 1e7
            if not perturbed
            else pert_e3sm_ds[f"AW_{name}_energy"]
            .sel(lead_time=all_lead_times_h)
            .squeeze()
            .values
            / 1e7
        ),
        label=f"E3SM {'perturbed' if perturbed else 'unperturbed'}",
    )
    for model in models:
        if (
            model == "Pangu24"
        ):  # this model outputs w/ a 24 hour timestep, so plot every 4th lead time to match
            ax.plot(
                all_lead_times_h[::4],
                model_ds[f"AW_{name}_energy"]
                .sel(model=model, lead_time=all_lead_times_h[::4])
                .squeeze()
                .values
                / 1e7,
                label=model,
            )
        else:
            ax.plot(
                all_lead_times_h,
                model_ds[f"AW_{name}_energy"]
                .sel(model=model, lead_time=all_lead_times_h)
                .squeeze()
                .values
                / 1e7,
                label=model,
            )
    ax.set_title(
        f"Area-weighted {name} Energy ({'Perturbed' if perturbed else 'Unperturbed'})"
    )
    ax.set_xlabel("Lead Time (days)")
    ax.set_xticks(all_lead_times_h[:: 4 * 7], all_lead_times_d[:: 4 * 7].round(0))
    ax.set_ylabel(f"Global Column Mean {name} Energy (10^7 J/m^2)")
    ax.set_ylim(energy_ylims[name])
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / f"unperturbed_{name}_energy_trends.png")
    print(f"Saved unperturbed {name} energy trend plot.")

for model in models:
    titles = [
        f"{model}: Column TE init {ic_dates[0].strftime('%d-%m-%Y %Hz')} @ {d} days lead time"
        for d in all_lead_times_d
    ]
    data = (
        model_ds[f"total_energy_column"]
        .sel(model=model, lead_time=all_lead_times_h)
        .squeeze()
    )  # select final init time
    gif_plot_var = f"{name}_{model}"
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=gif_plot_var,
        iter_var="lead_time",
        iter_vals=np.arange(len(all_lead_times_d)),
        plot_dir=plot_dir,
        units="J/m^2",
        cmap="PRGn",
        titles=titles,
        keep_images=False,
        dpi=300,
        fps=1,
        fig_size=(8, 4),
        vlims=(2.7e9, 3.4e9),  # Set vlims for better visualization
        central_longitude=180.0,
        adjust={
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
            "fontsize": 9,
        },
    )
    print(f"Made {gif_plot_var}.gif.")

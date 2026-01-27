import xarray as xr
from pathlib import Path
import numpy as np
from utils import general, model_info
import metpy
import scipy

config_path = Path(__file__).parent / "0.config.yaml"
config = general.read_config(config_path)

# unperturbed and perturbed E3SM dataset paths
upert_paths = [Path(upert_path) for upert_path in config["upert_e3sm_paths"]]
pert_paths = [Path(pert_path) for pert_path in config["pert_e3sm_paths"]]

upert_ds = xr.open_mfdataset(upert_paths, concat_dim="time", combine="nested")
pert_ds = xr.open_mfdataset(pert_paths, concat_dim="time", combine="nested")

# save paths for processed datasets
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
unperturbed_processed_e3sm_path = exp_dir / "upert_processed_e3sm.nc"
perturbed_processed_e3sm_path = exp_dir / "pert_processed_e3sm.nc"

# Get column areas for latitude-weighting fields
col_areas = upert_ds["area"].isel(time=0)
normed_col_areas = col_areas / col_areas.sum()

# Pressure levels for integration #
model_levels = model_info.STANDARD_13_LEVELS
model_levels_pa = 100 * np.array(
    model_levels
)  # convert to Pa from hPa, used for integration
model_levels_pa_w_units = model_levels_pa * metpy.units.units("Pa")

# Set constants
cp = 1005.0  # J/kg/K
g = 9.81  # m/s^2
Lv = 2.26e6  # J/kg
sb_const = (
    5.670374419e-8  # W/m^2/K^4, from https://physics.nist.gov/cgi-bin/cuu/Value?sigma
)

for i, full_ds in enumerate([upert_ds, pert_ds]):
    # Add level nearest to missing for variables like Q, V, Z, not outputted by E3SM for some reason
    vars_and_data = {
        "Q": {"missing_levs": [150, 250], "data": full_ds["Q"]},
        "Z": {"missing_levs": [150, 250], "data": full_ds["Z3"]},
        "V": {"missing_levs": [150], "data": full_ds["V"]},
    }
    for var, vinfo in vars_and_data.items():
        data = vinfo["data"]
        for lev in vinfo["missing_levs"]:
            nearest_level_idx = np.argmin(np.abs(lev - data.lev.values))
            print(
                f"Using {var} @ {data.lev.values[nearest_level_idx]:0.1f} hPa as {var}{lev}"
            )
            full_ds[f"{var}{lev}"] = data.isel(lev=nearest_level_idx)
    
    # Subset relevant variables and rename to E2S standards
    vars_in_ds = (
        set(map(str.upper, model_info.MASTER_VARIABLES_NAMES)) & set(full_ds.data_vars)
    ) | {"TMQ", "PS", "PHIS", "U050", "V050", "T050", "Z050", "Q050"}
    e3sm_to_cds = {var: var.lower() for var in vars_in_ds}
    e3sm_to_cds["U050"] = "u50"  # 50 hPa zonal wind
    e3sm_to_cds["V050"] = "v50"  # 50 hPa meridional wind
    e3sm_to_cds["T050"] = "t50"  # 50 hPa temperature
    e3sm_to_cds["Z050"] = "z50"  # 50 hPa geopotential height
    e3sm_to_cds["Q050"] = "q50"  # 50 hPa specific humidity
    e3sm_to_cds["PS"] = "sp"  # surface pressure
    e3sm_to_cds["PHIS"] = "z"  # surface geopotential
    e3sm_to_cds["TMQ"] = "tcwv"  # total column water vapor
    flat_ds = full_ds[vars_in_ds].rename(
        e3sm_to_cds
    )  # variables are on levels, e.g. z500 instead of z w/ dim "lev" including 500

    # change from single time coord to init_time (datetime) and lead_time (hours, int) coords
    flat_ds = flat_ds.expand_dims({"init_time": [flat_ds.time.values[0]]}, axis=0)
    lead_times = (
        (flat_ds.time - flat_ds.init_time).astype("int").values.flatten() / 1e9 / 3600
    )  # nanoseconds to hours
    flat_ds = flat_ds.assign_coords(time=lead_times).rename({"time": "lead_time"})

    # preprocess the data to put T, U, V, Z, Q into blocks by level
    level_blocks = {}
    for var in "tuvzq":
        levels = [level for level in model_levels if f"{var}{level}" in flat_ds]
        level_blocks[var.upper()] = [flat_ds[f"{var}{level}"] for level in levels]
        print(f"{len(levels)} {var} levels found: {levels}")

    # combine level blocks into single DataArrays
    for key in level_blocks:
        assert len(level_blocks[key]) == len(
            model_levels
        ), f"Level block for {key} has {len(level_blocks[key])} levels, expected {len(model_levels)}"
        level_blocks[key] = xr.concat(level_blocks[key], dim="level").assign_coords(
            level=model_levels
        )

    # create new dataset with level blocks
    ds = xr.Dataset(level_blocks)

    # add 2D variables
    for var in ["sp", "z", "tcwv"]:
        ds[var] = flat_ds[var]
    ds["normed_area"] = normed_col_areas

    ### Calculate total energy components ###
    integrate = lambda da: (1 / g) * scipy.integrate.trapezoid(
        da, model_levels_pa, axis=0
    )
    # sensible heat
    ds["sensible_heat_energy"] = cp * ds["T"]
    ds["sensible_heat_energy_column"] = (
        ("init_time", "lead_time", "ncol"),
        integrate(ds["sensible_heat_energy"]),
    )
    ds["AW_sensible_heat_energy"] = (
        ds["sensible_heat_energy_column"] * ds["normed_area"]
    ).sum(dim=["ncol"])

    # geopotential energy
    ds["geopotential_energy"] = (
        g * ds["Z"]
    )  # E3SM provides geopotential height in meters, see full_ds["Z500"].mean().values ~ 5000 (m)
    ds["geopotential_energy_column"] = (
        ("init_time", "lead_time", "ncol"),
        integrate(ds["geopotential_energy"]),
    )
    ds["AW_geopotential_energy"] = (
        ds["geopotential_energy_column"] * ds["normed_area"]
    ).sum(dim=["ncol"])

    # kinetic energy
    ds["kinetic_energy"] = 0.5 * ds["U"] ** 2 + 0.5 * ds["V"] ** 2
    ds["kinetic_energy_column"] = (
        ("init_time", "lead_time", "ncol"),
        integrate(ds["kinetic_energy"]),
    )
    ds["AW_kinetic_energy"] = (ds["kinetic_energy_column"] * ds["normed_area"]).sum(
        dim=["ncol"]
    )

    # latent heat needs no integration, already column total
    ds["latent_heat_energy"] = Lv * ds["Q"]  # use this for profile plots
    ds["latent_heat_energy_column"] = (
        Lv * ds["tcwv"]
    )  # use this for total column energy
    ds["AW_latent_heat_energy"] = (
        ds["latent_heat_energy_column"] * ds["normed_area"]
    ).sum(dim=["ncol"])

    # total energy
    ds["total_energy"] = (
        ds["sensible_heat_energy"]
        + ds["geopotential_energy"]
        + ds["kinetic_energy"]
        + ds["latent_heat_energy"]
    )
    ds["total_energy_column"] = (
        ds["sensible_heat_energy_column"]
        + ds["geopotential_energy_column"]
        + ds["kinetic_energy_column"]
        + ds["latent_heat_energy_column"]
    )
    ds["AW_total_energy"] = (ds["total_energy_column"] * ds["normed_area"]).sum(
        dim=["ncol"]
    )

    if i == 0:
        # unperturbed
        ds.to_netcdf(unperturbed_processed_e3sm_path, mode="w")
    elif i == 1:
        # perturbed
        ds.to_netcdf(perturbed_processed_e3sm_path, mode="w")
    print(f"Saved dataset {i+1}/2")

# also output the difference at t=0 of these two files for perturbation verification
difference_path = Path("/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/B.energy_balance/data/E3SM_runs/difference/perturbed_minus_original.nc")
diffs = pert_ds[vars_in_ds].isel(time=0) - upert_ds[vars_in_ds].isel(time=0)
diff_ds = upert_ds.isel(time=0).copy()
for var in diffs.data_vars:
    diff_ds[var] = diffs[var]
diff_ds.to_netcdf(difference_path, mode="w")
print(f"Saved initial condition difference dataset to {difference_path}")
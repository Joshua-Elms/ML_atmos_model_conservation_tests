import xarray as xr
from pathlib import Path
import numpy as np
from utils import general, model_info
import metpy
import scipy

config_path = Path(__file__).parent / "0.config.yaml"
config = general.read_config(config_path)

# unperturbed and perturbed E3SM dataset paths
upert_path = Path(config["upert_e3sm_path"])
pert_path = Path(config["pert_e3sm_path"])

upert_ds = xr.open_dataset(upert_path)
pert_ds = xr.open_dataset(pert_path)

# Get column areas for latitude-weighting fields
col_areas = upert_ds["area"]
normed_col_areas = col_areas / col_areas.sum()

# Pressure levels for integration #
model_levels = [level for level in model_info.STANDARD_13_LEVELS if level not in [150, 250]] # 
model_levels_pa = 100 * np.array(model_levels)  # convert to Pa from hPa, used for integration
model_levels_pa_w_units = model_levels_pa * metpy.units.units("Pa")

# Set constants
cp = 1005.0  # J/kg/K
g = 9.81  # m/s^2
Lv = 2.26e6  # J/kg
sb_const = 5.670374419e-8  # W/m^2/K^4, from https://physics.nist.gov/cgi-bin/cuu/Value?sigma

for full_ds in [upert_ds, pert_ds]:
    # Subset relevant variables and rename to E2S standards
    vars_in_ds = (set(map(str.upper, model_info.MASTER_VARIABLES_NAMES)) & set(full_ds.data_vars)) | {"TMQ", "PS", "PHIS", "U050", "V050", "T050", "Z050", "Q050", "lat"}
    e3sm_to_cds = {var: var.lower() for var in vars_in_ds}
    e3sm_to_cds["U050"] = "u50" # 50 hPa zonal wind
    e3sm_to_cds["V050"] = "v50" # 50 hPa meridional wind
    e3sm_to_cds["T050"] = "t50" # 50 hPa temperature
    e3sm_to_cds["Z050"] = "z50" # 50 hPa geopotential height
    e3sm_to_cds["Q050"] = "q50" # 50 hPa specific humidity
    e3sm_to_cds["PS"] = "sp" # surface pressure
    e3sm_to_cds["PHIS"] = "z" # surface geopotential
    e3sm_to_cds["TMQ"] = "tcwv" # total column water vapor
    flat_ds = full_ds[vars_in_ds].rename(e3sm_to_cds) # variables are on levels, e.g. z500 instead of z w/ dim "lev" including 500
    
    # preprocess the data to put T, U, V, Z, Q into blocks by level
    level_blocks = {}
    for var in "tuvzq":
        levels = [level for level in model_levels if f"{var}{level}" in flat_ds]
        level_blocks[var.upper()] = [flat_ds[f"{var}{level}"] for level in levels]
        print(f"{len(levels)} {var} levels found: {levels}")
        
    # combine level blocks into single DataArrays
    for key in level_blocks:
        assert len(level_blocks[key]) == len(model_levels), f"Level block for {key} has {len(level_blocks[key])} levels, expected {len(model_levels)}"
        level_blocks[key] = xr.concat(level_blocks[key], dim="level").assign_coords(level=model_levels)

    # create new dataset with level blocks
    ds = xr.Dataset(level_blocks)
    
    # add 2D variables
    for var in ["sp", "z", "tcwv", "lat"]:
        ds[var] = flat_ds[var]
    ds["normed_area"] = normed_col_areas

    ### Calculate total energy components ###
    # sensible heat
    sensible_heat_energy = cp * ds["T"]
    sensible_heat_energy_column = (1 / g) * scipy.integrate.trapezoid(sensible_heat_energy, model_levels_pa, axis=0)
    
    # geopotential energy
    geopotential_energy = g * ds["Z"] # E3SM provides geopotential height in meters, see full_ds["Z500"].mean().values ~ 5000 (m)
    geopotential_energy_column = (1 / g) * scipy.integrate.trapezoid(geopotential_energy, model_levels_pa, axis=0)
    
    # kinetic energy
    kinetic_energy = 0.5 * ds["U"] ** 2 + 0.5 * ds["V"] ** 2
    kinetic_energy_column = (1 / g) * scipy.integrate.trapezoid(kinetic_energy, model_levels_pa, axis=0)
    
    # latent heat needs no integration, already column total
    latent_heat_energy = Lv * ds["Q"] # use this for profile plots
    latent_heat_energy_column = Lv * ds["tcwv"] # use this for total column energy
    
    # total energy 
    total_energy = sensible_heat_energy + geopotential_energy + kinetic_energy + latent_heat_energy
    total_energy_column = sensible_heat_energy_column + geopotential_energy_column + kinetic_energy_column + latent_heat_energy_column
    
    breakpoint()

   
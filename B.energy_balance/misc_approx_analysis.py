"""
The current draft of the DCMIP 2025 paper's energy balance section relies on two approximations: 

1) Using T_500 as effective radiative temperature (ERT)
2) For short timescales, ISR - OLR = dE/dt

Here we test both of these approximation on ERA5 data to quantify their errors.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cdsapi
from utils.general import latitude_weighted_mean
import scipy
from pathlib import Path

output_directory = Path("/N/scratch/jmelms/era5_feb_2018_13_level/")
output_directory.mkdir(exist_ok=True)
output_fname = lambda var: f"{var}.nc"
standard_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
vars = dict(
    T = {"variable": "temperature", "level": [standard_levels], "dataset": "reanalysis-era5-pressure-levels"},
    U = {"variable": "u_component_of_wind", "level": [standard_levels], "dataset": "reanalysis-era5-pressure-levels"},
    V = {"variable": "v_component_of_wind", "level": [standard_levels], "dataset": "reanalysis-era5-pressure-levels"},
    Q = {"variable": "specific_humidity", "level": [standard_levels], "dataset": "reanalysis-era5-pressure-levels"},
    Z = {"variable": "geopotential", "level": [standard_levels], "dataset": "reanalysis-era5-pressure-levels"},
    ISR = {"variable": "top_net_solar_radiation", "dataset": "reanalysis-era5-single-levels"},
    OLR = {"variable": "top_net_thermal_radiation", "dataset": "reanalysis-era5-single-levels"}
)

### 1. Download ERA5 data for February 2018 at 6-hourly intervals ###
request_template = {
    "product_type": ["reanalysis"],
    "year": ["2018"],
    "month": ["02"],
    "day": [f"{day:02d}" for day in range(1, 29)],
    "time": ["00:00", "06:00", "12:00", "18:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

for var_name, var_info in vars.items():
    request = request_template.copy()
    request["variable"] = long_name = var_info["variable"]
    dataset = var_info["dataset"]
    if dataset == "reanalysis-era5-pressure-levels":
        request["level"] = var_info["level"]

    download_path = output_directory / output_fname(var_name)
    if download_path.exists():
        print(f"File {download_path} already exists, skipping download.")
        continue
    
    print(f"Downloading {var_name} data...")

    client = cdsapi.Client()
    client.retrieve(dataset, request, download_path)

### 2.1 Compute effective radiative temperature (ERT) from T_500 ###
sb_const = 5.670374419e-8  # W/m^2/K^4, from https://physics.nist.gov/cgi-bin/cuu/Value?sigma
ds_T500 = xr.open_dataset(output_directory / output_fname("T")).sel(level=500)
T_500 = ds_T500["temperature"]
MEAN_T_500 = latitude_weighted_mean(T_500, ds_T500.latitude)
OLR_estimated = sb_const * MEAN_T_500**4
OLR_actual = latitude_weighted_mean(xr.open_dataset(output_directory / output_fname("OLR"))["top_net_thermal_radiation"])

### 2.2 Compute total energy at each time step ###
# constants
cp = 1005.0  # J/kg/K
g = 9.81  # m/s^2
Lv = 2.26e6  # J/kg
standard_levels_pa = 100 * np.array(standard_levels)  # convert to Pa from hPa, used for integration

# util functions
open_var = lambda var: xr.open_dataset(output_directory / output_fname(var))
vertically_integrated = lambda da: scipy.integrate.trapezoid(da, standard_levels_pa, axis=0)

# components of energy
ds_T = open_var("T")
sensible_heat_pointwise = cp * ds_T["temperature"]
sensible_heat_column = vertically_integrated(sensible_heat_pointwise) / g
ds_T.close()

ds_U = open_var("U")
ds_V = open_var("V")
kinetic_energy_pointwise = 0.5 * (ds_U["u_component_of_wind"]**2 + ds_V["v_component_of_wind"]**2)
kinetic_energy_column = vertically_integrated(kinetic_energy_pointwise) / g
ds_U.close()
ds_V.close()

ds_Q = open_var("Q")
latent_heat_pointwise = Lv * ds_Q["specific_humidity"]
latent_heat_column = vertically_integrated(latent_heat_pointwise) / g
ds_Q.close()

ds_Z = open_var("Z")
gravitational_potential_energy_pointwise = ds_Z["geopotential"]
gravitational_potential_energy_column = vertically_integrated(gravitational_potential_energy_pointwise) / g
ds_Z.close()

total_energy_column = sensible_heat_column + kinetic_energy_column + latent_heat_column + gravitational_potential_energy_column

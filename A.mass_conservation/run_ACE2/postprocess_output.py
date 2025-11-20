# Imports
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from utils import general
import yaml

# Config
data_path = "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/output/autoregressive_predictions.nc"
ic_path = "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/output/initial_condition.nc"
config_path = "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/model_data/inference_config.yaml"

# Load data
ds = xr.open_dataset(data_path)
ic_ds = xr.open_dataset(ic_path)

# Calculate pressure on levels
ds = ds.rename_vars(
    {
        "TMP2m": "air_temperature_8",
        "Q2m": "specific_total_water_8",
        "UGRD10m": "eastward_wind_8",
        "VGRD10m": "northward_wind_8",
    }
)
init_time_coord = ds["init_time"].values
ds = ds.drop_vars("init_time")
ds = ds.rename_dims({"sample": "init_time"})
ds = ds.rename({"time": "lead_time"})
lead_time_hours = ds["lead_time"].values.astype(int) / 1e9 / 3600  # ns to hours
ds = ds.assign_coords(init_time=init_time_coord, lead_time=lead_time_hours)
levs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ak = xr.DataArray(ic_ds["ak"].values, dims=["level"], coords={"level": levs})
bk = xr.DataArray(ic_ds["bk"].values, dims=["level"], coords={"level": levs})
ps = ds["PRESsfc"]
p_levels = ak + bk * ps

# Q: Integrate specific humidity to get precipitable water and latent (moist) energy
q_vars = [ds[f"specific_total_water_{i}"].assign_attrs(level=i) for i in levs]
qds = xr.concat(q_vars, dim="level")
g = 9.81  # m/s^2
tcw = trapezoid(y=qds, x=p_levels, axis=0) / g
Lv = 2.5e6  # J/kg
column_latent_energy = Lv * tcw
ps_moist = tcw * g
ps_dry = ps - ps_moist

# T: Integrate temperature to get sensible energy
t_vars = [ds[f"air_temperature_{i}"].assign_attrs(level=i) for i in levs]
tds = xr.concat(t_vars, dim="level")
Cp = 1005.0  # J/kg/K
sensible_energy = Cp * tds
column_sensible_energy = trapezoid(y=sensible_energy, x=p_levels, axis=0) / g

# U,V: Integrate wind components to get kinetic energy
u_vars = [ds[f"eastward_wind_{i}"].assign_attrs(level=i) for i in levs]
v_vars = [ds[f"northward_wind_{i}"].assign_attrs(level=i) for i in levs]
uds = xr.concat(u_vars, dim="level")
vds = xr.concat(v_vars, dim="level")
kinetic_energy = 0.5 * (uds**2 + vds**2)  # J/kg
column_kinetic_energy = trapezoid(y=kinetic_energy, x=p_levels, axis=0) / g

# save pressure data
ps = ps.to_dataset(name="sp_total")
ps["sp_moist"] = (ps_dry.dims, ps_moist)
ps["sp_dry"] = ps_dry
ps["sp"] = ps["sp_total"]
ps["MEAN_sp"] = general.latitude_weighted_mean(ps["sp"], ds["lat"])
ps["MEAN_sp_moist"] = general.latitude_weighted_mean(ps["sp_moist"], ds["lat"])
ps = ps.assign_coords(model="ACE2")
ps.to_netcdf(
    "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/output/ACE2_pressure_output.nc"
)
dim_names = ["init_time", "lead_time", "lat", "lon"]
dim_vals = [ds[name].values for name in dim_names]
column_coords = {name: vals for name, vals in zip(dim_names, dim_vals)}
# combine energetic terms into one dataset and save
energy_ds = xr.Dataset(
    {
        "LE": (dim_names, column_latent_energy),
        "SE": (dim_names, column_sensible_energy),
        "KE": (dim_names, column_kinetic_energy),
    },
    coords=column_coords,
)
energy_ds["MEAN_LE"] = general.latitude_weighted_mean(energy_ds["LE"], ds["lat"])
energy_ds["MEAN_SE"] = general.latitude_weighted_mean(energy_ds["SE"], ds["lat"])
energy_ds["MEAN_KE"] = general.latitude_weighted_mean(energy_ds["KE"], ds["lat"])
energy_ds = energy_ds.assign_coords(model="ACE2")
energy_ds.to_netcdf(
    "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/output/ACE2_energy_output.nc"
)

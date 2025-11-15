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
if "Q2m" in ds.variables:
    ds = ds.rename_vars({"Q2m": "specific_total_water_8"})
levs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ak = xr.DataArray(ic_ds["ak"].values, dims=["level"], coords={"level": levs})
bk = xr.DataArray(ic_ds["bk"].values, dims=["level"], coords={"level": levs})
q_vars = [ds[f"specific_total_water_{i}"].squeeze().assign_attrs(level=i) for i in levs]
qds = xr.concat(q_vars, dim="level")
ps = ds["PRESsfc"].squeeze()
p_levels = ak + bk * ps
tcw = trapezoid(y=qds, x=p_levels, axis=0)
ps_moist = tcw
ps_dry = ps - ps_moist
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
ps = ps.to_dataset(name="sp_total")
ps = ps.assign_coords(
    init_time=[
        np.datetime64(t) for t in config["initial_condition"]["start_indices"]["times"]
    ]
)
ps["sp_moist"] = (ps_dry.dims, ps_moist)
ps["sp_moist"] = ps["sp_moist"].expand_dims("init_time")
ps["sp_dry"] = ps_dry
ps["sp_dry"] = ps["sp_dry"].expand_dims("init_time")
ps["sp"] = ps["sp_total"]
ps["sp"] = ps["sp"].expand_dims("init_time")
ps["MEAN_sp"] = general.latitude_weighted_mean(ps["sp"], ds["lat"])
ps["MEAN_sp_moist"] = general.latitude_weighted_mean(ps["sp_moist"], ds["lat"])
ps = ps.rename({"time": "lead_time"})
ps["lead_time"] = (
    ps["lead_time"].values.astype(int) / 1e9 / 3600
)  # ns per s and s per hr, so ns to hr
ps = ps.assign_coords(model="ACE2")
ps.to_netcdf(
    "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/output/ACE2_output.nc"
)

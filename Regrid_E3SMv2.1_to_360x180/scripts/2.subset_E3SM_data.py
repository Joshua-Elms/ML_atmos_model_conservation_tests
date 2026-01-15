"""
Because the regrid command will attempt to regrid all available data in the file we point it to
and we only need the initial condition, we isolate the initial timestep and write it out.
"""

import xarray as xr
import numpy as np
from pathlib import Path


# unperturbed
input_path = Path(
    "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/B.energy_balance/data/E3SM_runs/v2.1.WCYCLSSP370_20180201-20180401_12n32p_original_20260114/run/v2.1.WCYCLSSP370_20180201-20180401_12n32p_original_20260114.eam.h1.2018-02-01-00000.nc"
)
output_path = Path(
    "/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/original_native_grid/E3SM_IC.nc"
)

# set fill values for all vars
ds = xr.open_dataset(input_path).isel(time=0)
coords = list(ds.drop_vars("time").coords)
coord_fill_values = {coord: {"_FillValue": None} for coord in coords}
data_vars = list(ds.data_vars)
data_var_fill_values = {
    data_var: {"_FillValue": -999_999_999} for data_var in data_vars
}
encodings = {
    **coord_fill_values,
    **data_var_fill_values,
}

# subset to first timestep and write to output file
time = ds["time"].item().isoformat()
ds = ds.drop_vars("time")
ds = ds.assign_attrs({"ic_date": time})
ds.to_netcdf(output_path, encoding=encodings)

print(f"Saving initial timestep from {input_path.name}")

# perturbed
input_path = Path(
    "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/B.energy_balance/data/E3SM_runs/v2.1.WCYCLSSP370_20180201-20180401_12n32p_trial_07/run/v2.1.WCYCLSSP370_20180201-20180401_12n32p_trial_07.eam.h1.2018-02-01-00000.nc"
)
output_path = Path(
    "/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/p5k_native_grid/E3SM_IC.nc"
)

# set fill values for all vars
ds = xr.open_dataset(input_path).isel(time=0)
coords = list(ds.drop_vars("time").coords)
coord_fill_values = {coord: {"_FillValue": None} for coord in coords}
data_vars = list(ds.data_vars)
data_var_fill_values = {
    data_var: {"_FillValue": -999_999_999} for data_var in data_vars
}
encodings = {
    **coord_fill_values,
    **data_var_fill_values,
}

# subset to first timestep and write to output file
time = ds["time"].item().isoformat()
ds = ds.drop_vars("time")
ds = ds.assign_attrs({"ic_date": time})
ds.to_netcdf(output_path, encoding=encodings)

print(f"Saving initial timestep from {input_path.name}")

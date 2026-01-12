import xarray as xr
import numpy as np
from pathlib import Path
from utils import model_info
import metpy

# E2S -> E3SM variable conversion table
e2s_to_e3sm = dict(
    u10m="UBOT",
    v10m="VBOT",
    u100m="UBOT",
    v100m="VBOT",
    t2m="TBOT",
    sp="ps",
    msl=None,
    tcwv="TCWV",
    tp06=None,
)
for lev in model_info.STANDARD_13_LEVELS:
    for var in ["u", "v", "t", "q", "r", "z"]:
        e2s_to_e3sm[f"{var}{lev}"] = (
            f"{var.upper()}{lev}"  # E3SM just capitalizes varnames
        )

# choose unperturbed or perturbed run
pert = False

if pert:
    regridded_e3sm_path = Path(
        "/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/p5k_regridded"
    )
else:
    regridded_e3sm_path = Path(
        "/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/original_regridded"
    )

regridded_e3sm_path = regridded_e3sm_path / "E3SM_original_regridded_IC.nc"

# models to create ICs for
models = ["SFNO", "Pangu24"]

# location to save ICs to
output_directory = (
    Path("/N/slate/jmelms/projects/IC/E3SM") / "perturbed" if pert else "unperturbed"
)

### create super-duper IC with all fields
full_ds = xr.open_dataset(regridded_e3sm_path)

# add TCWV field

# add relative humidity (R) field on levels
breakpoint()
for lev in model_info.STANDARD_13_LEVELS:
    p = int(lev) * metpy.units.units("hPa")  # convert hPa to Pa
    t_var = f"T{lev}"
    q_var = f"Q{lev}"
    if t_var in full_ds.data_vars and q_var in full_ds.data_vars:
        T = full_ds[t_var] * metpy.units.units("K")  # K
        q = full_ds[q_var] * metpy.units.units("g/kg")  # %
        # clip below 0. this will cause nan td, so we'll then replace those with 0 q.
        q = q.clip(min=0.0 * metpy.units.units.percent)
        td = metpy.calc.dewpoint_from_specific_humidity(T, q)
        r = metpy.calc.relative_humidity_from_dewpoint(
            p, td, phase="auto"
        ).metpy.magnitude
        q = np.nan_to_num(q, nan=0.0)
        tmp_ds[f"q{level}"] = (tmp_ds[rh_var].dims, q)

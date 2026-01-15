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
    sp="PS",
    msl="PSL",
    tcwv="TMQ",
    tp06="TP06",
)
for lev in model_info.STANDARD_13_LEVELS:
    for var in ["u", "v", "t", "q", "r", "z"]:
        e2s_to_e3sm[f"{var}{lev}"] = (
            f"{var.upper()}{lev:03}"  # E3SM just capitalizes varnames
        )

# choose unperturbed or perturbed run
pert = True

if pert:
    regridded_e3sm_path = Path(
        "/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/p5k_regridded"
    )
else:
    regridded_e3sm_path = Path(
        "/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/original_regridded"
    )

regridded_e3sm_path = regridded_e3sm_path / "E3SM_IC.nc"

# models to create ICs for
models = ["SFNO", "Pangu24"]

# location to save ICs to
output_directory = Path("/N/slate/jmelms/projects/IC/E3SM") / (
    "perturbed" if pert else "unperturbed"
)

### create super-duper IC with all fields
full_ds = xr.open_dataset(regridded_e3sm_path)

# add tp06 field from convective (C) and large-scale (L) precipitation, units m/s -> m/6 hrs
RR = full_ds["PRECC"] + full_ds["PRECL"]
TP06 = RR * 60 * 60 * 6
TP06 = TP06.assign_attrs(
    long_name="Accumulated 6-hourly total precipitation", units="m"
)
full_ds["TP06"] = TP06

# see if we have a problem with extreme values at the poles
tmax = full_ds["T1000"].max().item()
tmin = full_ds["T1000"].min().item()
tmean = full_ds["T1000"].mean().item()
tsd = full_ds["T1000"].std().item()
if max(tmax - tmean, tmean - tmin) > 8 * tsd:
    print(
        f"Warning: Polar anomalies likely; T1000 values 8 SDs greater than mean detected"
    )
    print(f"T1000 Max: {tmax:0.2f} K")
    print(f"T1000 Min: {tmin:0.2f} K")

    # to fix problems in all fields, times, and levels at +/- 90 N, [90, 180, 270] E
    # average the values to the left and right of those points to update them
    fix_coords = [(90, 90), (90, 180), (90, 270), (-90, 90), (-90, 180), (-90, 270)]
    for i, (lat, lon) in enumerate(fix_coords):
        print(
            f"1000 hPa Temperature @ {lat}N {lon}E ({i+1}/6): {full_ds['T1000'].sel(lat=lat, lon=lon):0.3f}"
        )

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

# add relative humidity (R) field on levels
for lev in model_info.STANDARD_13_LEVELS:
    p = int(lev) * metpy.units.units("hPa")  # convert hPa to Pa
    t_var = f"T{lev:03}"
    q_var = f"Q{lev:03}"
    if t_var in full_ds.data_vars and q_var in full_ds.data_vars:
        T = full_ds[t_var] * metpy.units.units("K")  # K
        q = full_ds[q_var] * metpy.units.units("g/g")  #
        # clip below 0. this will cause nan td, so we'll then replace those with 0 q.
        q = q.clip(min=0.0)
        Td = metpy.calc.dewpoint_from_specific_humidity(
            pressure=p, specific_humidity=q
        ).metpy.convert_units("K")
        if (Td > T).sum().values.item() > 0:
            print(
                f"Warning! Nonphysicality detected. Count of cells where Td > T @ {lev} hPa: {(Td > T).sum().values.item()}"
            )
        r = metpy.calc.relative_humidity_from_dewpoint(
            temperature=T, dewpoint=Td, phase="auto"
        ).metpy.magnitude
        if np.isnan(r).sum() > 0:
            print(
                f"Warning! {np.isnan(r).sum()} NaNs detected in relative humidity field @ {lev} hPa."
            )
            print("Exiting program, please debug me")
            exit()
        if (r.min().item() < 0) or (r.max().item() > 1):
            print(
                f"Warning! Relative humidity outside of 0-100% detected @ {lev} hPa, clipping values to correct range"
            )
            r = r.clip(min=0.0, max=1.0)
        full_ds[f"R{lev:03}"] = (full_ds[q_var].dims, r)

for lev in model_info.STANDARD_13_LEVELS:
    for var in "UVZQTR":
        if f"{var}{lev:03}" not in full_ds.data_vars:
            print(f"{var}{lev:03} not found in dataset")

models = [
    "SFNO",
    "Pangu6",
    "Pangu6x",
    "Pangu24",
    "FuXi",
    "FuXiShort",
    "FuXiMedium",
    "FuXiLong",
    "FCN3",
    "FCN",
]
for model in models:
    print(f"Saving {'pert' if pert else 'unpert'} IC for: {model}")
    model_var_names = model_info.MODEL_VARIABLES.get(model)["names"]
    e3sm_names = [e2s_to_e3sm[e2s_name] for e2s_name in model_var_names]
    model_ds = full_ds[e3sm_names]
    model_ds.to_netcdf(output_directory / f"{model}_pert={pert}")

import metpy
from utils import general, model_info
from torch.cuda import mem_get_info
import xarray as xr
from earth2studio.io import XarrayBackend, NetCDF4Backend
from earth2studio.data import CDS
import earth2studio.run as run
import numpy as np
import scipy
from pathlib import Path
import datetime as dt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run_experiment(model_name: str, config_path: str) -> str:
    # read dotenv
    general.load_dotenv(override=True)

    # read config file
    config = general.read_config(config_path)

    print(f"Running experiment for model: {model_name}")
    print(
        f"GPU memory: {mem_get_info()[0] / 1e9:.2f} GB available out of {mem_get_info()[1] / 1e9:.2f} GB"
    )

    # set output paths
    output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
    nc_output_file = output_dir / f"{model_name}_output.nc"

    # load the model
    model = general.load_model(model_name)

    # load the initial condition times
    ic_dates = [
        dt.datetime.strptime(str_date, "%Y-%m-%dT%Hz")
        for str_date in config["ic_dates"]
    ]
    tmp_output_dir = Path(
        config.get("tmp_dir", output_dir)
    )  # if tmp_dir not specified, use output_dir
    tmp_output_dir.mkdir(parents=False, exist_ok=True)
    tmp_output_files = [
        tmp_output_dir
        / f"{model_name}_output_{ic_date.strftime('%Y%m%dT%H')}_tmp_{np.random.randint(10000)}.nc"
        for ic_date in ic_dates
    ]

    # figure out which variables to keep, given that this model may not output
    # all variables requested in config
    model_vars = model_info.MODEL_VARIABLES[model_name]["names"]
    keep_vars = [var for var in config["keep_vars"] if var in model_vars + ["ssp"]]
    req_vars = ["msl", "t2m"] + []

    ds_list = []

    # have to iterate like this to work w/ GraphCastOperational
    for i, ic_date in enumerate(ic_dates):

        fpath = tmp_output_files[i]

        # interface between model and data
        io = NetCDF4Backend(fpath)

        # get ERA5 data from the ECMWF CDS
        data_source = CDS(verbose=False)

        # allows hook func to know whether it's the first call
        global initial
        initial = True

        # allows hook func to find index of pressure fields
        global pressure_indices
        pressure_indices = [
            idx for idx, var in enumerate(model_vars) if var in ["sp", "msl"]
        ]

        # set front hook to pressure multiplier
        def pressure_multiplier(x, coords):
            global initial
            global pressure_indices
            if initial:
                initial = False
                print(f"\n\n Shape of x: {x.shape}\n")
                print(
                    f"Applying pressure multiplier of {config['pressure_multiplier']} to indices {pressure_indices}\n\n"
                )
                x[..., pressure_indices, :, :] *= config["pressure_multiplier"]
                return x, coords
            else:
                return x, coords

        model.front_hook = pressure_multiplier

        # run the model for all initial conditions at once
        run.deterministic(
            time=np.atleast_1d(ic_date),
            nsteps=config["n_timesteps"],
            prognostic=model,
            data=data_source,
            io=io,
            device=config["device"],
        )
        keep_vars_cp = keep_vars.copy()
        extract_vars = list(set(keep_vars_cp + req_vars) - {"ssp"})
        rh_vars = [var for var in model_vars if "r" in var and var[1:].isdigit()]
        q_vars = [var for var in model_vars if "q" in var and var[1:].isdigit()]
        tcw_present = "tcwv" in model_vars
        extract_vars += rh_vars + q_vars
        if tcw_present:
            extract_vars.append("tcwv")
        tmp_ds = xr.open_dataset(fpath)

        # add synthetic SP variable if requested
        if "ssp" in keep_vars_cp:
            assert all(
                var in tmp_ds.data_vars for var in req_vars
            ), f"Missing required variables for ssp calculation: {req_vars}"
            Zs = xr.open_dataset(config["surface_geopotential_path"])["geopotential"]
            g = 9.80665  # m/s^2
            zs = Zs / g  # convert to height in m
            zs = zs.rename({"longitude": "lon", "latitude": "lat"})
            Rd = 287.05  # J/(kg*K)
            T2M = tmp_ds["t2m"]  # K
            p0 = tmp_ds["msl"]  # in Pa
            lapse_rate = 0.0065  # K/m
            exponent = g / (Rd * lapse_rate)
            ssp = p0 * (T2M / (T2M + lapse_rate * zs)) ** exponent
            tmp_ds["ssp"] = ssp


        if config["compute_sp_moist"]:
            print("Computing sp_moist variable.")
            # check for relative humidity variable, compute q if present
            rh_vars = [var for var in model_vars if "r" in var and var[1:].isdigit()]
            q_vars = [var for var in model_vars if "q" in var and var[1:].isdigit()]
            tcw_present = "tcwv" in model_vars
            if tcw_present:
                print(
                    "Total column water (tcwv) present; skipping r -> q, q -> tcw conversion."
                )
                moisture_var = "tcwv"
            elif len(q_vars) >= 2:
                print(
                    "Specific humidity (q) variables already present; skipping r -> q conversion."
                )
                moisture_var = "q"
            elif len(rh_vars) >= 2:
                print(
                    "Relative humidity (r) variables present; computing specific humidity (q) from r."
                )
                moisture_var = "q"
                for rh_var in rh_vars:
                    level = rh_var[1:]
                    p = int(level) * metpy.units.units("hPa")  # convert hPa to Pa
                    t_var = f"t{level}"
                    if t_var in tmp_ds.data_vars and rh_var in tmp_ds.data_vars:
                        T = tmp_ds[t_var] * metpy.units.units("K")  # K
                        r = tmp_ds[rh_var] * metpy.units.units.percent  # %
                        # clip below 0. this will cause nan td, so we'll then replace those with 0 q.
                        r = r.clip(min=0.0 * metpy.units.units.percent)
                        td = metpy.calc.dewpoint_from_relative_humidity(T, r)
                        q = metpy.calc.specific_humidity_from_dewpoint(
                            p, td, phase="auto"
                        ).metpy.magnitude
                        q = np.nan_to_num(q, nan=0.0)
                        tmp_ds[f"q{level}"] = (tmp_ds[rh_var].dims, q)
            else:
                print(
                    "No moisture variable (tcwv, q, or r) present, var sp_moist will be missing."
                )
                moisture_var = None

            if moisture_var == "q":
                print("Computing total column water (tcwv) from specific humidity (q).")
                model_levels = model_info.STANDARD_13_LEVELS
                levs_Pa = np.array(model_levels) * metpy.units.units("hPa")
                q_levels = [level for level in model_levels if f"q{level}" in tmp_ds]
                q_dat = [tmp_ds[f"q{level}"] for level in q_levels]
                Q = xr.concat(q_dat, dim="level").assign_coords(level=model_levels)
                tcwv = (1 / g) * scipy.integrate.trapezoid(Q, levs_Pa, axis=0)
                tmp_ds["tcwv"] = (Q.dims[1:], tcwv)
                moisture_var = "tcwv"

            if moisture_var is None:
                print("Moisture variable not available; skipping sp_moist calculation.")

            elif moisture_var == "tcwv":
                print(
                    "Using total column water (tcwv) for moisture variable in sp_moist calculation."
                )
                tmp_ds["sp_moist"] = g * tmp_ds["tcwv"]  # in Pa
                keep_vars_cp.append("sp_moist")

            tmp_ds = tmp_ds[keep_vars_cp]

        # sort by latitude
        tmp_ds = general.sort_latitudes(tmp_ds, model_name, input=False)

        # postprocess data
        for var in keep_vars_cp:
            if var in ["msl", "ssp", "sp"]:
                print(f"Model: {model_name}")
                print(
                    f"Mean before: {var} at lead_time=0: {tmp_ds[var].isel(lead_time=0).values.mean()}"
                )
                # multi-step inference models show the perturbation in IC by default (I predict, unverified for GraphCastOperational)
                if model_name not in ["FuXi", "FuXiShort", "GraphCastOperational"]:
                    tmp_ds[var][dict(lead_time=0)] = (
                        tmp_ds[var].isel(lead_time=0) * config["pressure_multiplier"]
                    )
                print(
                    f"Mean after: {var} at lead_time=0: {tmp_ds[var].isel(lead_time=0).values.mean()}"
                )
            tmp_ds[f"MEAN_{var}"] = general.latitude_weighted_mean(
                tmp_ds[var], tmp_ds.lat
            )

        # if storage is tight, drop the full 3D fields
        if config.get("keep_base_fields", True) is False:
            tmp_ds = tmp_ds.drop_vars(keep_vars_cp)

        # overwrite temporary file
        fpath.unlink()
        tmp_ds.to_netcdf(fpath, mode="w")

    # combine all temporary files into one dataset, not using openmf because it's slow
    ds_list = [xr.open_dataset(file) for file in tmp_output_files]
    ds = xr.concat(ds_list, dim="time")
    for tmp_file in tmp_output_files:
        tmp_file.unlink()  # delete temporary file
    print(f"Combined dataset has dimensions: {ds.dims}")

    # add model dimension to enable opening with open_mfdataset
    ds = ds.assign_coords(model=model_name)

    # for clarity
    ds = ds.rename({"time": "init_time"})

    # save data
    ds.to_netcdf(nc_output_file)


if __name__ == "__main__":
    general.run_experiment_controller(
        calling_directory=Path(__file__).parent,
        run_experiment=run_experiment,
        config_path=Path(__file__).parent / "0.config.yaml",
    )

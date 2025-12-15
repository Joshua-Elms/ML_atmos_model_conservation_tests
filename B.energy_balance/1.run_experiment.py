from utils import general, model_info
from torch.cuda import mem_get_info
import xarray as xr
from earth2studio.io import XarrayBackend, NetCDF4Backend
from earth2studio.data import CDS
import earth2studio.run as run
import numpy as np
from pathlib import Path
import datetime as dt
import metpy
import scipy
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run_experiment(model_name: str, config_path: str) -> str:
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
    model_vars = model_info.MODEL_VARIABLES[model_name]["names"]

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
    
    # load surface geopotential height data if needed
    zs_path = Path(config["surface_geopotential_path"])
    zs = xr.open_dataset(zs_path)["geopotential"]

    # set indices of variables to perturb
    all_temp_vars = ["t2m"] + [f"t{level}" for level in model_info.STANDARD_13_LEVELS]
    pert_vars = [var for var in all_temp_vars if var in model_vars]
    global pert_var_idxs
    pert_var_idxs = [model_vars.index(var) for var in pert_vars]

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

        # set front hook to temperature perturber
        def temp_perturber(x, coords):
            global initial
            global pert_var_idxs
            if initial:
                initial = False
                print(f"\n\n Shape of x: {x.shape}\n")
                print(
                    f"Applying temperature perturbation to indices {pert_var_idxs}\n\n"
                )
                x[..., pert_var_idxs, :, :] += config["temp_perturbation_degC"]
                return x, coords
            else:
                return x, coords

        model.front_hook = temp_perturber

        # run the model for all initial conditions at once
        run.deterministic(
            time=np.atleast_1d(ic_date),
            nsteps=config["n_timesteps"],
            prognostic=model,
            data=data_source,
            io=io,
            device=config["device"],
        )

        tmp_ds = xr.open_dataset(fpath)

        # sort by latitude
        tmp_ds = general.sort_latitudes(tmp_ds, model_name, input=False)

        # this exists because the inference code outputs the IC before the front hook (so no perturbation applied)
        if model_name not in ["FuXi", "FuXiShort", "GraphCastOperational"]:
            for var in pert_vars:
                tmp_ds[var][dict(lead_time=0)] = (
                    tmp_ds[var].isel(lead_time=0) + config["temp_perturbation_degC"]
                )

        ### calculate energetics
        # preprocess the data to put T, U, V, Z, Q into blocks
        model_levels = model_info.STANDARD_13_LEVELS
        ### Get pressure for integration ###
        model_levels_pa = 100 * np.array(
            model_levels
        )  # convert to Pa from hPa, used for integration
        g = 9.81  # m/s^2
        
        # need to add, model_levels_pa Z field so that I can integrate w/r/t surface
        Z_block = {}
        print("Collecting geopotential height (Z) field for energy calculations.")
        for var in "z":
            levels = [level for level in model_levels if f"{var}{level}" in tmp_ds]
            if levels:
                Z_block[var.upper()] = [
                    tmp_ds[f"{var}{level}"] for level in levels
                ]
            else:
                print(f"Skipping {var} because no levels found.")
            print(f"{len(levels)} {var} levels found: {levels}")

        # combine level blocks into single DataArrays
        for key in Z_block:
            assert len(Z_block[key]) == len(
                model_levels
            ), f"Level block for {key} has {len(Z_block[key])} levels, expected {len(model_levels)}"
            Z_block[key] = xr.concat(Z_block[key], dim="level").assign_coords(
                level=model_levels
            )
        Z = Z_block["Z"]

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
                "No moisture variable (tcwv, q, or r) present, latent heat will not be calculated."
            )
            moisture_var = None

        if moisture_var == "q":
            print("Computing total column water (tcwv) from specific humidity (q).")
            q_levels = [level for level in model_levels if f"q{level}" in tmp_ds]
            q_dat = [tmp_ds[f"q{level}"] for level in q_levels]
            Q = xr.concat(q_dat, dim="level").assign_coords(level=model_levels)
            print(general.surface_aware_integrate.__doc__)
            tcwv = (1 / g) * general.surface_aware_integrate(Q, Z/g, zs/g, model_levels_pa)
            tmp_ds["tcwv"] = (Q.dims[1:], tcwv)
            moisture_var = "tcwv"

        level_blocks = {}
        for var in "tuvzq":
            levels = [level for level in model_levels if f"{var}{level}" in tmp_ds]
            if levels:
                level_blocks[var.upper()] = [
                    tmp_ds[f"{var}{level}"] for level in levels
                ]
            else:
                print(f"Skipping {var} because no levels found.")
            print(f"{len(levels)} {var} levels found: {levels}")

        # combine level blocks into single DataArrays
        for key in level_blocks:
            if key == "TCW":
                continue  # TCW is already column-integrated
            assert len(level_blocks[key]) == len(
                model_levels
            ), f"Level block for {key} has {len(level_blocks[key])} levels, expected {len(model_levels)}"
            level_blocks[key] = xr.concat(level_blocks[key], dim="level").assign_coords(
                level=model_levels
            )

        # create new dataset from level blocks
        ds_3d = xr.Dataset(level_blocks)
        ds_3d["tcwv"] = tmp_ds["tcwv"]

        # Set constants
        cp = 1005.0  # J/kg/K
        Lv = 2.5e6  # J/kg
        sb_const = 5.670374419e-8  # W/m^2/K^4, from https://physics.nist.gov/cgi-bin/cuu/Value?sigma

        ### Calculate total energy components ###
        integrate = lambda da: (1 / g) * scipy.integrate.trapezoid(
            da, model_levels_pa, axis=0
        )
        # sensible heat
        ds_3d["sensible_heat_energy"] = cp * ds_3d["T"]
        if "t2m" in model_vars:
            sfc_sensible_heat = cp * tmp_ds["t2m"]
        else:
            sfc_sensible_heat = None
        ds_3d["sensible_heat_energy_column"] = (("time", "lead_time", "lat", "lon"), (1/g)*general.surface_aware_integrate(ds_3d["sensible_heat_energy"], ds_3d["Z"]/g, zs/g, model_levels_pa, sfc_sensible_heat))
        ds_3d["AW_sensible_heat_energy"] = general.latitude_weighted_mean(
            ds_3d["sensible_heat_energy_column"], tmp_ds.lat
        )
        # geopotential energy -- already in J/kg, no need to multiply by g
        ds_3d["geopotential_energy"] = ds_3d["Z"]
        ds_3d["geopotential_energy_column"] = (("time", "lead_time", "lat", "lon"), (1/g)*general.surface_aware_integrate(ds_3d["geopotential_energy"], ds_3d["Z"]/g, zs/g, model_levels_pa))
        ds_3d["AW_geopotential_energy"] = general.latitude_weighted_mean(
            ds_3d["geopotential_energy_column"], tmp_ds.lat
        )
        # kinetic energy
        ds_3d["kinetic_energy"] = 0.5 * ds_3d["U"] ** 2 + 0.5 * ds_3d["V"] ** 2
        if "u10m" in model_vars and "v10m" in model_vars:
            sfc_kinetic_energy = 0.5 * tmp_ds["u10m"] ** 2 + 0.5 * tmp_ds["v10m"] ** 2
        else:
            sfc_kinetic_energy = None
        ds_3d["kinetic_energy_column"] = (("time", "lead_time", "lat", "lon"), (1/g)*general.surface_aware_integrate(ds_3d["kinetic_energy"], ds_3d["Z"]/g, zs/g, model_levels_pa, sfc_kinetic_energy))
        ds_3d["AW_kinetic_energy"] = general.latitude_weighted_mean(
            ds_3d["kinetic_energy_column"], tmp_ds.lat
        )
        # latent heat
        ds_3d["latent_heat_energy"] = Lv * ds_3d["Q"]
        ds_3d["latent_heat_energy_column"] = Lv * ds_3d["tcwv"]
        ds_3d["AW_latent_heat_energy"] = general.latitude_weighted_mean(
            ds_3d["latent_heat_energy_column"], tmp_ds.lat
        )

        # total energy
        ds_3d["total_energy"] = (
            ds_3d["sensible_heat_energy"]
            + ds_3d["geopotential_energy"]
            + ds_3d["kinetic_energy"]
            + ds_3d["latent_heat_energy"]
        )
        ds_3d["total_energy_column"] = (
            ds_3d["sensible_heat_energy_column"]
            + ds_3d["geopotential_energy_column"]
            + ds_3d["kinetic_energy_column"]
            + ds_3d["latent_heat_energy_column"]
        )
        ds_3d["AW_total_energy"] = general.latitude_weighted_mean(
            ds_3d["total_energy_column"], tmp_ds.lat
        )

        # if storage is tight, drop the full 3D fields
        energy_varnames = [
            "sensible_heat_energy",
            "geopotential_energy",
            "kinetic_energy",
            "latent_heat_energy",
            "total_energy",
            "sensible_heat_energy_column",
            "geopotential_energy_column",
            "kinetic_energy_column",
            "latent_heat_energy_column",
            "total_energy_column",
            "AW_sensible_heat_energy",
            "AW_geopotential_energy",
            "AW_kinetic_energy",
            "AW_latent_heat_energy",
            "AW_total_energy",
        ]
        if config.get("keep_base_fields", True) is False:
            ds_3d = ds_3d[energy_varnames]

        # overwrite temporary file
        fpath.unlink()
        ds_3d.to_netcdf(fpath, mode="w")

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

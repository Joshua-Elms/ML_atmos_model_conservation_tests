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

        if model_name not in ["FuXi", "FuXiShort", "GraphCastOperational"]:
            for var in pert_vars:
                tmp_ds[var][dict(lead_time=0)] = (
                    tmp_ds[var].isel(lead_time=0) + config["temp_perturbation_degC"]
                )

        ### calculate energetics
        # preprocess the data to put T, U, V, Z, Q into blocks
        model_levels = model_info.STANDARD_13_LEVELS
        g = 9.81  # m/s^2
        level_blocks = {}

        for var in "tuvz":
            levels = [level for level in model_levels if f"{var}{level}" in tmp_ds]
            level_blocks[var.upper()] = [tmp_ds[f"{var}{level}"] for level in levels]
            print(f"{len(levels)} {var} levels found: {levels}")

        # # figure out whether to use tcwv, q, or r for moisture
        # moisture_var = "tcw" if "tcwv" in model_vars else None
        # # if tcwv, use that
        # if moisture_var == "tcw":
        #     level_blocks["TCW"] = [tmp_ds[moisture_var]]
        # # otherwise, check for q or r
        # else:
        #     for var in model_vars:
        #         if var.startswith("q") and int(var[1:]) in model_levels:
        #             moisture_var = "q"
        #             levels = [level for level in model_levels if f"q{level}" in tmp_ds]
        #             level_blocks[var[0].upper()] = [
        #                 tmp_ds[f"q{level}"] for level in levels
        #             ]
        #             print(f"{len(levels)} q levels found: {levels}")
        #             break
        #         elif var.startswith("r") and int(var[1:]) in model_levels:
        #             moisture_var = "r"
        #             levels = [level for level in model_levels if f"r{level}" in tmp_ds]
        #             level_blocks[var[0].upper()] = [
        #                 tmp_ds[f"r{level}"] for level in levels
        #             ]
        #             print(f"{len(levels)} r levels found: {levels}")
        #             break
        #     # if still none, raise error
        #     else:
        #         raise ValueError(
        #             f"No suitable moisture variable found for energy calculation in model {model_name}."
        #         )
                
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
            model_levels = model_info.STANDARD_13_LEVELS
            levs_Pa = np.array(model_levels) * metpy.units.units("hPa")
            q_levels = [level for level in model_levels if f"q{level}" in tmp_ds]
            q_dat = [tmp_ds[f"q{level}"] for level in q_levels]
            Q = xr.concat(q_dat, dim="level").assign_coords(level=model_levels)
            tcwv = (1 / g) * scipy.integrate.trapezoid(Q, levs_Pa, axis=0)
            tmp_ds["tcwv"] = (Q.dims[1:], tcwv)
            moisture_var = "tcwv"


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

        # Set constants
        cp = 1005.0  # J/kg/K
        Lv = 2.5e6  # J/kg
        sb_const = 5.670374419e-8  # W/m^2/K^4, from https://physics.nist.gov/cgi-bin/cuu/Value?sigma

        ### Step 4b: Get pressure for integration ###
        model_levels_pa = 100 * np.array(
            model_levels
        )  # convert to Pa from hPa, used for integration
        model_levels_pa_w_units = model_levels_pa * metpy.units.units("Pa")

        ### Step 4c: Calculate total energy components ###
        # sensible heat
        sensible_heat = cp * level_blocks["T"]
        # geopotential energy
        geopotential_energy = level_blocks[
            "Z"
        ]  # geopotential energy is already in J/kg, no need to multiply by g
        # kinetic energy
        kinetic_energy = 0.5 * level_blocks["U"] ** 2 + 0.5 * level_blocks["V"] ** 2
        # latent heat
        latent_heat = Lv * tmp_ds["tcwv"] 
        # if moisture_var == "TCW":
        #     latent_heat = Lv * level_blocks["TCW"]
        # elif moisture_var in ["q", "r"]:
        #     if moisture_var == "r":
        #         for rh_var in rh_vars
        #         print(
        #             "Warning: Using 'r' for moisture content in latent heat calculation, converting to q"
        #         )
        #         breakpoint()
        #         T = tmp_ds[t_var] * metpy.units.units("K")  # K
        #         r = tmp_ds[rh_var] * metpy.units.units.percent  # %
        #         # clip below 0. this will cause nan td, so we'll then replace those with 0 q.
        #         r = r.clip(min=0.0 * metpy.units.units.percent)
        #         td = metpy.calc.dewpoint_from_relative_humidity(T, r)
        #         q = metpy.calc.specific_humidity_from_dewpoint(
        #             p, td, phase="auto"
        #         ).metpy.magnitude
        #         q = np.nan_to_num(q, nan=0.0)
        #         tmp_ds[f"q{level}"] = (tmp_ds[rh_var].dims, q)
        #         level_blocks["Q"] = q
        #     latent_heat = Lv * level_blocks["Q"]

        ### Step 4d: Make dataset with each term as a variable
        integrate = lambda da: (1 / g) * scipy.integrate.trapezoid(
            da, model_levels_pa, axis=0
        )
        energy_vars_dict = {
            "sensible_heat_energy": sensible_heat,
            "geopotential_energy": geopotential_energy,
            "kinetic_energy": kinetic_energy,
            "latent_heat_energy": latent_heat,
        }
        for var in [
            "sensible_heat_energy",
            "geopotential_energy",
            "kinetic_energy",
            "latent_heat_energy",
            "total_energy",
        ]:
            if var != "total_energy":
                tmp_ds[var] = (
                        ("time", "lead_time", "lat", "lon"),
                        integrate(energy_vars_dict[var]),
                    )
                # if var == "latent_heat_energy" and moisture_var == "TCW":
                #     tmp_ds[var] = (
                #         ("time", "lead_time", "lat", "lon"),
                #         latent_heat.values,
                #     )
                # else:
                #     tmp_ds[var] = (
                #         ("time", "lead_time", "lat", "lon"),
                #         integrate(energy_vars_dict[var]),
                #     )
            else:
                tmp_ds["total_energy"] = (
                    tmp_ds["sensible_heat_energy"]
                    + tmp_ds["geopotential_energy"]
                    + tmp_ds["kinetic_energy"]
                    + tmp_ds["latent_heat_energy"]
                )
            tmp_ds[var].assign_attrs({"units": "J/m^2"})
            tmp_ds[f"LW_{var}"] = general.latitude_weighted_mean(
                tmp_ds[var], tmp_ds.lat
            )

        # if storage is tight, drop the full 3D fields
        energy_varnames = [
            "sensible_heat_energy",
            "geopotential_energy",
            "kinetic_energy",
            "latent_heat_energy",
            "total_energy",
            "LW_sensible_heat_energy",
            "LW_geopotential_energy",
            "LW_kinetic_energy",
            "LW_latent_heat_energy",
            "LW_total_energy",
        ]
        if config.get("keep_base_fields", True) is False:
            tmp_ds = tmp_ds[energy_varnames]

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

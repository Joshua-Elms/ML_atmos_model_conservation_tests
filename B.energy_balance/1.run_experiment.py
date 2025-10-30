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
    tmp_output_dir = Path(config.get("tmp_dir", output_dir)) # if tmp_dir not specified, use output_dir
    tmp_output_dir.mkdir(parents=False, exist_ok=True)
    tmp_output_files = [tmp_output_dir / f"{model_name}_output_{ic_date.strftime('%Y%m%dT%H')}_tmp_{np.random.randint(10000)}.nc" for ic_date in ic_dates]

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
                print(f"Applying temperature perturbation to indices {pert_var_idxs}\n\n")
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
                tmp_ds[var][dict(lead_time=0)] = tmp_ds[var].isel(lead_time=0) + config["temp_perturbation_degC"]
        
        ### calculate energetics
        # preprocess the data to put T, U, V, Z, Q into blocks
        model_levels = model_info.STANDARD_13_LEVELS
        level_blocks = {}
        
        for var in "tuvz":
            levels = [level for level in model_levels if f"{var}{level}" in tmp_ds]
            level_blocks[var.upper()] = [tmp_ds[f"{var}{level}"] for level in levels]
            print(f"{len(levels)} {var} levels found: {levels}")
            
        # figure out whether to use tcwv, q, or r for moisture
        moisture_var = None
        # if tcwv, use that
        if moisture_var is None:
            for var in model_vars:
                if var.startswith("tcw"):
                    moisture_var = var
                    level_blocks["TCW"] = [tmp_ds[moisture_var]]
                    continue
        # second choice: q if available
        elif moisture_var is None:
            for var in model_vars:
                if var.startswith("q") and int(var[1:]) in model_levels:
                    moisture_var = "q"
                    levels = [level for level in model_levels if f"q{level}" in tmp_ds]
                    level_blocks[var.upper()] = [tmp_ds[f"q{level}"] for level in levels]
                    print(f"{len(levels)} q levels found: {levels}")
                    continue
        # third choice: r if available
        elif moisture_var is None:
            for var in model_vars:
                if var.startswith("r") and int(var[1:]) in model_levels:
                    moisture_var = "r"
                    levels = [level for level in model_levels if f"r{level}" in tmp_ds]
                    level_blocks[var.upper()] = [tmp_ds[f"r{level}"] for level in levels]
                    print(f"{len(levels)} r levels found: {levels}")
                    continue
        # if still none, raise error
        else:
            raise ValueError(f"No suitable moisture variable found for energy calculation in model {model_name}.")

        # combine level blocks into single DataArrays
        for key in level_blocks:
            if key == "TCW":
                continue  # TCW is already column-integrated
            assert len(level_blocks[key]) == len(model_levels), f"Level block for {key} has {len(level_blocks[key])} levels, expected {len(model_levels)}"
            level_blocks[key] = xr.concat(level_blocks[key], dim="level").assign_coords(level=model_levels)
            
        # Set constants
        cp = 1005.0  # J/kg/K
        g = 9.81  # m/s^2
        Lv = 2.26e6  # J/kg
        sb_const = 5.670374419e-8  # W/m^2/K^4, from https://physics.nist.gov/cgi-bin/cuu/Value?sigma

        ### Step 4b: Get pressure for integration ###
        model_levels_pa = 100 * np.array(model_levels)  # convert to Pa from hPa, used for integration
        model_levels_pa_w_units = model_levels_pa * metpy.units.units("Pa")

        ### Step 4c: Calculate total energy components ###
        # sensible heat
        sensible_heat = cp * level_blocks["T"]
        # geopotential energy
        geopotential_energy = level_blocks["Z"] # geopotential energy is already in J/kg, no need to multiply by g
        # kinetic energy
        kinetic_energy = 0.5 * level_blocks["U"] ** 2 + 0.5 * level_blocks["V"] ** 2
        # latent heat
        if moisture_var == "TCW":
            latent_heat = Lv * level_blocks["TCW"]
        elif moisture_var in ["q", "r"]:
            if moisture_var == "r":
                print("Warning: Using 'r' for moisture content in latent heat calculation, converting to q")
                td = metpy.calc.dewpoint_from_relative_humidity(level_blocks["T"], level_blocks["R"])
                q = metpy.calc.specific_humidity_from_dewpoint(model_levels_pa_w_units, td, phase="auto").to("kg/kg").magnitude
                level_blocks["Q"] = q
            latent_heat = Lv * level_blocks["Q"]

        ### Step 4d: Calculate total energy by adding all components ###
        # total energy w/ or w/o moisture depending on variable availability
        precursor_total_energy = sensible_heat + geopotential_energy + kinetic_energy
        if moisture_var in ["q", "r"]:
            precursor_total_energy += latent_heat
        # column integration
        print(f"Integrating precursor total energy with shape {precursor_total_energy.shape} and pa with shape {model_levels_pa.shape}")
        precursor_total_energy_column = (1 / g) * scipy.integrate.trapezoid( # add nan-safe version
            precursor_total_energy, model_levels_pa, axis=0
        )
        if moisture_var == "TCW":
            total_energy_column = precursor_total_energy_column + latent_heat.values
        else:
            total_energy_column = precursor_total_energy_column

        # sum
        tmp_ds["te"] = (("time", "lead_time", "lat", "lon"), total_energy_column)
        tmp_ds["te"].assign_attrs(
            {"units": "J/m^2", "long_name": "Total Energy"}
        )

        ### Step 4e: Weight by latitude ####
        # get latitude weighted total energy (time, ensemble)
        tmp_ds["lw_te"] = general.latitude_weighted_mean(tmp_ds["te"], tmp_ds.lat)
        tmp_ds["lw_te"].assign_attrs(
            {"units": "J/m^2", "long_name": "Latitude-Weighted Total Energy"}
        )
            
        # if storage is tight, drop the full 3D fields
        if config.get("keep_base_fields", True) is False:
            tmp_ds = tmp_ds[["te", "lw_te"]] # keep only total energy and latitude-weighted total energy

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
        config_path=Path(__file__).parent / "0.config.yaml"
    )

from utils import model_info
from earth2studio.utils.type import TimeArray, VariableArray
from earth2studio.models.px.base import PrognosticModel
from earth2studio.data.utils import prep_data_array
from earth2studio.io import XarrayBackend
import earth2studio.run as run
import earth2studio.models.px
import xarray as xr
import numpy as np
import torch
import yaml
import shutil
import subprocess
import datetime as dt
from pathlib import Path
from datetime import datetime
from time import perf_counter
from dotenv import load_dotenv
from collections.abc import Callable


class DataSet:
    """An in-memory xarray dataset data source.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray dataset to use as data source.
    """

    def __init__(self, dataset: xr.Dataset, model_name: str):
        self.ds = dataset
        self.model_name = model_name

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Loaded data array
        """
        # check to make sure requested times and variables are in dataset
        if isinstance(time, datetime):
            time = [time]
        for t in time:
            if t not in self.ds["time"]:
                raise ValueError(f"Requested time {t} not in dataset time coordinate.")
        if isinstance(variable, str):
            variable = [variable]
        for v in variable:
            if v not in self.ds:
                raise ValueError(f"Requested variable {v} not in dataset variables.")

        # loop over variables and concatenate the data arrays
        da_list = [self.ds[v].sel(time=np.atleast_1d(time)) for v in variable]
        da = xr.concat(da_list, dim="variable")
        da = da.assign_coords(variable=variable)
        da = da.transpose("time", "variable", "lat", "lon")
        return da


def read_config(config_path: Path) -> dict:
    """Read the configuration file."""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Check for required keys
    required_keys = [
        "experiment_dir",
        "experiment_name",
        "device",
        "n_timesteps",
        "models",
        "experiment_subdirectories",
        "debug_run",
        "overwrite",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(
                f"Missing required key in config: {key}\nPlease set it and try again."
            )

    # add config path to config object for reference in downstream functions
    if "config_path" not in config:
        config["config_path"] = config_path

    return config


def prepare_output_directory(config: dict) -> Path:
    """Prepare the output directory for an experiment."""
    # set up directories
    exp_dir = (
        Path(config["experiment_dir"]) / config["experiment_name"]
    )  # all data for experiment stored here

    if exp_dir.exists():
        if config["overwrite"]:
            print(f"Experiment directory '{exp_dir}' already exists. Overwriting.")
            shutil.rmtree(exp_dir)
        else:
            raise FileExistsError(
                f"Experiment directory '{exp_dir}' already exists. 'overwrite' set to False, so please delete it or change experiment_name."
            )

    exp_dir.mkdir(parents=True, exist_ok=False)  # make dir if it doesn't exist
    if "experiment_subdirectories" in config:
        for subdir in config["experiment_subdirectories"]:
            (
                exp_dir / subdir
            ).mkdir()  # create subdirectories if specified in config, e.g. "plots", "tendencies"

    # copy config to experiment directory
    config_path_exp = exp_dir / "config.yaml"
    shutil.copy(config["config_path"], config_path_exp)

    # let user know where to find config
    print(f"Ready for experiment output at '{exp_dir}'.")

    return exp_dir


def load_model(model_name: str) -> PrognosticModel:
    """Load a model by name. Currently loads default model weights from cache, or downloads them to cache if not present."""
    load_dotenv()
    if model_name not in model_info.SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not supported. Supported models are: {model_info.SUPPORTED_MODELS}."
        )
    model_class = getattr(earth2studio.models.px, model_name)
    start = perf_counter()
    print(f"Loading model '{model_name}'...")
    package = model_class.load_default_package()
    model = model_class.load_model(package)
    end = perf_counter()
    print(f"Model '{model_name}' loaded in {end - start:.2f} seconds.")
    return model


def run_experiment_controller(
    calling_directory: Path,
    run_experiment: Callable[[str, str], None],
    config_path: Path,
) -> None:
    """Code to orchestrate running an experiment for multiple models. This
    function is useful both for initializing an experiment config file and
    for running each model separately via subprocess, which prevents the GPU
    memory from being overloaded by models failing to release memory after use.

    Parameters
    ----------
    calling_directory : Path
        The directory from which the experiment is being called.
    run_experiment : Callable[[str, str], None]
        The function to run the experiment.
    config_path : Path
        The path to the experiment configuration file.

    Returns
    -------
    None
    """
    # load config file
    config = read_config(config_path)
    print(f"Experiment name:'{config['experiment_name']}'")
    print(f"Models: {config['models']}")

    # get ready to output data to disk
    exp_dir = prepare_output_directory(config)

    # see whether debug run or full run
    if config["debug_run"]:
        print("Running in mode: debug")
        if len(config["models"]) != 1:
            raise ValueError(
                "In debug mode, only one model can be run at a time. Please set 'debug_run' to False or choose a single model in the config. Exiting."
            )
        status = run_experiment(config["models"][0], str(config_path.resolve()))

    else:
        print("Running in mode: full")
        # loop over models and run the experiment for each
        for model_name in config["models"]:
            # code in third arg from: https://stackoverflow.com/questions/27189044/import-with-dot-name-in-python
            subprocess.run(
                [
                    "python",
                    "-c",
                    f"""
import importlib.util;
spec = importlib.util.spec_from_file_location(
    name='run_experiment_pyfile',
    location='{str((calling_directory / '1.run_experiment.py').resolve())}'
);
module = importlib.util.module_from_spec(spec);
spec.loader.exec_module(module);
module.run_experiment('{model_name}', '{str(config_path.resolve())}');
""",
                ],
                check=True,
                cwd=calling_directory,
            )

    print(f"Experiment completed. Results written to {exp_dir}")


def run_deterministic_w_perturbations(
    run_kwargs: dict,
    tendency_reversion: bool,
    model_name: str,
    tendency_file: Path = None,
    initial_perturbation: xr.Dataset = None,
    recurrent_perturbation: xr.Dataset = None,
) -> xr.Dataset:
    """Run a deterministic forecast with tendency reversion.
    Code modified from Travis O'Brien."""

    # set up a hook function that returns the input as is
    def identity(x, coords):
        """Returns the input as is"""
        return x, coords

    # set up a post-model hook function that reverts the tendency
    def tendency_reversion_without_user_rpert(x, coords):
        """Reverts the tendency to the first state. This horrendous-looking return is necessary because
        tend could either be a Tensor of the correct shape to be added to x, or as an integer (probably 0)
        which represents the parent inference function being called with tendency_reversion = False.
        For some reason, if the return statement is unwrapped into normal code w/ variable assignments,
        there's an unbounded variable error coming from tend."""
        return (
            x.to(run_kwargs["device"])
            + (tend if isinstance(tend, int) else tend.to(run_kwargs["device"])),
            coords,
        )

    # set up a post-model hook function that reverts the tendency & adds a recurrent perturbation
    def tendency_reversion_with_user_rpert(x, coords):
        """Reverts the tendency to the first state and adds a user-selected perturbation. This horrendous-looking
        return is necessary because tend could either be a Tensor of the correct shape to be added
        to x, or as an integer (probably 0) which represents the parent inference function being
        called with tendency_reversion = False. For some reason, if the return statement is unwrapped
        into normal code w/ variable assignments, there's an unbounded variable error coming from tend.
        The foregoing comments apply equally to rpert_from_user."""
        return (
            (
                x.to(run_kwargs["device"])
                + (tend if isinstance(tend, int) else tend.to(run_kwargs["device"]))
                + (
                    rpert_from_user
                    if isinstance(rpert_from_user, int)
                    else rpert_from_user.to(run_kwargs["device"])
                )
            ),
            coords,
        )

    if tendency_reversion:
        print("Running deterministic forecast with tendency reversion.")
        dummy_io = XarrayBackend()
        model = run_kwargs["prognostic"]

        # set up a hook function that appends the model state to a list
        states = []

        def append_state(x, coords):
            """Appends the states to a list."""
            states.append(x.clone().cpu())
            return x, coords

        model.rear_hook = append_state
        model.front_hook = append_state

        # run the model for one step; this will populate the states[] list above
        keep_kwargs = {k: v for k, v in run_kwargs.items() if k not in ("nsteps", "io")}
        run.deterministic(**keep_kwargs, nsteps=1, io=dummy_io)

        if tendency_file:
            print(f"Saving tendency to {tendency_file}.")
            tendency_ds = dummy_io.root.isel(lead_time=0) - dummy_io.root.isel(
                lead_time=1
            )
            tendency_ds.to_netcdf(tendency_file)

        # tendencies that will be used
        tend = states[0] - states[1]

        # set the tendency of static variables to 0
        idx = np.nonzero(np.array(model_info.MODEL_VARIABLES[model_name]["types"]) == model_info.IN)[0]
        if len(idx) > 0:
            tend[..., idx, :, :] = 0

    else:
        tend = 0

    # get the recurrent perturbation
    if recurrent_perturbation is not None:
        if not isinstance(recurrent_perturbation, xr.Dataset):
            raise TypeError(
                f"Expected xr.Dataset for recurrent_perturbation, got {type(recurrent_perturbation)}"
            )
        recurrent_pert_source = DataSet(recurrent_perturbation, model_name)
        variables = model.input_coords()["variable"]
        da = recurrent_pert_source(run_kwargs["time"], variables)
        rpert_from_user, rpert_coords = prep_data_array(da)
    else:
        rpert_from_user = 0

    if tendency_reversion:

        # reset the model hooks
        model.front_hook = identity
        model.rear_hook = tendency_reversion_without_user_rpert

        # run validation step
        test_TR_io = XarrayBackend()
        test_TR_ds = run.deterministic(**keep_kwargs, nsteps=1, io=test_TR_io).root

        # print diagnostic information for tendency
        ds_diff = test_TR_ds.diff(dim="lead_time")
        print(
            f"Tendency reversion applied. Difference between 0th and 1st lead time should be close to zero:"
        )

        summary = []
        line = []
        for i, (var_name, var_val) in enumerate(ds_diff.data_vars.items()):
            formatted = f"{var_name:<5} - {var_val.max().item():.1e}"
            line.append(formatted)
            if (i % 4 == 3) or (i == len(ds_diff.data_vars) - 1):
                line_str = " :: ".join(line)
                line = []
                summary.append(line_str)
        print("\n".join(summary))

    # run the model for the full number of steps with tendency reversion
    # no need to set front hook, if TR active it will be identity here
    model.rear_hook = tendency_reversion_with_user_rpert

    # add optional initial perturbation
    if initial_perturbation is not None:
        if not isinstance(initial_perturbation, xr.Dataset):
            raise TypeError(
                f"Expected xr.Dataset for initial_perturbation, got {type(initial_perturbation)}"
            )
        else:
            run_kwargs["data"].ds += initial_perturbation

    ds = run.deterministic(**run_kwargs).root

    return ds


def create_initial_condition(
    model: PrognosticModel,
    fill_value: float = 0.0,
    init_time: dt.datetime = dt.datetime(2000, 1, 1, 0, 0),
) -> xr.Dataset:
    """Create a dataset with the correct coordinates and variables for a given model."""
    coords = model.input_coords()
    real_coords = {
        "time": [init_time + t for t in coords["lead_time"]],
        "lat": coords["lat"],
        "lon": coords["lon"],
    }
    ds = xr.Dataset(coords=real_coords)
    for var in coords["variable"]:
        da = xr.DataArray(
            data=fill_value,
            coords=real_coords,
            dims=["time", "lat", "lon"],
        )
        ds[var] = da

    return ds


def latitude_weighted_mean(da, latitudes, device="cpu"):
    """
    Calculate the latitude weighted mean of a variable using torch operations on GPU.
    Needs tests to ensure it works correctly.

    Parameters:
    -----------
    da : xarray.DataArray or torch.Tensor
        The data to average
    latitudes : xarray.DataArray or numpy.ndarray
        The latitude values

    Returns:
    --------
    torch.Tensor
        The latitude-weighted mean
    """
    # Convert inputs to torch tensors if needed
    coords = {
        dim: da[dim]
        for dim in da.dims
        if dim not in ["latitude", "longitude", "lat", "lon"]
    }
    if isinstance(da, xr.DataArray):
        da = torch.from_numpy(da.values)
    if isinstance(latitudes, xr.DataArray):
        latitudes = latitudes.values

    # Move to GPU if available
    da = da.to(device)

    # Calculate weights
    lat_radians = torch.from_numpy(np.deg2rad(latitudes)).to(device)
    weights = torch.cos(lat_radians) / (
        torch.sum(torch.cos(lat_radians)) * da.shape[-1]
    )

    # Expand weights to match data dimensions
    weights = weights.view(1, -1, 1)  # Add dims for broadcasting

    # Calculate weighted mean
    weighted_data = da * weights
    averaged = weighted_data.nansum(dim=(-2, -1))  # Average over lat, lon dimensions
    return xr.DataArray(averaged.cpu().numpy(), coords=coords)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Adapted from Hakim and Masanam (2024) repository: https://github.com/modons/DL-weather-dynamics/blob/main/panguweather_utils.py#L145
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = list(map(np.radians, [lon1, lat1, lon2, lat2]))
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km


def gen_circular_perturbation(
    lat_2d, lon_2d, ilat, ilon, amp, locRad=1000.0, Z500=False
):
    """
    Adapted from Hakim and Masanam (2024) repository: https://github.com/modons/DL-weather-dynamics/blob/main/panguweather_utils.py#L162
    """
    grav = 9.81
    nlat = lat_2d.shape[0]
    nlon = lon_2d.shape[1]
    site_lat = lat_2d[ilat, 0]
    site_lon = lon_2d[0, ilon]
    lat_vec = np.reshape(lat_2d, [nlat * nlon])
    lon_vec = np.reshape(lon_2d, [nlat * nlon])
    dists = np.zeros(shape=[nlat * nlon])
    dists = np.array(haversine(site_lon, site_lat, lon_vec, lat_vec), dtype=np.float64)

    hlr = 0.5 * locRad  # work with half the localization radius
    r = dists / hlr

    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)  # closest
    ind_outer = np.where(dists > hlr)  # close
    ind_out = np.where(dists > 2.0 * hlr)  # out

    # Gaspari-Cohn function
    covLoc = np.ones(shape=[nlat * nlon], dtype=np.float64)

    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (
        ((-0.25 * r[ind_inner] + 0.5) * r[ind_inner] + 0.625) * r[ind_inner]
        - (5.0 / 3.0)
    ) * (r[ind_inner] ** 2) + 1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = (
        (
            (
                ((r[ind_outer] / 12.0 - 0.5) * r[ind_outer] + 0.625) * r[ind_outer]
                + 5.0 / 3.0
            )
            * r[ind_outer]
            - 5.0
        )
        * r[ind_outer]
        + 4.0
        - 2.0 / (3.0 * r[ind_outer])
    )
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    # prevent negative values: calc. above may produce tiny negative
    covLoc[covLoc < 0.0] = 0.0

    if Z500:
        # 500Z:
        print("500Z perturbation...")
        perturb = np.reshape(covLoc * grav * amp, [nlat, nlon])
    else:
        # heating:
        print("heating perturbation...")
        perturb = np.reshape(covLoc * amp, [nlat, nlon])

    return perturb


def gen_elliptical_perturbation(lat, lon, k, ylat, xlon, locRad):
    """
    center a localized ellipse at (xlat,xlon)

    Adapted from Hakim and Masanam (2024) repository: https://github.com/modons/DL-weather-dynamics/blob/main/panguweather_utils.py#L208

    k: meridional wavenumber; disturbance is non-zero up to first zero crossing in cos
    xlat: latitude, in degrees to center the function
    xlon: longitude, in degrees to center the function
    locRad: zonal GC distance, in km
    """
    km = 1.0e3
    nlat = len(lat)
    nlon = len(lon)

    ilon = xlon * 4.0  # lon index of center
    ilat = int((90.0 - ylat) * 4.0)  # lat index of center
    yfunc = np.cos(np.deg2rad(k * (lat - ylat)))

    # first zero-crossing
    crit = np.cos(np.deg2rad(k * (lat[ilat] - ylat)))
    ll = np.copy(ilat)
    while crit > 0:
        ll -= 1
        crit = yfunc[ll]

    yfunc[: ll + 1] = 0.0
    yfunc[2 * ilat - ll :] = 0.0

    # gaspari-cohn in logitude only, at the equator
    dx = 6380.0 * km * 2 * np.pi / (360.0)  # 1 degree longitude at the equator
    dists = np.zeros_like(lon)
    for k in range(len(lon)):
        dists[k] = dx * np.min([np.abs(lon[k] - xlon), np.abs(lon[k] - 360.0 - xlon)])

    # locRad = 10000.*km
    hlr = 0.5 * locRad  # work with half the localization radius
    r = dists / hlr

    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)  # closest
    ind_outer = np.where(dists > hlr)  # close
    ind_out = np.where(dists > 2.0 * hlr)  # out

    # Gaspari-Cohn function
    covLoc = np.ones(nlon)

    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (
        ((-0.25 * r[ind_inner] + 0.5) * r[ind_inner] + 0.625) * r[ind_inner]
        - (5.0 / 3.0)
    ) * (r[ind_inner] ** 2) + 1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = (
        (
            (
                ((r[ind_outer] / 12.0 - 0.5) * r[ind_outer] + 0.625) * r[ind_outer]
                + 5.0 / 3.0
            )
            * r[ind_outer]
            - 5.0
        )
        * r[ind_outer]
        + 4.0
        - 2.0 / (3.0 * r[ind_outer])
    )
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    # prevent negative values: calc. above may produce tiny negative
    covLoc[covLoc < 0.0] = 0.0

    # make the function
    [a, b] = np.meshgrid(covLoc, yfunc)
    perturb = a * b

    return perturb


def gen_baroclinic_wave_perturbation(
    lat, lon, ylat, xlon, u_pert_base, locRad, a=6.371e6
):
    """
    Implementation of baroclinic wave perturbation from Bouvier et al. (2024).

    Produces a "localised unbalanced [u-]wind perturbation" to be added to a "baroclinically unstable background state".

    input:
    ------
    lat : numpy.ndarray
        the latitude values
    lon : numpy.ndarray
        the longitude values
    ylat : float
        the latitude of the center of the perturbation
    xlon : float
        the longitude of the center of the perturbation
    u_pert_base : float
        the base amplitude of the u-wind perturbation
    locRad : float
        the localization radius (approximate size of perturbation) in km
    a (optional) : float
        the radius of the earth in m (default is 6.371e6 m)

    output:
    -------
    perturb : numpy.ndarray
        the perturbation array with shape (nlat, nlon)
    """
    radlat = np.deg2rad(lat)
    radlon = np.deg2rad(lon)
    radylat = np.deg2rad(ylat)
    radxlon = np.deg2rad(xlon)

    # make the grid
    lon_2d, lat_2d = np.meshgrid(radlon, radlat)

    # calculate distance from center of perturbation for each grid point
    great_circle_dist = a * np.arccos(
        np.sin(radylat) * np.sin(lat_2d)
        + np.cos(radylat) * np.cos(lat_2d) * np.cos(lon_2d - radxlon)
    )
    perturb = u_pert_base * np.exp(-((great_circle_dist / locRad) ** 2))

    return perturb


def sort_latitudes(ds: xr.Dataset, model_name: str, input: bool):
    
    default_lat_direction = "ascending"
    if input:
        lat_direction = model_info.MODEL_LATITUDE_ORDERING[model_name]
    else:
        lat_direction = default_lat_direction
    lat = ds["lat"]
    if lat_direction == "ascending":
        if lat[0] > lat[-1]:
            print(
                f"Latitude coordinates should be ascending for {model_name}, reversing."
            )
            ds = ds.sortby("lat", ascending=True)
    else:
        if lat[0] < lat[-1]:
            print(
                f"Latitude coordinates should be descending for {model_name}, reversing."
            )
            ds = ds.sortby("lat", ascending=False)

    return ds

if __name__ == "__main__":
    # test of latitude_weighted_mean function
    
    ### test 1: uniform random 0-1 data, should get ~0.5 back
    lat = np.linspace(-90, 90, 181)
    data = np.random.rand(181, 3600)
    da = xr.DataArray(
        data,
        coords={
            "lat": lat,
            "lon": np.arange(3600),
        },
        dims=["lat", "lon"],
    )
    lw_mean = latitude_weighted_mean(da, latitudes=lat)
    print(f"Test 1: mean should be ~0.5, got {lw_mean.values}")
    
    ### test 2: set latitude bands to known values, should get weighted average back
    # area ~ cos(lat), so for lat = [-90, 0, 90], weights are [0, 1, 0], so mean should be equator value
    lat = np.array([-90, 0, 90])
    data = np.array([[0], [1], [0]])
    da = xr.DataArray(
        data,
        coords={
            "lat": lat,
            "lon": np.arange(1),
        },
        dims=["lat", "lon"],
    )
    lw_mean = latitude_weighted_mean(da, latitudes=lat)
    print(f"Test 2: mean should be 1.0, got {lw_mean.values}")
    
    ### test 3: set latitude bands to known values, should get weighted average back
    # area ~ cos(lat), so for lat = [-60, 0, 60], weights are [0.5, 1, 0.5]
    lat = np.array([-60, 0, 60])
    data = np.array([[10], [5], [10]])
    da = xr.DataArray(
        data,
        coords={
            "lat": lat,
            "lon": np.arange(1),
        },
        dims=["lat", "lon"],
    )
    lw_mean = latitude_weighted_mean(da, latitudes=lat)
    print(f"Test 3: mean should be 7.5, got {lw_mean.values}")
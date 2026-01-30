"""
1. Read ic dates from config
2. Download ERA5 data for all model variables specified in model_info at those ic dates
3. Save superset datafiles to ERA5 directory
"""

from utils import model_info, general
import cdsapi
import xarray as xr
import numpy as np
import multiprocessing as mp
import datetime as dt
from pathlib import Path


def sl_raw_fname(var: str, date: dt.datetime) -> str:
    """Generate filename for a single {time, sl_level, variable} chunk."""
    return f"v={var}_l=sl_d={date.strftime('%Y%m%d%H')}.nc"


def pl_raw_fname(var: str, level: int, date: dt.datetime) -> str:
    """Generate filename for a single {time, pl_level, variable} chunk."""
    return f"v={var}_l={level}_d={date.strftime('%Y%m%d%H')}.nc"


def in_raw_fname(var: str) -> str:
    """Generate filename for an invariant variable file."""
    return f"v={var}_l=in_d=in.nc"


def output_path(date: dt.datetime) -> str:
    """Generate filename for superset ERA5 file"""
    return f"{date.strftime('%Y%m%dT%H')}.png"


def generate_tp06_dates(dates: list[dt.datetime]) -> list[dt.datetime]:
    """Return a list of the 6 hours preceding each date in the input list."""
    tp06_dates = []
    for date in dates:
        # don't change the reverse ordering below
        # func "aggregate_tp_files" depends on it
        for i in range(6):
            tp06_dates.append(date - dt.timedelta(hours=i))
    return tp06_dates


def download_chunk(
    variable: str, date: dt.datetime, level: int | str, dataset: str, download_dir: Path
):
    """
    Downloads a single timestep of data for one variable from the CDS API.
    """
    request = {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": date.year,
        "month": str(date.month).zfill(2),
        "day": str(date.day).zfill(2),
        "time": str(date.hour).zfill(2) + ":00",
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if isinstance(level, int):
        request["pressure_level"] = [str(level)]  # add to request
        fname = pl_raw_fname(variable, level, date)

    elif level == "single":
        fname = sl_raw_fname(variable, date)

    else:
        print(
            f"ERROR: level must be int or 'single', got level '{level}' of type {type(level)} instead."
        )
        return f"invalid_level_{level}", "failed"

    if (download_dir / fname).exists():
        # print(f"File {fname} already exists, skipping download.")
        return fname, "skipped"

    # Download the data
    client = cdsapi.Client()
    try:
        client.retrieve(dataset, request, download_dir / fname)
        # if we don't do this here, downstream apps
        # get confused by these names; is u10
        # the 10m u wind, or the 10hPa u wind?
        rewrite_vars = {
            "10m_v_component_of_wind": ("v10", "v10m"),
            "10m_u_component_of_wind": ("u10", "u10m"),
            "100m_v_component_of_wind": ("v100", "v100m"),
            "100m_u_component_of_wind": ("u100", "u100m"),
        }
        if variable in rewrite_vars:
            old_name, new_name = rewrite_vars[variable]
            ds = xr.open_dataset(download_dir / fname)
            ds = ds.rename_vars({old_name: new_name})
            (download_dir / fname).unlink()  # remove original file
            ds.to_netcdf(download_dir / fname)
    except Exception as e:
        print(f"ERROR: request {fname} failed: {e}")
        return fname, "failed"

    return fname, "succeeded"


def run_parallel_download(dates, var_names, var_types, ncpus):
    pl_dataset = "reanalysis-era5-pressure-levels"
    sfc_dataset = "reanalysis-era5-single-levels"
    args_list = []
    for date in dates:
        for var_name, var_type in zip(var_names, var_types):
            if var_type == model_info.PL:
                p_level = int(var_name[1:])  # extract level from var name
                args_list.append(
                    (
                        model_info.E2S_TO_CDS[var_name[0]],
                        date,
                        p_level,
                        pl_dataset,
                        raw_data_dir,
                    )
                )
            elif var_type == model_info.SL:
                args_list.append(
                    (
                        model_info.E2S_TO_CDS[var_name],
                        date,
                        "single",
                        sfc_dataset,
                        raw_data_dir,
                    )
                )
            else:
                continue  # invariant variable, skip download

    # download the data in parallel
    print(f"mp.Pool using ncpus={ncpus}")
    print(f"downloading {len(args_list)} files in parallel")

    with mp.Pool(processes=ncpus) as pool:
        results = pool.starmap(download_chunk, args_list)

    failed_downloads = [fname for fname, status in results if status == "failed"]
    skipped_downloads = [fname for fname, status in results if status == "skipped"]
    succeeded_downloads = [fname for fname, status in results if status == "succeeded"]

    failed_downloads_str = (
        "\n\tfname - ".join(failed_downloads) if failed_downloads else "None"
    )
    print(f"Download stats:")
    print(f"\tFailed: {failed_downloads_str}")
    print(f"\tSkipped: {skipped_downloads}")
    print(f"\tSucceeded: {succeeded_downloads}")
    if failed_downloads:
        print(f"\tFailed downloads were: {failed_downloads}")

    return results


def aggregate_tp_files(tp_dates: list[dt.datetime], download_dir: Path):
    """
    Aggregates the hourly total precipitation files for the given dates into 6-hourly total precipitation files
    and writes those completed fields to the same location.
    """
    # divide tp_dates into 6-hourly intervals
    tp_dates_6chunked = [tp_dates[i : i + 6] for i in range(0, len(tp_dates), 6)]
    for chunk in tp_dates_6chunked:
        output_path = download_dir / sl_raw_fname("total_precipitation_06", chunk[0])
        if output_path.exists():
            # print(f"File {output_path} already exists, skipping.")
            continue
        fpaths = [
            download_dir / sl_raw_fname("total_precipitation", date) for date in chunk
        ]
        ds = xr.open_mfdataset(
            fpaths, combine="nested", parallel=True, engine="netcdf4"
        )
        t = chunk[0]
        # add chunk[0] as valid_time coord
        ds = ds.sum(dim="valid_time")
        ds["tp"] = ds["tp"].expand_dims(
            dict(valid_time=[np.datetime64(t)]),
            axis=0,
        )
        ds = ds.rename_vars({"tp": "tp06"})
        # write 6-hourly total precipitation file valid for the end of the interval
        ds.to_netcdf(output_path)


def create_super_IC_from_vars(
    dates: list[dt.datetime], vars_dir: Path, var_names: list[str], var_types: list[int]
) -> xr.Dataset:
    """Creates initial conditions for each IC date given all the input steps."""
    # open all time mean files and concatenate them into a single dataset
    # where each variable is a separate DataArray
    datasets = [xr.Dataset() for _ in range(len(dates))]
    for i, date in enumerate(dates):
        for variable, var_type in zip(var_names, var_types):
            if var_type == model_info.PL:
                level = int(variable[1:])  # extract level from var name
                var = variable[0]
                file_path = vars_dir / pl_raw_fname(
                    model_info.E2S_TO_CDS[var], level, date
                )
                da = xr.open_dataarray(file_path).squeeze("pressure_level", drop=True)
            elif var_type == model_info.SL:
                file_path = vars_dir / sl_raw_fname(
                    model_info.E2S_TO_CDS[variable], date
                )
                da = xr.open_dataarray(file_path)
            elif var_type == model_info.IN:
                file_path = vars_dir / in_raw_fname(model_info.E2S_TO_CDS[variable])
                da = xr.open_dataarray(file_path)

            datasets[i][variable] = da

    # join together individual dates for models like GraphCast and FuXi which require multiple timesteps
    ds = xr.concat(datasets, dim="valid_time")
    # make it match the E2S format
    ds = ds.rename({"latitude": "lat", "longitude": "lon", "valid_time": "time"})
    ds = ds.drop_vars(["number", "expver"], errors="ignore")

    return ds


def make_perturbation(
    IC_ds: xr.Dataset, perturbation_degK: float | int, make_balanced: bool
) -> xr.Dataset:
    """
    Defines the perturbation that will be applied in the experiment.

    For the energy balance experiment, this will be a temperature perturbation with
    an optional "balancing" component which consists of changes to bring the
    geopotential heights into alignment with the hypsometric equation.
    """
    pert_ds = IC_ds.copy() * 0
    Rd = 287.053  # J/kg/K
    g = 9.807  # m/s^2
    ps = IC_ds["sp"]  # in hPa
    for lev in model_info.STANDARD_13_LEVELS:
        # set temp pert to given value
        tfield = f"t{lev}"
        pert_ds[tfield] = perturbation_degK

        # set z pert
        zfield = f"z{lev}"
        zpert = Rd * perturbation_degK * np.log(ps / lev) / g # original EQ from travis on board shows lev/ps, but we think it should flipped from hydrostatic
        pert_ds[zfield] = zpert


if __name__ == "__main__":
    ncpus = 4  # number of CPUs to use for parallelization, don't exceed available ncpus

    config_path = Path(__file__).parent / "0.config.yaml"
    config = general.read_config(config_path)
    raw_data_dir = Path(config["ERA5_variable_slices_directory"]).resolve()
    output_data_dir = Path(config["ERA5_full_ICs_directory"]).resolve()
    perturbation_value = config["temp_perturbation_degC"]
    balance_perturbation = config["balance_perturbation"]
    perturbed = config["temp_perturbation_degC"] != 0

    ic_dates = [
        dt.datetime.strptime(str_date, "%Y-%m-%dT%Hz")
        for str_date in config["ic_dates"]
    ]
    for date in ic_dates:
        print(f"Processing downloading IC for {date}")
        all_vars = [var for var in model_info.MASTER_VARIABLES_NAMES if var != "tp06"]
        all_types = [
            vtype
            for var, vtype in zip(
                model_info.MASTER_VARIABLES_NAMES, model_info.MASTER_VARIABLES_TYPES
            )
            if var != "tp06"
        ]
        two_timesteps = [date - dt.timedelta(hours=6), date]
        run_parallel_download(
            two_timesteps,
            all_vars,
            all_types,
            ncpus,
        )
        tp_dates = generate_tp06_dates(two_timesteps)
        run_parallel_download(
            tp_dates,
            ["tp"],
            [model_info.SL],
            ncpus,
        )
        downloaded_tp_raw = True
        aggregate_tp_files(tp_dates, raw_data_dir)
        # static fields are weird
        # because they are always single level
        # and only one of them needs to be downloaded
        # so we download them to raw and then stick them
        # in the time_means
        for var in model_info.IN_VARIABLES:
            raw_path = raw_data_dir / in_raw_fname(model_info.E2S_TO_CDS[var])
            if not raw_path.exists():
                print(
                    FileNotFoundError(
                        f"Field {var} not found at {raw_path}, see extract_static_graphcast_fields.py"
                    )
                )
            # copy to time_means if needed
            fpath = raw_data_dir / in_raw_fname(var)
            if fpath.exists():
                fpath.unlink()
            fpath.symlink_to(raw_path)
        save_path = output_data_dir / f"{date.strftime('%Y%m%dT%H')}.nc"
        skip_IC_save = False
        if save_path.exists():
            skip_IC_save = True
        if not skip_IC_save or perturbed:
            IC_ds = create_super_IC_from_vars(
                two_timesteps,
                raw_data_dir,
                model_info.MASTER_VARIABLES_NAMES,
                model_info.MASTER_VARIABLES_TYPES,
            )
            IC_ds.to_netcdf(save_path)
            print(f"Saving {date.strftime('%Y%m%dT%H')} to {save_path}")
        else:
            print(f"File {save_path} already exists, skipping.")

        if perturbed:
            perturbation_ds = make_perturbation(
                IC_ds, perturbation_value, balance_perturbation
            )
            perturbation_ds.to_netcdf()
            print(f"Saving {balance_perturbation=} with {perturbation_value=}")

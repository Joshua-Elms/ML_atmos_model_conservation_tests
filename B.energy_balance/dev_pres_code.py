import xarray as xr
import numpy as np
from pathlib import Path
from utils import model_info, general
import metpy
import scipy

model_name = "Pangu6x"
data_path = Path(
    "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/B.energy_balance/data/pert=0C_nt=4_nic=1_NO_GCO/Pangu6x_output.nc"
)
zs_path = Path(
    "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/B.energy_balance/surface_geopotential.nc"
)
model_vars = model_info.MODEL_VARIABLES[model_name]["names"]
tmp_ds = xr.open_dataset(data_path)
zs_ds = xr.open_dataset(zs_path)

model_levels = tmp_ds["level"].values
### Get pressure for integration ###
model_levels_pa = 100 * np.array(
    model_levels
)  # convert to Pa from hPa, used for integration
g = 9.81  # m/s^2
cp = 1005.0  # J/kg/K
Lv = 2.5e6  # J/kg
sb_const = (
    5.670374419e-8  # W/m^2/K^4, from https://physics.nist.gov/cgi-bin/cuu/Value?sigma
)

### Calculate total energy components ###
integrate = lambda da: (1 / g) * scipy.integrate.trapezoid(da, model_levels_pa, axis=0)


def surface_aware_integrate(da, z, zs, surface_field=None):
    """Integrate accounting for surface geopotential height."""

    if "level" not in da.dims:
        raise ValueError(
            f"DataArray must have 'level' dimension for surface-aware integration."
        )
    if da["level"].size != model_levels_pa.size:
        raise ValueError(
            f"DataArray 'level' dimension size must match model levels size."
        )
    if da.shape != z.shape:
        raise ValueError(
            f"Integrable-field DataArray and geopotential height DataArray shapes must match. Got {da.shape} and {z.shape}."
        )

    # keep track of axis to integrate over
    lev_ax = da.get_axis_num("level")
    if lev_ax != 0:
        print("Moving level axis to index 0 for manipulation.")
        original_dims = da.dims
        # move level axis to front for easier manipulation
        da = da.transpose("level", *[dim for dim in da.dims if dim != "level"])
        z = z.transpose("level", *[dim for dim in z.dims if dim != "level"])

    # convert da to numpy array for manipulation
    da_np = da.data

    # add level below bottom level equal to bottom level (for now) in case surface is below lowest model level
    extended_levels_pa = np.append(
        model_levels_pa, model_levels_pa[-1]
    )  # assumes pressure increasing on level axis
    # add level to z equal to lowest model level (for now)
    extended_z_np = np.concatenate(
        [z.data, z.isel(level=[-1]).data], axis=0
    )  # assumes level axis is 0
    # make levels shaped liked da for broadcasting
    reshape_dims = [1] * da.ndim
    reshape_dims[0] = extended_levels_pa.size
    tile_dims = list(da.shape)
    tile_dims[0] = 1
    full_levels_pa = np.tile(extended_levels_pa.reshape(reshape_dims), tile_dims)
    level_idxs = np.arange(extended_levels_pa.size)
    full_level_idxs = np.tile(level_idxs.reshape(reshape_dims), tile_dims)
    extended_da_np = np.concatenate(
        [da_np, np.zeros_like(da.isel(level=[0])).data], axis=0
    )
    # mask: interpolate vs extrapolate depending on surface geopotential height
    extrapolate_mask = (
        zs.data < z.isel(level=-1).data
    )  # True where surface is below lowest model level
    interpolate_mask = (
        ~extrapolate_mask
    )  # True where surface is above lowest model level

    # make all necessary arrays to be partially populated by both interpolation and extrapolation methods
    single_level_shape = z.isel(level=0).data.shape
    j0_idxs = np.full(
        single_level_shape, np.nan
    )  # lower bounding layer index (interp) or original lowest model level (extrap)
    j1_idxs = np.full(
        single_level_shape, np.nan
    )  # upper bounding layer index (interp) or original 2nd lowest model level (extrap)
    dpdz = np.full(
        single_level_shape, np.nan
    )  # dP/dz between bounding layers (interp) or between lowest 2 model levels (extrap)
    ps = np.full(single_level_shape, np.nan)  # calculated surface pressure
    fs = np.full(
        single_level_shape, np.nan
    )  # calculated field value at surface, will be overwritten if surface_field is given

    # extrapolation
    j0_idxs[extrapolate_mask] = level_idxs[-2]  # original lowest model level
    j1_idxs[extrapolate_mask] = level_idxs[-3]  # original 2nd lowest model level
    # interpolation
    diff = z.data - zs.data
    nearest_level = np.argmin(np.abs(diff), axis=0)
    min_abs_diff = np.min(np.abs(diff), axis=0)
    pos_diff_mask = min_abs_diff > 0
    neg_diff_mask = min_abs_diff <= 0
    eq_diff_mask = min_abs_diff == 0
    # nearest level above surface
    j0_idxs[interpolate_mask & pos_diff_mask] = (
        nearest_level[interpolate_mask & pos_diff_mask] + 1
    )
    j1_idxs[interpolate_mask & pos_diff_mask] = nearest_level[
        interpolate_mask & pos_diff_mask
    ]
    j0_idxs[interpolate_mask & neg_diff_mask] = nearest_level[
        interpolate_mask & neg_diff_mask
    ]
    j1_idxs[interpolate_mask & neg_diff_mask] = (
        nearest_level[interpolate_mask & neg_diff_mask] - 1
    )

    # convert index arrays to boolean
    j0 = j0_idxs == full_level_idxs
    j1 = j1_idxs == full_level_idxs

    # whether interp or extrap, following works:
    dpdz = (
        (full_levels_pa[j1] - full_levels_pa[j0])
        / (extended_z_np[j1] - extended_z_np[j0])
    ).reshape(single_level_shape)
    ps = extended_z_np[j0].reshape(single_level_shape) + (
        dpdz * (zs.data - extended_z_np[j0].reshape(single_level_shape))
    )
    if surface_field is not None:  # model returns surface field, like t2m or u10m
        fs = surface_field.data
    else:  # model does not return surface field, -polate
        da_0 = extended_da_np[j0].reshape(single_level_shape)
        da_1 = extended_da_np[j1].reshape(single_level_shape)
        dfdz = (da_1 - da_0) / (
            extended_z_np[j1].reshape(single_level_shape)
            - extended_z_np[j0].reshape(single_level_shape)
        )
        fs = da_0 + (dfdz * (zs.data - extended_z_np[j0].reshape(single_level_shape)))

    # finishing extrapolation case
    full_levels_pa[-1][extrapolate_mask] = ps[
        extrapolate_mask
    ]  # set surface pressure at extended level
    extended_da_np[-1][extrapolate_mask] = fs[
        extrapolate_mask
    ]  # set surface field value at extended level

    # finishing interpolation case
    full_levels_pa[j0][interpolate_mask.flatten()] = ps[
        interpolate_mask
    ]  # set surface pressure at appropriate level
    extended_da_np[j0][interpolate_mask.flatten()] = fs[interpolate_mask]

    # iteratively go to (lower height, higher pressure) levels and set pressure values to surface pressure at each to make zero thickness layers
    for i in range(level_idxs.size - j0_idxs.min().astype(int)):
        breakpoint()
        higher_j0_idxs = j0_idxs.copy()
        higher_level_exists_mask = higher_j0_idxs < (level_idxs.size - 1)
        print(
            f"Iteration {i}, setting higher levels for {np.sum(higher_level_exists_mask)} points."
        )
        higher_j0_idxs[higher_level_exists_mask] += 1
        higher_j0 = higher_j0_idxs == full_level_idxs
        full_levels_pa[higher_j0][interpolate_mask.flatten()] = ps[interpolate_mask]
        j0_idxs = higher_j0_idxs

    # finally, integrate
    integrated = (1 / g) * scipy.integrate.simpson(
        extended_da_np, full_levels_pa, axis=0
    )

    return integrated


zs_ds = zs_ds.rename({"longitude": "lon", "latitude": "lat"})
sensible_total_energy = cp * surface_aware_integrate(
    tmp_ds["T"], tmp_ds["Z"] / g, zs_ds["geopotential"] / g, tmp_ds["T"].sel(level=1000)
)

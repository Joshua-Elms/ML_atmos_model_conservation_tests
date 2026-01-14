import xarray as xr
import numpy as np
from pathlib import Path

res = 0.25
lat_start = 90
lat_stop = -90
lon_start = 0
lon_stop = 360

print("Making lat/lon centers from given resolution and extents:")
lats = np.arange(lat_start, lat_stop - res, -res)
lons = np.arange(lon_start, lon_stop, res)
nlat = len(lats)
nlon = len(lons)
m_lat = lats.repeat(nlon)  # lats look like 90, 90, 90, ..., -90, -90, -90
m_lon = np.tile(lons, nlat)  # lons look like 0, 0.25, ..., 359.75, 360, 0, 0.25, ...
print("Lat/Lons derived from res and extents:")
print(m_lat)
print(m_lon)

print(f"Now make the corners from the centers and resolution")
ul_lat_est = m_lat + res / 2
ur_lat_est = m_lat + res / 2
ll_lat_est = m_lat - res / 2
lr_lat_est = m_lat - res / 2
ul_lon_est = m_lon - res / 2
ur_lon_est = m_lon + res / 2
ll_lon_est = m_lon - res / 2
lr_lon_est = m_lon + res / 2

# order of corners is [ll, lr, ur, ul]
lat_corners = np.array([ll_lat_est, lr_lat_est, ur_lat_est, ul_lat_est]).T
lat_corners = np.clip(
    lat_corners, min=-90, max=90
)  # handles ERA5 grid oddity, giving polar grid boxes half-sized lat bounds
lon_corners = np.array([ll_lon_est, lr_lon_est, ur_lon_est, ul_lon_est]).T

print("Latitude corners:")
print(lat_corners)
print("Longitude corners:")
print(lon_corners)

print("Calculate grid areas as proportion of spherical surface area:")
R = 6371e3  # radius of the Earth in meters
total_sfc_area = 4 * np.pi * R**2
lat_radians = np.deg2rad(m_lat)
props = np.cos(lat_radians) / np.sum(np.cos(lat_radians))
grid_areas = props * total_sfc_area
print("Grid areas (m^2):")
print(grid_areas)

grid_ds = xr.Dataset(
    data_vars={
        "grid_dims": ("grid_rank", [nlon, nlat]),
        "grid_area": ("grid_size", grid_areas),
        "grid_imask": ("grid_size", np.ones(nlat * nlon, dtype=int)),
        "grid_center_lat": ("grid_size", m_lat),
        "grid_center_lon": ("grid_size", m_lon),
        "grid_corner_lat": (("grid_size", "grid_corners"), lat_corners),
        "grid_corner_lon": (("grid_size", "grid_corners"), lon_corners),
    },
    coords={
        "grid_rank": (("grid_rank"), [0, 1]),
        "grid_size": (("grid_size"), np.arange(nlat * nlon)),
        "grid_corners": (("grid_corners"), [0, 1, 2, 3]),
    },
)
output_path = Path(
    "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/Regrid_E3SMv2.1_to_360x180/Grids_and_Maps/era5_721x1440.nc"
)
if output_path.exists():
    output_path.unlink()
    print(f"Deleted existing file at {output_path}")
grid_ds.to_netcdf(output_path)
print(f"Saved to {output_path}")

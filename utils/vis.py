import xarray as xr
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib import colormaps
import cartopy.crs as ccrs
import shutil
import yaml

class MidpointNormalize(colors.Normalize):
    """
    Code from Joe Kingston: https://chris35wills.github.io/matplotlib_diverging_colorbar/

    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = ([self.vmin, self.midpoint, self.vmax], [0, 0.5, 1])
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def dir2gif(img_dir, output_path, fps, loop: int = 0):
    """
    Convert a directory of images to a gif.

    Inputs:
        img_dir (Path): directory containing images
        output_path (Path): path to save gif
        fps (int): frames per second
        loop (int): number of times to loop the gif (0 for infinite)

    Outputs:
        gif saved to output_path
    """
    img_paths = sorted(img_dir.glob("*"))
    images = [imageio.imread(img_path) for img_path in img_paths]
    imageio.mimsave(output_path, images, fps=fps, loop=loop)


def plotting_loop(
    dim_name,
    iterable,
    data,
    fig,
    obj,
    update_func,
    output_dir,
    fname_prefix,
    titles,
    central_longitude,
    keep_images=False,
    fps=1,
    dpi=100,
):
    """
    Inputs:
        dim_name (str): name of the dimension to loop over
        iterable (normally time or variable)
        data (xarray dataset)
        fig (matplotlib figure)
        obj (prepared matplotlib image object on axis)
        update_func (function that updates the plot)
        output_dir (directory to save images)
        fname_prefix (prefix for image filenames)
        titles (list of titles for each image, must match length of iterable)
        central_longitude (central longitude for the map projection)
        keep_images (if True, images will not be deleted after the loop)
        fps (frames per second for gif)

    Outputs
        a gif of the plot loop at the specified output_dir

    Return:
        fig
    """
    save_dir = output_dir / fname_prefix
    save_dir.mkdir(exist_ok=True, parents=True)
    # if user specifies a central longitude, roll the data
    # away from default of 180 °E by specified amount
    roll_by = round(180 - central_longitude) * 4
    for i, idx in enumerate(iterable):
        data_slice = data.isel({dim_name: idx}).values
        rolled_slice = np.roll(data_slice, shift=roll_by, axis=-1)
        update_func(fig, obj, rolled_slice, title=titles[i])
        fig.savefig(save_dir / f"{fname_prefix}_{i:03}.png", dpi=dpi)

    dir2gif(save_dir, output_dir / f"{fname_prefix}.gif", fps=fps)

    if not keep_images:
        shutil.rmtree(save_dir)

    return fig


def create_and_plot_variable_gif(
    data: xr.DataArray,
    plot_var: str,
    iter_var: str,
    iter_vals: list | range,
    plot_dir: Path,
    units: str,
    cmap: str,
    titles: list[str],
    central_longitude: float = 180.,
    extent: list = None,
    adjust: dict = None,
    dpi: int = 100,
    fps: int = 2,
    dt: int = 6,
    fig_size: tuple = (5, 2.8),
    nlat_ticks: int = 5,
    nlon_ticks: int = 7,
    vlims: tuple = None,
    vmid: float = None,
    keep_images=True,
    cbar_kwargs: dict = {
        "rotation": "horizontal",
        "y": -0.15,
        "horizontalalignment": "right",
        "labelpad": 0,
        "fontsize": 9
        },
):
    """Creates and saves an animated GIF of a variable's evolution over time or another dimension.

    Parameters
    ----------
    data : xr.DataArray
        Input data array containing the variable to plot. Must have dimensions 'latitude', 'longitude', and an additional dimension for iteration.
    plot_var : str
        Name of the variable to plot from the dataset
    iter_var : str
        Name of the dimension to iterate over (e.g., 'time')
    iter_vals : list or range
        Values to iterate over for creating animation frames
    plot_dir : Path
        Directory path where to save the output GIF and temporary files
    units : str
        Units of the variable for labeling
    cmap : str
        Colormap to use for plotting, e.g. 'viridis'
    titles : str
        List of titles for each frame of the animation. Must match the length of iter_vals.
    central_longitude : float, optional
        Central longitude for the map projection. Default is 180 °E
    extent : list, optional
        List of [lon_min, lon_max, lat_min, lat_max] to set the extent of the plot. If None, uses the full range of data.
    adjust : dict, optional
        Dictionary of subplot adjustment parameters for matplotlib
    fig_size : tuple, optional
        Size of the figure in inches (width, height). Default is (5, 2.8).
    dpi : int, optional
        Resolution of the output images in dots per inch. Default is 100.
    fps : int, optional
        Frames per second for the output GIF. Default is 2.
    dt : int, optional
        Time step between frames in hours. Default is 6.
    nlat_ticks : int, optional
        Number of latitude ticks to display on the plot. Default is 5.
    nlon_ticks : int, optional
        Number of longitude ticks to display on the plot. Default is 7.
    vlims : tuple, optional
        Tuple of (vmin, vmax) for color scaling. If None, calculated from data.
    vmid : float, optional
        Midpoint for the colormap normalization, vmin < vmid < vmax. If None, calculated as the average of vmin and vmax. Especially useful for diverging colormaps which have assymetric distances from the midpoint.
    keep_images : bool, optional
        Whether to keep temporary image files after creating GIF. Default is True.
    cbar_kwargs : dict, optional
        Additional keyword arguments for the colorbar, such as label and orientation.

    Returns
    -------
    None
        Saves the animation as a GIF file in the specified plot_dir
    """
    ### prepare helper variables for plotting
    # lat/lon info
    lat, lon = data.lat.values, data.lon.values
    nlat, nlon = len(lat), len(lon)
    
    # check titles
    assert len(titles) == len(iter_vals), "Num. titles must match length of iter_vals"

    # vlims can be set for manual color scaling
    if vlims:
        vmin, vmax = vlims
    else:
        vmin, vmax = data.values.min(), data.values.max()
        
    # set up colormap
    cmap = colormaps.get_cmap(cmap)

    ### make plot
    # set up figure and axis
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={"projection": ccrs.PlateCarree(central_longitude=central_longitude)}) # , subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)}

    # set up first frame to be updated in later loop
    if extent is None:
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    else:
        assert len(extent) == 4, "Extent must be a list of [lon_min, lon_max, lat_min, lat_max]"
        assert extent[0] <= extent[1], "Longitude min must be less than max"
        assert extent[2] <= extent[3], "Latitude min must be less than max"

    if vmid is None:
        vmid = (vmin + vmax) / 2
        
    else:
        assert vmin < vmid < vmax, "vmin must be less than vmid and vmid must be less than vmax for MidpointNormalize to work correctly"
        
    # norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=vmid, clip=True)
    # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    im = ax.imshow(
        data.isel({iter_var:iter_vals[0]}), 
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=extent,
        transform=ccrs.PlateCarree(),
    )   
    ax.coastlines()
    ax.set_global()

    ### Set axis information
    # # labels
    # xax_label = "longitude [°E]"
    # yax_label = "latitude [°N]"
    # ax.set_xlabel(xax_label)
    # ax.set_ylabel(yax_label)

    # ticks and ticklabels
    gl = ax.gridlines(crs=ccrs.PlateCarree(),linewidth=1.0,color='gray', alpha=0.5,linestyle='--', draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ### Set other plot aesthetics
    # background color
    fig.patch.set_facecolor("xkcd:white")

    # proper positioning
    if adjust is None:
        adjust = {
        "top": 1,
        "bottom": 0.03,
        "left": 0.13,
        "right": 0.82,
        "hspace": 0.0,
        "wspace": 0.0,
    }
    fig.subplots_adjust(**adjust)

    # title
    fig.suptitle(titles[0], x=(ax.get_position().x0 + ax.get_position().x1) / 2)
    
    ### colorbar
    # manually add colorbar axis to get the correct positioning
    # not possible to do otherwise, afaik
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.02,
            ax.get_position().y0,
            0.03,
            ax.get_position().y1 - ax.get_position().y0,
        ]
    )
    
    # add colorbar
    cbar_label = f"[{units}]"
    cbar = fig.colorbar(im, cax=cax, orientation="vertical", fraction=0.1)
    
    cbar.ax.set_ylabel(
        cbar_label,
        **cbar_kwargs
        )

    ### define function to update plot at each timestep
    def plot_updater(fig, plot_obj, data, title):
        plot_obj.set_data(data)
        fig.suptitle(title)

    ### run plotting loop
    plotting_loop(
        iter_var,
        iter_vals,
        data,
        fig,
        im,
        plot_updater,
        plot_dir,
        f"{plot_var}",
        titles,
        central_longitude,
        keep_images=keep_images,
        fps=fps,
        dpi=dpi,
    )
    plt.clf()
    plt.close()


if __name__ == "__main__":
    ### set these variables
    config_str = "/glade/work/jmelms/data/dcmip2025_idealized_tests/experiments/long_sim_0/config.yml"
    cmap_str = "viridis" # options here: matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = colormaps.get_cmap(cmap_str)

    ### the rest should work itself out
    config_path = Path(config_str)
    assert config_path.exists(), f"Config file does not exist @ {config_path}"
    print(f"Config file exists @ {config_path}")
    # import config file
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    ic_params = config["initial_condition_parameters"]
    keys, val_pairs = zip(*ic_params.items())
    vals, units = zip(*val_pairs)
    param_val_pairs = dict(zip(keys, vals))
    val_types = [isinstance(v, list) for v in vals]
    exp_param_idx = val_types.index(
        1
    )  # this is the parameter that was varied in the experiment
    exp_param = keys[exp_param_idx]
    exp_vals = np.array(vals[exp_param_idx])
    exp_param_units = units[exp_param_idx]
    assert sum(val_types) == 1, "Only one experimental parameter allowed"

    exp_dir = (
        Path(config["experiment_dir"]) / config["experiment_name"]
    )  # all data for experiment stored here
    ic_nc_dir = exp_dir / "ic_nc"  # contains processed ICs in nc format
    data_path = exp_dir / "output.nc"  # where output from inference is saved

    nt = config["inference_parameters"]["n_steps"]
    dt = 6  # hours. all experiments have this timestep
    n_expvar = len(exp_vals)
    ne = 1  # no ensemble feature has been implemented yet
    ensemble_colors = cmap(np.linspace(0, 1, n_expvar))
    units_table = {
        "VAR_2T": "K",
        "VAR_10U": "m/s",
        "VAR_10V": "m/s",
        "VAR_100U": "m/s",
        "VAR_100V": "m/s",
        "T": "K",
        "U": "m/s",
        "V": "m/s",
        "Z": "m",
        "SP": "hPa",
        "MSL": "hPa",
        "TCWV": "kg/m^2",
        "Q": "kg/kg",
    }

    print(f"Loading data from {data_path}")
    print(f"Iter var: {exp_param}: {exp_vals} {exp_param_units}")

    ds = xr.open_dataset(data_path).isel(ensemble=0)  # data only has one member
    latitudes = ds.latitude  # will be used for calc global mean fields
    time_hours = (ds.time - ds.time[0]) / np.timedelta64(
        1, "h"
    )  # set time coord relative to start time
    ds.update({"time": time_hours})
    ds = ds.assign_attrs({"time units": "hours since start"})

    ### plot settings
    plot_var = "VAR_2T"
    data = ds[plot_var].isel({exp_param:0})
    iter_var = "time"
    iter_vals = np.arange(0, nt, 40)
    units = units_table[plot_var]
    title_str = "{var_name} [{units}] at {time} hours"
    adjust = {
        "top": 1,
        "bottom": 0.03,
        "left": 0.13,
        "right": 0.82,
        "hspace": 0.0,
        "wspace": 0.0,
    }
    dpi = 100
    fps = 4
    keep_images = True
    plot_dir = exp_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    dpi = 300
    fps = 4
    keep_images = False

    create_and_plot_variable_gif(
        data,
        plot_var,
        iter_var,
        iter_vals,
        plot_dir,
        units,
        cmap_str,
        title_str,
        adjust,
        dpi=dpi,
        fps=fps,
        dt=dt,
        keep_images=keep_images
    )

    print(f"Plots saved to {plot_dir}")
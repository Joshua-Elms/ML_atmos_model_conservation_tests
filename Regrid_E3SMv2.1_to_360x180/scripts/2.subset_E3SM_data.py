"""
Because the regrid command will attempt to regrid all available data in the file we point it to
and we only need the initial condition, we isolate the initial timestep and write it out.
"""

import xarray as xr
from pathlib import Path

# unperturbed
input_path = Path(
    "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/B.energy_balance/data/E3SM_runs/v2.1.WCYCLSSP370_20180201-20180401_12n32p_original_20251013/run/v2.1.WCYCLSSP370_20180201-20180401_12n32p_original_20251013.eam.h1.2018-02-01-00000.nc"
)
output_path = Path(
    "/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/original_native_grid/E3SM_IC.nc"
)

# subset to first timestep and write to output file
xr.open_dataset(input_path).isel(time=0).to_netcdf(output_path)

print(f"Saving initial timestep from {input_path.name}")


# perturbed
input_path = Path(
    "/N/project/cascade/For_Josh_from_Paul/2_month_runs_5C_increase/v2.1.WCYCLSSP370_20180201-20180401_12n32p_trial_06/run/v2.1.WCYCLSSP370_20180201-20180401_12n32p_trial_06.eam.h1.2018-02-01-00000.nc"
)
output_path = Path(
    "/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/p5k_native_grid/E3SM_IC.nc"
)

# subset to first timestep and write to output file
xr.open_dataset(input_path).isel(time=0).to_netcdf(output_path)

print(f"Saving initial timestep from {input_path.name}")

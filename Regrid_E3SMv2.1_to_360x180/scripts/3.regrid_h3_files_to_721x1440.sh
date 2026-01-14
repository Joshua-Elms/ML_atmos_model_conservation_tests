#!/bin/bash

# Defines the map file (mapfile) for regridding.
# Sets a single input and output base directory.
# Constructs the full input (h3_input_files) and output (h3_output_files) directories.
# Checks if the input directory exists before running ncremap.
# Creates the output directory if it doesn’t exist.
# Runs the ncremap command to regrid the NetCDF files.
# Prints a message indicating success or failure.

# need this to run ncremap
module load nco

# Path to the mapfile
mapfile="/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/Regrid_E3SMv2.1_to_360x180/Grids_and_Maps/map_ne30pg2_to_era5_721x1440.20260108.nc"

# if mapfile not exist, generate it
if [ ! -f $mapfile ]; then
    echo "File not found!"
    ncremap -s ../Grids_and_Maps/ne30pg2.nc -g ../Grids_and_Maps/era5_721x1440.nc -m $mapfile
fi

### Unperturbed
# Define the input and output directories #
input_dir="/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/original_native_grid"
output_dir="/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/original_regridded"

# Check if input directory exists
if [ -d "$input_dir" ]; then
    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Run the ncremap command
    ncremap -m ${mapfile} -I ${input_dir} -O ${output_dir}

    echo "ncremap executed successfully for directory: ${input_dir}"
else
    echo "Input directory $input_dir does not exist."
fi

### Perturbed
input_dir="/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/p5k_native_grid"
output_dir="/N/scratch/jmelms/ML_atmos_model_conservation_tests_scratch_data/B.energy_balance/data/p5k_regridded"

# Check if input directory exists
if [ -d "$input_dir" ]; then
    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Run the ncremap command
    ncremap -m ${mapfile} -I ${input_dir} -O ${output_dir}

    echo "ncremap executed successfully for directory: ${input_dir}"
else
    echo "Input directory $input_dir does not exist."
fi

# don't need this any more
module unload nco 
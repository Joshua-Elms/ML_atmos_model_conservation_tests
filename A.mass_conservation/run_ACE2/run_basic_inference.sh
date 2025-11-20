#!/bin/bash

source /N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/ace2venv/bin/activate
python -m fme.ace.inference model_data/inference_config.yaml
source /N/slate/jmelms/projects/earth2studio-cu126/.venv2/bin/activate
python postprocess_output.py
echo "Inference and postprocessing complete."
echo "Copying output to respective folders -- make sure to go move them into the right folders!"
cp /N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/output/ACE2_pressure_output.nc /N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/data
cp /N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/output/ACE2_energy_output.nc /N/slate/jmelms/projects/ML_atmos_model_conservation_tests/B.energy_balance/data
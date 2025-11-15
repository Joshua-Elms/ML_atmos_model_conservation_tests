#!/bin/bash

source /N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/ace2venv/bin/activate
python -m fme.ace.inference model_data/inference_config.yaml
source /N/slate/jmelms/projects/earth2studio-cu126/.venv2/bin/activate
python postprocess_output.py
echo "Inference and postprocessing complete."
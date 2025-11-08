#!/bin/bash

source /N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/ace2venv/bin/activate
python -m fme.ace.inference model_data/inference_config.yaml

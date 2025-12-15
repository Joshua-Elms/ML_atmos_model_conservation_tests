#!/bin/bash

#SBATCH -J B.energy_balance
#SBATCH -p hopper
#SBATCH -q hopper
#SBATCH -o output.out
#SBATCH -e log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmelms@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node h100:1
#SBATCH --time=8:00:00
#SBATCH --mem=500GB
#SBATCH -A r00389

source /N/slate/jmelms/projects/earth2studio-cu126/.venv2/bin/activate
python /N/slate/jmelms/projects/ML_atmos_model_conservation_tests/B.energy_balance/1.run_experiment.py
python /N/slate/jmelms/projects/ML_atmos_model_conservation_tests/B.energy_balance/2.e3sm_postprocessing.py
python /N/slate/jmelms/projects/ML_atmos_model_conservation_tests/B.energy_balance/3.run_analysis.py

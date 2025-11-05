#!/bin/bash

#SBATCH -J A.mass_conservation
#SBATCH -p gpu
#SBATCH -o output.out
#SBATCH -e log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmelms@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node v100:1
#SBATCH --time=3:00:00
#SBATCH --mem=100GB
#SBATCH -A r00389

source /N/slate/jmelms/projects/earth2studio-cu126/.venv2/bin/activate
python /N/slate/jmelms/projects/dcmip2025_idealized_tests/experiments/A.mass_conservation/1.run_experiment.py
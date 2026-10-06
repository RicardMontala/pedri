#!/bin/bash
#SBATCH -J BO_PEDRI   	# Job name
#SBATCH -p gpp_tuareg  	# Partition associated to your user
#SBATCH --qos=gpp_std	# Queue to launch to
#SBATCH --time=1:00:00
#SBATCH -o out-%j.out   # STDOUT
#SBATCH -e out-%j.err   # STDERR

## Run configuration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# Load modules
module purge
module load python

unset PYTHONPATH
source pedri-env/bin/activate

# Run
date
python3 run_pedri_shapeOpt.py
date

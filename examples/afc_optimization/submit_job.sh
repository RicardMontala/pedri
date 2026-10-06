#!/bin/bash
#SBATCH -J BO_PEDRI   	# Job name
#SBATCH -p acc_tuareg  	# Partition associated to your user
#SBATCH --qos=acc_pro # Queue to launch to
#SBATCH --time=24:00:00 # Max 00:30:00
#SBATCH -o out-%j.out   # STDOUT
#SBATCH -e out-%j.err   # STDERR
#SBATCH --nodelist=junogpu3

## Run configuration
## Rule: {ntasks} \times {cpus-per-task} = 40
## Rule: --gres=gpu:{ntasks}
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:10

# Load modules
module purge
module load cmake nvhpc hdf5 python

unset PYTHONPATH
source pedri-env/bin/activate

# Run
date
python3 run_pedri_AFCsod2d.py
date

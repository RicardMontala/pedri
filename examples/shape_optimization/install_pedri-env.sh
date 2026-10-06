#!/bin/bash
#
# PEDRI virtual environment for python

# Modules to be loaded
module purge
module load python

# Create the virtual environment
unset PYTHONPATH
python -m venv pedri-env
source pedri-env/bin/activate

# Install PEDRI library into the environment
git clone https://github.com/RicardMontala/pedri.git
cd pedri/
pip install -e ".[neuralfoil]"

deactivate

#!/bin/bash
#
# PEDRI virtual environment for python

# Modules to be loaded
module purge
module load gcc cmake python

# Create the virtual environment
unset PYTHONPATH
python -m venv pedri-env
source pedri-env/bin/activate

# Install PEDRI library into the environment
cd pedri
pip install -e ".[smartsim]"

deactivate

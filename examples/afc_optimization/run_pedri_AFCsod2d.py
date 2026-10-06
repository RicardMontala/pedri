import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import qmc

from pedri.launcher import SOD2DLauncher
from pedri.postprocessor import SOD2DPostprocessor
from pedri.optimizer import BayesianOptimizer
from pedri.utils.plotting import plot_parameter_slice
from pedri.utils.plotting import plot_convergence

import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, # Set logging verbosity: INFO / DEBUG / WARNING /ERROR
    format='%(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout # stderr by default
)
cwd = os.path.dirname(os.path.realpath(__file__))


# --------------- PARAMETERS ----------------
bounds = {"A_DC": [-1.0, 1.0], "A": [0, 1.0], "f": [0.4, 0.7]}

N_CFDs = 10
n_tasks_per_CFD = 1

convergence_threshold = 0.1 # Minimum relative improvement between iterations (in %) defining the convergence threshold
convergence_patience = 5    # Number of consecutive iterations below the threshold required for convergence
max_iterations = 50         # Maximum number of BO iterations

savePlot_step = 1           # Frequency at which GP manifold slice plots are saved
saveCheckpoint_step = 1     # Frequency for saving BO checkpoints

# Define objective function during BO optimization
def compute_objective(Cl, Cd, Cl_std, Cl_base=0.810, Cd_base=0.399, alpha=0.3, beta=0.0):
    return (Cd_base-Cd) - alpha*Cl_std + beta*(Cl-Cl_base)


# ----------- INITIALIZE MODULES ----------
# Initialize SOD2D launcher
launcher = SOD2DLauncher(
    smartsim_launcher="slurm",
    n_tasks_per_env=n_tasks_per_CFD,
    sod_JSON="BluffBodySolverIncompAFC_pedri",
    N_CFDs=N_CFDs,
    control_parameters=None, 
    cwd=cwd,
)

# Initialize SOD2D postprocessor
postprocessor = SOD2DPostprocessor(
        N_CFDs=N_CFDs,
        mesh_name="mesh20_NACA",
        surf2analyse=1,
        n_tasks_per_env=n_tasks_per_CFD,
        initial_avgTime=20.0,
        rho=1.0,
        Uinf=1.0,
        Lref=1.0,
        Lz=1.0,
        AoA=20.0,
        cwd=cwd,
)


# ------------- LOAD CHECKPOINT / CREATE INITIAL TRAINING DATASET -----------------
if os.path.isfile(os.path.join(launcher.experiment_name, 'checkpoint.pt')):  # If the checkpoint.pt file exists -> restart point
    optimizer = BayesianOptimizer.load_checkpoint(path=launcher.experiment_name)

else:
    # Use Latin Hyercube Sampling method to select the initial points to evaluate (good coverage of the parameter space)
    sampler = qmc.LatinHypercube(d=len(bounds))
    samples = sampler.random(n=N_CFDs)

    bounds_min = np.array([bound[0] for bound in bounds.values()])
    bounds_max = np.array([bound[1] for bound in bounds.values()])

    train_X = qmc.scale(samples, bounds_min, bounds_max) # Shape (n, d) == (N_CFDs, Dimension of the parameter space)
    logger.info(f'Initial training X selected')

    # Launch SOD2D simulations with these parameters
    launcher.control_parameters = train_X
    launcher.execute()

    # Postprocess simulations and create initial surrogate
    Cl, Cd, Cl_std, Cd_std = postprocessor.compute_ClCd()
    train_y = compute_objective(Cl, Cd, Cl_std)
    logger.info(f'Initial training y computed')

    # Initialize BO optimizer
    optimizer = BayesianOptimizer(bounds, convergence_threshold=convergence_threshold,
                                  convergence_patience=convergence_patience, max_iterations=max_iterations)
    optimizer.initialize(train_X, train_y)

    # Save checkpoint
    optimizer.save_checkpoint(path=launcher.experiment_name, dump2csv=True)


# --------------- OPTIMIZATION LOOP ------------------------
converged = False

while not converged:

    # Current iteration being executed
    iteration = optimizer.get_iter()+1
    logger.info('\n')
    logger.info(f'ITERATION {iteration}')

    # Suggest new points
    new_train_X = optimizer.suggest(N_CFDs)
    logger.info(f'New training X selected')

    # Evaluate new points
    launcher.control_parameters = new_train_X
    launcher.execute()

    Cl, Cd, Cl_std, Cd_std = postprocessor.compute_ClCd()
    new_train_y = compute_objective(Cl, Cd, Cl_std)
    logger.info(f'New training y computed')

    # Update GP with the new observations
    optimizer.update(new_train_X, new_train_y)

    # Check convergence
    converged = optimizer.check_convergence()
    plot_convergence(optimizer, path=launcher.experiment_name)

    # Plot slices of the GP manifold
    if iteration%savePlot_step == 0:
        X_best, _ = optimizer.get_best()
        plot_parameter_slice(optimizer, X_best, path=launcher.experiment_name, include_iter=True)

    # Save checkpoint
    if iteration%saveCheckpoint_step == 0:
        optimizer.save_checkpoint(path=launcher.experiment_name, dump2csv=True)


import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.stats import qmc

from pedri.utils.geom_parametrization import parametrize_airfoil
from pedri.utils.geom_parametrization import bezier_airfoil
from pedri.utils.geom_parametrization import transform_ctlpts
from pedri.optimizer import BayesianOptimizer

import neuralfoil as nf 

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,                 # Set logging verbosity: INFO / DEBUG / WARNING /ERROR
    format='%(levelname)s - %(name)s - %(message)s'
)

plt.rcParams.update({'font.size': 18}) # Matplotlib fornt size


# ------------------------- PARAMETERS --------------------------------
filename = 'naca0012.dat' # Initial geometry

Re = 1000           # Reynolds number
AoA = 0             # Angle of attack

Ncte_upper = 3      # Number of control points upper surface
Ncte_lower = 3      # Number of control points lower surface
N = Ncte_upper + Ncte_lower # Total number of control points to optimize

Nini_samples = 100  # Number of samples of the initial dataset to start iterating
N_batch = 10        # Number of new samples in each BO iteration

convergence_error = 1   # Accepted relative error for convergence (in %)
max_iterations = 20     # Maximum number of allowed iteration in the BO optimization

nf_size = "medium"         # Neuralfoil model size: "xxsmall","xsmall","small","medium","large","xlarge","xxlarge","xxxlarge"

# Parameter bounds (We optimize perturbations (Ax,Ay) so that the final control point is ctePt = [x+Ax, y+Ay])
# Control points are ordered as follows: TE -> upper surf -> LE -> lower surf -> TE
bound_x = 0.1
bound_y = 0.025

keys = [f'A{axis}_{k}' for k in range(N) for axis in ('x', 'y')]
values = [(-bound_x, bound_x), (-bound_y, bound_y)] * N

bounds = {key:value for key, value in zip(keys, values)}

bounds[f'Ax_{Ncte_upper-1}'] = [0, bound_x] # Prevent the LE upper control point from moving upstream of the LE
bounds[f'Ax_{Ncte_upper}']   = [0, bound_x] # Prevent the LE lower control point from moving upstream of the LE

logger.info("Imposed bounds: %s", bounds)

# Define objective function during BO optimization
def compute_objective(Cl, Cd, Cl_base=0.0, Cd_base=0.12, alpha=0.5):
    return (Cd_base-Cd) + alpha*(Cl-Cl_base)


# --------------------- PARAMETRIZE AIRFOIL GEMOETRY ------------------
# Read airfoil coordinates
xy_airfoil = np.loadtxt(filename)

# Parametrize the airfoil
ctePts, xy_param = parametrize_airfoil(xy_airfoil, NctePts_upper=Ncte_upper, NctePts_lower=Ncte_lower)

# Plot the obtained airfoil
plt.figure(figsize=(14,5))
plt.plot(xy_airfoil[:,0], xy_airfoil[:,1], color='black', linewidth=2, label='original points')
plt.plot(xy_param[:,0], xy_param[:,1], '--', color='gray', linewidth=2, label='parametrized curve')
plt.plot(ctePts[:,0], ctePts[:,1], 'o', color='gray', markersize=10, markerfacecolor='none', markeredgewidth=2, label='control points')
plt.legend()
plt.xlabel('x/c')
plt.ylabel('y/c')
plt.axis('equal')
plt.tight_layout()
plt.savefig('Initial_geometry.png', dpi=150)
plt.close()

# Flatten ctePts and remove LE and TE
mask = (ctePts[:,0] > 0.0) & (ctePts[:,0] < 1.0)
ctePte_flat = ctePts[mask].flatten()


# -------------------- CREATE INITIAL TRAINING DATASET ---------------------
# Use Latin Hyercube Sampling method to select the initial points to evaluate (good coverage of the parameter space)
sampler = qmc.LatinHypercube(d=len(bounds))
samples = sampler.random(n=Nini_samples)

bounds_min = np.array([bound[0] for bound in bounds.values()])
bounds_max = np.array([bound[1] for bound in bounds.values()])

train_X = qmc.scale(samples, bounds_min, bounds_max) # Shape (n, d) == (Nini_samples, Dimension of the parameter space)

# Create initial gemoetries and compute the corresponding Cl and Cd
Cl = np.zeros(Nini_samples)
Cd = np.zeros(Nini_samples)

ncols = math.ceil(math.sqrt(Nini_samples ))
nrows = math.ceil(Nini_samples / ncols)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2.5, nrows*0.8), layout='constrained')
ax = ax.flatten()
for a in ax:
    a.axis('off')

for i in range(Nini_samples):

    # Create gemoetry
    new_ctePte_flat = ctePte_flat + train_X[i]   # This is [x1, y1, x2, y2, ...]
    new_ctePte = transform_ctlpts(new_ctePte_flat, Ncte_upper, Ncte_lower) # This is [[x1, y1], [x2, y2],...] + LE and TE

    airfoil_i = bezier_airfoil(xy_param[:,0], new_ctePte)

    # Plot the generated airfoil
    ax[i].plot(airfoil_i[:, 0], airfoil_i[:, 1], linewidth=0.5)
    #ax[i].plot(new_ctePte[:, 0], new_ctePte[:, 1], 'o', markersize=2)
    ax[i].set_aspect('equal', adjustable='box')
    ax[i].autoscale(enable=True, axis='both', tight=True)
    #ax[i].set_title(f'{i}')

    # Evaluate aero coefficients
    aero = nf.get_aero_from_coordinates(coordinates=airfoil_i, alpha=AoA, Re=Re, model_size=nf_size)
    Cl[i] = aero['CL'][0]
    Cd[i] = aero['CD'][0]

fig.savefig('Initial_dataset.png', dpi=150)
plt.close()

logger.info("Initial dataset created")
logger.info("Obtained Cl: %s", np.array2string(Cl, precision=2))
logger.info("Obtained Cd: %s", np.array2string(Cd, precision=2))

# Initialize BO optimizer
train_y = compute_objective(Cl, Cd)
optimizer = BayesianOptimizer(bounds)
optimizer.initialize(train_X, train_y)

# ------------------- OPTIMIZATION LOOP ---------------------
converged = False
iterations = 1
error_iterations = []
while converged==False:

    logger.info(f'ITERATION {iterations}')

    # Suggest next points
    train_X = optimizer.suggest(N_batch)
    logger.info(f'New training X: {train_X}')

    # Evaluate those points
    Cl = np.zeros(N_batch)
    Cd = np.zeros(N_batch)

    for i in range(N_batch):

        # Create gemoetry
        new_ctePte_flat = ctePte_flat + train_X[i]   # This is [x1, y1, x2, y2, ...]
        new_ctePte = transform_ctlpts(new_ctePte_flat, Ncte_upper, Ncte_lower) # This is [[x1, y1], [x2, y2],...] + LE and TE

        airfoil_i = bezier_airfoil(xy_param[:,0], new_ctePte)

        # Evaluate aero coefficients
        aero = nf.get_aero_from_coordinates(coordinates=airfoil_i, alpha=AoA, Re=Re, model_size=nf_size)
        Cl[i] = aero['CL'][0]
        Cd[i] = aero['CD'][0]

    train_y = compute_objective(Cl, Cd)
    logger.info(f'New training y: {train_y}')

    # Check convergence
    prediction_y, _ = optimizer.predict(train_X)                        # Predict those points with the current model
    relative_error = np.abs((train_y - prediction_y) / train_y)     # Real evaluation vs. model prediction

    max_error = 100 * np.max(relative_error)    # Express the maximum error in %
    error_iterations.append(max_error)          # Save relative error to plot it later
    logger.info(f'Max error (%): {max_error}')

    if (max_error < convergence_error) or (iterations > max_iterations):
        converged = True

    # Plot error during training
    plt.figure()
    plt.plot(range(0, len(error_iterations)), error_iterations, 'o-')
    plt.xlabel('BO iteration')
    plt.ylabel('Max relative error (%)')
    plt.tight_layout()
    plt.savefig('Convergence_error.png', dpi=150)
    plt.close()

    # Update GP surrogate with the new points
    optimizer.update(train_X, train_y)

    # Increase iteration counter
    iterations += 1

logger.info('CONVERGENCE ACHIEVED!')


# ------------------ PLOTS RESULTS -------------------
best_X, best_y = optimizer.get_best()

new_ctePte_flat = ctePte_flat + best_X   # This is [x1, y1, x2, y2, ...]
new_ctePte = transform_ctlpts(new_ctePte_flat, Ncte_upper, Ncte_lower) # This is [[x1, y1], [x2, y2],...] + LE and TE

airfoil_opt = bezier_airfoil(xy_param[:,0], new_ctePte)


plt.figure(figsize=(14,5))
plt.plot(xy_param[:,0], xy_param[:,1], '--', color='gray', linewidth=2, label='original airfoil')
plt.plot(airfoil_opt[:,0], airfoil_opt[:,1], '--', color='blue', linewidth=2, label='optimized airfoil')
plt.plot(ctePts[:,0], ctePts[:,1], 'o', color='gray', markersize=10, markerfacecolor='none', markeredgewidth=2, label='original control points')
plt.plot(new_ctePte[:,0], new_ctePte[:,1], 'o', color='blue', markersize=10, markerfacecolor='none', markeredgewidth=2, label='optimized control points')
plt.legend()
plt.xlabel('x/c')
plt.ylabel('y/c')
plt.axis('equal')
plt.tight_layout()
plt.savefig('Final_geometry.png', dpi=150)
plt.close()
import numpy as np
import matplotlib.pyplot as plt

from pedri.optimizer import BayesianOptimizer
from pedri.utils.plotting import plot_convergence


# ------------ DEFINE FUNCTION TO OPTIMIZE -------------
def branin(x):
    """
    Branin-Hoo function.

    x : array-like
        [x1, x2]

    output: float
        Function value to minimize.
    """
    x1, x2 = x

    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    return (a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1.0 - t) * np.cos(x1) + s)

def objective(x):
    # We want to find the mimima (not the maxima). Thus, include a "-" sign.
    # This should be f(x) \approx 0.398 at the points (-\pi, 12.275), (\pi, 2.275) or (3\pi, 2.475)
    return -branin(x)


# --------------- PARAMETERS ----------------
bounds = {"x1": [-5.0, 10.0], "x2": [0.0, 15.0]}    # Exploration bounds of the parameters

N_Samples = 10              # Number of samples used to update the GP each time

convergence_threshold = 0.1 # Minimum relative improvement between iterations (in %) defining the convergence threshold
convergence_patience = 5    # Number of consecutive iterations below the threshold required for convergence
max_iterations = 50         # Maximum number of BO iterations


# -------------  CREATE INITIAL TRAINING DATASET -----------------

# Initial evaluation
training_x1 = np.random.uniform(bounds["x1"][0], bounds["x1"][1], N_Samples)
training_x2 = np.random.uniform(bounds["x2"][0], bounds["x2"][1], N_Samples)

training_y = np.array([objective([x1,x2]) for x1, x2 in zip(training_x1, training_x2)])

# Initialize BO optimizer
training_x = np.array([training_x1, training_x2]).transpose()

optimizer = BayesianOptimizer(bounds, convergence_threshold=convergence_threshold,
                              convergence_patience=convergence_patience, max_iterations=max_iterations)
optimizer.initialize(training_x, training_y)


# --------------- OPTIMIZATION LOOP ------------------------
converged = False

while converged==False:

    # Current iteration being executed
    iteration = optimizer.get_iter()+1
    print(f'\nITERATION {iteration}')

    # Suggest next points
    new_training_x = optimizer.suggest(N_Samples)
    print('New training X selected')

    # Evaluate those points
    new_training_y = np.array([objective(x) for x in new_training_x])
    print('New training y computed')

    # Update GP surrogate with the new points
    optimizer.update(new_training_x, new_training_y)

    # Check convergence
    converged = optimizer.check_convergence()
    plot_convergence(optimizer)


# ---------------------- PLOT RESULTS -------------------------------
# Obtain best results
X_best, y_best = optimizer.get_best()
print('\nOBTAINED RESULTS: ')
print(f'Local minima at [x1, x2]: {X_best}')
print(f'Minima value f(x1,x2): {-y_best}')

# Plot
num_evals=100
X1_eval = np.linspace(bounds["x1"][0], bounds["x1"][1], num=num_evals)
X2_eval = np.linspace(bounds["x2"][0], bounds["x2"][1], num=num_evals)

X1_grid, X2_grid = np.meshgrid(X1_eval, X2_eval)

y_eval = np.array([branin([x1, x2]) for x1, x2 in zip(X1_grid.ravel(), X2_grid.ravel())]).reshape(X1_grid.shape)

plt.figure()
plt.contourf(X1_grid, X2_grid, y_eval, levels=50, cmap='viridis')
plt.plot(X_best[0], X_best[1], 'ro')
plt.xlabel('X1 param.')
plt.ylabel('X2 param.')
plt.tight_layout()
plt.savefig('Optimization_branin_test.png', dpi=300)

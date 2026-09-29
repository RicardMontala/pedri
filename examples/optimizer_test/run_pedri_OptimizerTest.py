import numpy as np
import matplotlib.pyplot as plt

from pedri.optimizer import BayesianOptimizer


# ------------------------------------------------------
# Function to optimize
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


# ------------------------------------------------------
# Define Parameters
bounds = {"x1": [-5.0, 10.0], "x2": [0.0, 15.0]}
N_CFDs = 10
convergence_error = 0.1 # Convergence relative error

# Initialize BO optimizer
optimizer = BayesianOptimizer(bounds)

# Initial evaluation
training_x1 = np.random.uniform(bounds["x1"][0], bounds["x1"][1], N_CFDs)
training_x2 = np.random.uniform(bounds["x2"][0], bounds["x2"][1], N_CFDs)

training_y = np.array([objective([x1,x2]) for x1, x2 in zip(training_x1, training_x2)])

#print('Initial dataset: ')
#print(training_x1)
#print(training_x2)
#print(training_y)

# Initial GP surrogate
training_x = np.array([training_x1, training_x2]).transpose()
#print('Training X: ', training_x)

optimizer.initialize(training_x, training_y)

# Optimization loop
converged = False
iterations = 0
while converged==False:

    print(f'ITERATION {iterations}')

    # Suggest next points
    new_training_x = optimizer.suggest(N_CFDs)
    #print('New training X: ', new_training_x)

    # Evaluate those points
    new_training_y = np.array([objective(x) for x in new_training_x])
    #print('New training Y: ', new_training_y)

    # Check convergence 
    prediction_y, _ = optimizer.predict(new_training_x) # Predict those points
    relative_error = np.abs((new_training_y - prediction_y) / new_training_y) # Real evaluation vs. model prediction

    max_error = np.max(relative_error)

    if max_error < convergence_error:
        converged = True

    # Update GP surrogate with the new points
    optimizer.update(new_training_x, new_training_y)

    iterations += 1

print('CONVERGENCE ACHIEVED!')

# Obtain best results
X_best, y_best = optimizer.get_best()
print('OBTAINED RESULTS: ')
print(f'Local minima at [x1, x2]: {X_best}')
print(f'Minima  value f(x1,x2): {-y_best}')


# ------------------------------------------------------
# Analyse results
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

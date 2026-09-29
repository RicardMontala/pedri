import os
import numpy as np
import matplotlib.pyplot as plt


def plot_parameter_slice(optimizer, bounds, X,
                         n_points=50, confidence=1.96, path='./', iter=None):
    """
    Plot the GP prediction while varying one parameter and
    keeping all other parameters fixed at the values indicated in X.

    optimizer: pedri class
        Optimizer class used during the optimization.

    bounds: dictionary {key: value}
        Keys are the different parameters being optimized and value is an array containing the (min, max) values for the parameter.

    X: input array (N_params,)
        List containing the parameters where the slice will be performed.

    n_points: int, optional
         Number of points used to sweep the varied parameter.

    confidende: float, optional
        Multiplier used to define the confidence interval around the
        GP mean prediction (e.g., 1.96 corresponds approximately
        to a 95% confidence interval for a Gaussian distribution).

    path: sting, optional
        Path to the directory where the image will be saved.

    iter: int, optional
        Optimization iteration to append to the output filename.
    """

    # For each control parameter
    for i, (key, bound) in enumerate(bounds.items()):
    
            # Values of the parameter to sweep
            test_param = np.linspace(bound[0], bound[1], n_points)
    
            # Create an array of (n_points, 1) with each row containing the repeated parameters "X"
            test_params = np.tile(X, (n_points, 1))
    
            # Replace only the column of the parameter currently being analysed "i"
            test_params[:, i] = test_param
    
            # GP prediction
            test_mean, test_std = optimizer.predict(test_params)
    
            # Create plot
            plt.figure()
    
            # Plot GP
            plt.plot(test_param, test_mean, label="GP mean")
            plt.fill_between(test_param, test_mean - confidence*test_std, test_mean + confidence*test_std, alpha=0.2, label=f"{confidence:.2f}% confidence interval")
    
            # Plot the actual CFD points
            plt.scatter(optimizer.training_X[:, i], optimizer.training_y, color="black", label="CFD data")
    
            # Other options
            plt.xlabel(key)
            plt.ylabel("Objective")
            plt.legend()
    
            title_params_keys = {key: value for j, (key, value) in enumerate(bounds.items()) if j != i}
            title_params = np.delete(X, i)
            title = ", ".join(f"{key}={value:.3f}" for key, value in zip(title_params_keys, title_params))
            plt.title(title)
    
            # Save figure
            if iter is None:
                filename = f'{key}_slice.png'
            else:
                filename  = f'{key}_slice_iter{iter}.png'

            filename = os.path.join(path, filename )

            plt.savefig(filename, dpi=150)
            plt.close()
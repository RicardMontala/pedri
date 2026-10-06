import os
import numpy as np
import matplotlib.pyplot as plt


def plot_parameter_slice(optimizer, X,
                         n_points=50, confidence=1.96, path='./', include_iter=False):
    """
    Plot the GP prediction while varying one parameter and
    keeping all other parameters fixed at the values indicated in X.

    optimizer: pedri class
        Optimizer class used during the optimization.

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

    iter: bool, optional
        If True, the BO iteration is appended to the output filename.
    """

    # For each control parameter
    bounds = optimizer.pbounds
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
            if not include_iter:
                filename = f'{key}_slice.png'
            else:
                iter = optimizer.get_iter()
                filename  = f'{key}_slice_iter{iter}.png'

            filename = os.path.join(path, filename )

            plt.savefig(filename, dpi=150)
            plt.close()


def plot_convergence(optimizer, path='./',):
        '''
        Plot the maximum objective value at each iteration, as well as the
        relative absolute change with respect to the previous iteration.
        '''

        
        best_y_iterations = np.asarray(optimizer.best_history)

        relative_change_iterations = (100 * np.abs(best_y_iterations[1:] - best_y_iterations[:-1])/ np.maximum(np.abs(best_y_iterations[:-1]), 1e-12))

        iterations  = np.arange(len(best_y_iterations)) # position 0 is the initial dataset, while the remaining ones are the BO iterations (1,2,...)

        fig, ax1 = plt.subplots()
        ax1.plot(iterations, best_y_iterations, 'o-')
        ax1.set_xlabel('BO iteration')
        ax1.set_ylabel('Best objective value (-)')
        ax2 = ax1.twinx()
        ax2.plot(iterations[1:], relative_change_iterations,'s--')
        ax2.set_ylabel('Relative change in % (--)')
        fig.tight_layout()
        fig.savefig(os.path.join(path,'convergence.png'), dpi=150)
        plt.close(fig)
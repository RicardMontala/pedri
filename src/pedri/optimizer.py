import torch
import numpy as np
import os

# https://botorch.org/
# https://github.com/meta-pytorch/botorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf

from gpytorch.mlls import ExactMarginalLogLikelihood

import logging
logger = logging.getLogger(__name__)


class BayesianOptimizer:

    def __init__(self, pbounds):
        '''
        pbounds: dict
            Dictionary with parameters names as keys and a tuple with minimum and maximum values.
            e.g. {'A_DC': [-1.0, 1.0], 'A':[0, 0.5], 'f':[0.1, 1.0]}

        '''

        self.pbounds = pbounds

        # Number of control parameters
        self.n_parameters = len(pbounds)

        # Bounds as a tensor:
        # [[lower_1, lower_2, ...],
        #  [upper_1, upper_2, ...]]
        self.bounds = torch.tensor(
            [
                [pbounds[name][0] for name in pbounds],
                [pbounds[name][1] for name in pbounds],
            ],
            dtype=torch.double,
        )

        # Training data
        self.training_X = None
        self.training_y = None

        # Gaussian Process
        self.model = None


    def initialize(self, X, y):
        '''
        Register existing CFD simulation data.

        X: array
            Set of parameters values to train the model (N_samples, N_parameters)

        y: array
            Set of objective values to train the model (N_samples)
        '''

        self.training_X = torch.tensor(X, dtype=torch.double)
        self.training_y = torch.tensor(y, dtype=torch.double).reshape(-1, 1)

        self._fit_model()


    def update(self, X, y):
        '''
        Register new CFD simulation results.

        X: array
            Set of parameters values to train the model (N_samples, N_parameters)

        y: array
            Set of objective values to train the model (N_samples)
        '''

        X = torch.tensor(X, dtype=torch.double)
        y = torch.tensor(y, dtype=torch.double).reshape(-1, 1)

        if self.training_X is None:

            self.initialize(X, y)
            return

        self.training_X = torch.cat([self.training_X, X], dim=0)
        self.training_y = torch.cat([self.training_y, y], dim=0)

        self._fit_model()


    def _fit_model(self):
        """
        Internal function to fit the Gaussian Process regression Model according to the available (training_X, training_y) data.
        """

        logger.info(f"Fitting GP with {self.training_X.shape[0]} samples...")

        self.model = SingleTaskGP(train_X=self.training_X, train_Y=self.training_y,
            input_transform=Normalize(d=self.n_parameters),
            outcome_transform=Standardize(m=1),
            )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        fit_gpytorch_mll(mll)


    def suggest(self, N_CFDs):
        """
        Suggest the next CFD cases to run.
        
        N_CFDs: int
            Desired number of new set of parameters (matching the allocated number of CFD simulations available)

        Output: numpy array
            Contains the new set of parameters, i.e, (N_CFDs, N_parameters)
        """

        if self.model is None:
            raise RuntimeError("The Bayesian model has not been initialized.")

        # Batch acquisition function
        acquisition = qLogExpectedImprovement(
            model=self.model,
            best_f=self.training_y.max(),
        )

        # Find N_CFDs points simultaneously
        candidates, acquisition_value = optimize_acqf(
            acq_function=acquisition,
            bounds=self.bounds,
            q=N_CFDs,
            num_restarts=10,
            raw_samples=512,
        )

        return candidates.detach().cpu().numpy()


    def predict(self, X):
        """
        Predict the target values for the given input points.

        X : array
            N input points with shape (N, N_parameters).

        Output 1: numpy array
            Predicted mean target values with shape (N,).

        Output 2: numpy array
            Standard deviation of the target values with shape (N,).
        """

        if self.model is None:
            raise RuntimeError("The GP model has not been initialized.")

        X = torch.as_tensor(X, dtype=torch.double)

        posterior = self.model.posterior(X)

        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)
        
        return (
            mean.detach().cpu().numpy(),
            std.detach().cpu().numpy()
        )


    def get_best(self):
        '''
        Return the best observed objective and its correspondng parameters.

        Output 1: numpy array
            Parameters corresponding to the best observed value with shape (N_parameters,).

        Output 2: float
            Best observed objective value
        '''

        index = torch.argmax(self.training_y)

        return (
            self.training_X[index].detach().cpu().numpy(),
            self.training_y[index].item(),
        )


    def get_dataset(self):
        '''
        Return the current training dataset containing N_samples.

        Output 1: numpy array
            Observed parameter values with shape (N_samples, N_parameters).

        Output 2: numpy array
            Observed objective values with shape (N_samples,).
        '''

        return (
            self.training_X.detach().cpu().numpy(),
            self.training_y.squeeze(-1).detach().cpu().numpy(),
        )


    def save_checkpoint(self, iteration, path='./', dump2csv=False):
        '''
        Save the current BO state, including the training dataset, parameter
        bounds, and current iteration. The file is called checkpoint.pt.

        iteration: int
            Current BO iteration.

        path: sting, optional
            Direcotry where the checkpoint file will be stored.

        dump2csv: bool, optional
            If True, also save the training dataset to a CSV file.
        '''

        os.makedirs(path, exist_ok=True)

        # Save in torch binary format (can be used later by "load_checkpoint()" routine)
        checkpoint = {
            "training_X": self.training_X,
            "training_y": self.training_y,
            "pbounds": self.pbounds,
            "iteration": iteration,
        }
        torch.save(checkpoint, os.path.join(path, 'checkpoint.pt'))

        # Save in CSV
        if dump2csv:
            X, y = self.get_dataset()
            data = np.column_stack((X, y))

            parameter_names = list(self.pbounds.keys())
            header = parameter_names + ["objective"]

            np.savetxt(os.path.join(path, 'checkpoint.csv'), data, delimiter=",", header=",".join(header))


    @staticmethod
    def load_checkpoint(path='./'):
        '''
        Load a BO checkpoint saved in checkpoint.pt file.

        path: str, optional
            Directory where the checkpoint file is stored.

        output 1: class
            Pedri bayesian optimizer initialized with the data stored in the checkpoint.

        output 2: int
            BO iteration at which the checkpoint was saved.
        '''

        checkpoint = torch.load(os.path.join(path, 'checkpoint.pt'), weights_only=False)

        optimizer = BayesianOptimizer(checkpoint["pbounds"])
        optimizer.initialize(checkpoint["training_X"], checkpoint["training_y"])

        logger.info(f"Optimization restarted from iteration {checkpoint['iteration']}.")
        logger.info(f"Continuing optimization...")

        return optimizer, checkpoint["iteration"]
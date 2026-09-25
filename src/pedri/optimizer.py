import torch

from smartsim.log import get_logger

# https://botorch.org/
# https://github.com/meta-pytorch/botorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf

from gpytorch.mlls import ExactMarginalLogLikelihood


logger = get_logger(__name__)


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

        X: array [N_samples, N_parameters]
        y: array [N_samples]
        '''

        self.training_X = torch.tensor(X, dtype=torch.double)
        self.training_y = torch.tensor(y, dtype=torch.double).reshape(-1, 1)

        self._fit_model()


    def update(self, X, y):
        '''
        Register new CFD simulation results.
        """
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
        Internal function to fit the Gaussian Process regression Model according to the available [training_X, training_y] data.
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
            Contains the new set of parameters, i.e, [N_CFDs, N_parameters]
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

        Output : numpy array
            Predicted mean target values with shape (N,).
        """

        if self.model is None:
            raise RuntimeError("The GP model has not been initialized.")

        X = torch.as_tensor(X, dtype=torch.double)

        posterior = self.model.posterior(X)

        y_pred = posterior.mean.squeeze(-1)

        return y_pred.detach().cpu().numpy()


    def get_best(self):

        index = torch.argmax(self.training_y)

        return (
            self.training_X[index].detach().cpu().numpy(),
            self.training_y[index].item(),
        )
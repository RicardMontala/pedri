import os
import logging
import numpy as np
import torch

# https://botorch.org/
# https://github.com/meta-pytorch/botorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import (optimize_acqf,
                           optimize_acqf_mixed_alternating,)
from gpytorch.mlls import ExactMarginalLogLikelihood

from pedri.optimizer.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian Optimization using BoTorch SingleTaskGP and qLogExpectedImprovement.
    """

    def __init__(
        self,
        pbounds: dict[str, tuple[float, float]],
        parameter_types: dict[str, str] | None = None,
        convergence_threshold: float = 0.1,
        convergence_patience: int = 5,
        max_iterations: int = 50,
        dtype: torch.dtype = torch.double,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            pbounds=pbounds,
            parameter_types=parameter_types,
            convergence_threshold=convergence_threshold,
            convergence_patience=convergence_patience,
            max_iterations=max_iterations,
        )

        self.dtype = dtype
        self.device = torch.device(device)

        # PyTorch Tensor Bounds: shape (2, n_parameters)
        # [[lower_1, lower_2, ...],
        #  [upper_1, upper_2, ...]]
        self.bounds = torch.tensor(
            [
                [pbounds[name][0] for name in pbounds],
                [pbounds[name][1] for name in pbounds],
            ],
            dtype=self.dtype,
            device=self.device,
        )

        self.model: SingleTaskGP | None = None

        # For 'discrete' parameters, construct candidate arrays
        # e.g., {1: [0..9], 3: [2..3]} means that parameters 1 and 3
        # are discrete with the corresponding candidate values
        self.discrete_dims = {}
        for idx, name in enumerate(self.parameter_names):
            if self.parameter_types[name] == "discrete":
                # Extract lower and upper bounds for this parameter
                min_val = int(self.bounds[0, idx].item())
                max_val = int(self.bounds[1, idx].item())
                
                # Map feature index -> list of allowed values
                self.discrete_dims[idx] = list(range(min_val, max_val + 1))


    def _fit_model(self) -> None:
        """
        Fit the Gaussian Process regression model according
        to the available (training_X, training_y) data.
        """

        if self.training_X is None or self.training_y is None:
            return

        logger.info(f"Fitting GP with {self.training_X.shape[0]} samples...")

        train_X_t = torch.tensor(self.training_X, dtype=self.dtype, device=self.device)
        train_y_t = torch.tensor(self.training_y, dtype=self.dtype, device=self.device)

        self.model = SingleTaskGP(
            train_X=train_X_t,
            train_Y=train_y_t,
            input_transform=Normalize(d=self.n_parameters),
            outcome_transform=Standardize(m=1),
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)


    def suggest(self, n_samples: int) -> np.ndarray:
        """
        Suggest new candidate parameter sets via qLogExpectedImprovement.

        Parameters
        ----------
        n_samples: int
            Desired number of new set of parameters

        Returns
        -------
        candidates: np.ndarray
            Contains the new set of parameters, i.e, (n_samples, m_parameters)
        """
        
        if self.model is None or self.training_y is None:
            raise RuntimeError("The Bayesian model has not been initialized.")

        train_y_t = torch.tensor(self.training_y, dtype=self.dtype, device=self.device)

        acquisition = qLogExpectedImprovement(
            model=self.model,
            best_f=train_y_t.max(),
        )

        if len(self.discrete_dims) == 0:
            # All parameters are continuous
            candidates, acquisition_value = optimize_acqf(
                acq_function=acquisition,
                bounds=self.bounds,
                q=n_samples,
                num_restarts=10,
                raw_samples=512,
            )

        else:
            # Mixed discrete/continuous parameters
            candidates, acquisition_value = optimize_acqf_mixed_alternating(
                acq_function=acquisition,
                bounds=self.bounds,
                discrete_dims=self.discrete_dims,
                q=n_samples,
                num_restarts=10,
                raw_samples=512,
            )

        return candidates.detach().cpu().numpy()


    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the target values for the given input points.

        Parameters
        ----------
        X: np.ndarray
            N input points with shape (N, n_parameters).

        Returns
        -------
        mean: np.ndarray
            Predicted mean target values with shape (N,).

        std: np.ndarray
            Standard deviation of the target values with shape (N,).
        
        """

        if self.model is None:
            raise RuntimeError("The GP model has not been initialized.")

        X_tensor = torch.as_tensor(X, dtype=self.dtype, device=self.device)

        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.sqrt().squeeze(-1)

        return (
            mean.detach().cpu().numpy(),
            std.detach().cpu().numpy(),
        )

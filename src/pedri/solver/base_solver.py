from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseSolver(ABC):
    """
    Abstract Base Class for all PEDRI solvers (CFD, surrogate models, analytical codes).
    """

    def __init__(self, objective_fn: Callable[..., np.ndarray | float] | None = None):
        """
        Parameters
        ----------
        objective_fn : Callable, optional
            Function that converts raw solver metrics into scalar objective values y.
            Accepts named metric arrays as keyword arguments (e.g., fn(Cl, Cd, **kwargs)).
        """
        self.objective_fn = objective_fn


    @abstractmethod
    def run(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Execute simulation/surrogate for candidate parameters X.

        Parameters
        ----------
        X: np.ndarray
            Parameter matrix of shape (N_samples, n_parameters).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping variable names to 1D output arrays of length N_samples.
            e.g., {"Cl": np.ndarray, "Cd": np.ndarray, "Cl_std": np.ndarray}
        """

        pass


    def evaluate(self, X: np.ndarray, objective_fn: Callable[..., np.ndarray | float] | None = None,
            ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Run simulations and map raw outputs to scalar objective values y.

        Parameters
        ----------
        X: np.ndarray
            Parameter matrix of shape (n_samples, n_parameters).
        objective_fn: Callable, optional
            Override default objective function for this call.

        Returns
        -------
        y: np.ndarray
            Scalar objective values of shape (N_samples,).
        raw_metrics: dict[str, np.ndarray]
            Complete physical outputs dictionary for logging and visualization.
        """

        raw_metrics = self.run(X)

        fn = objective_fn or self.objective_fn
        if fn is None:
            raise ValueError("No objective_fn defined. Pass objective_fn to BaseSolver __init__ or to evaluate().")

        # Unpack raw metrics directly into objective_fn keyword arguments
        y = fn(**raw_metrics)
        y_arr = np.asarray(y, dtype=np.float64)

        return y_arr, raw_metrics
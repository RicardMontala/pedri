from abc import ABC, abstractmethod

import numpy as np


class BaseParametrization(ABC):
    """Base interface for airfoil parametrization methods."""

    def __init__(self, n_points=200, relative_mode:bool=False):
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")

        self.n_points = n_points
        self.reference_parameters = None
        self.relative_mode = relative_mode


    @abstractmethod
    def get_initial_parameters(self, target_airfoil=None):
        """
        Determine the reference parameters.
        """

        raise NotImplementedError


    @abstractmethod
    def generate_geometry(self, params):
        """
        Generate airfoil coordinates from parameters.
        """

        raise NotImplementedError


    def generate_perturbed_geometry(self, params:np.ndarray) -> np.ndarray:
        """
        Generate new geometry from candidate parameters coming from the optimizer.

        Parameters
        ----------
        params: np.ndarray
            Candidate parameters (n_parameters,). If relative_mode=True,
            the incoming params represent perturbations relative to
            reference_parameters (X = X_ref + AX). Otherwise, they
            already represent the absolute parametrization parameters.

        Returns
        -------
        airfoil_xy: np.ndarray
            Generated geometry coordinates.
        """

        params = np.asarray(params, dtype=float)

        if self.relative_mode:
            if self.reference_parameters is None:
                raise RuntimeError("Call get_initial_parameters() before generating geometry.")
            if params.shape != self.reference_parameters.shape:
                raise ValueError("Perturbations must have the same shape as reference_parameters.")

            params = self.reference_parameters + params

        return self.generate_geometry(params)

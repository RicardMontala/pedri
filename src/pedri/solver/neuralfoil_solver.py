from typing import Callable
import numpy as np

# https://github.com/peterdsharpe/NeuralFoil
import neuralfoil as nf

from pedri.solver.base_solver import BaseSolver
from pedri.parametrization import BaseParametrization


class NeuralFoilSolver(BaseSolver):
    """
    Surrogate solver for fast airfoil aerodynamic predictions using NeuralFoil.
    """

    def __init__(
        self,
        geom_parametrization: BaseParametrization,
        reynolds: float = 1e4,
        aoa: float = 0.0,
        model_size: str = "medium",
        objective_fn: Callable[..., np.ndarray | float] | None = None,
    ):
        """
        Parameters
        ----------
        geom_transform_fn: Callable
            Function mapping a 1D parameter array x_i to airfoil xy coordinates (N_points, 2).
        reynolds: float
            Reynolds number.
        aoa: float
            Angle of attack in degrees.
        model_size: str
            NeuralFoil model size ("xxsmall","xsmall","small","medium","large","xlarge","xxlarge","xxxlarge").
        objective_fn: Callable, optional
            Objective mapping function.
        """

        super().__init__(objective_fn=objective_fn)
        self.geom_patrametrization = geom_parametrization
        self.reynolds = reynolds
        self.aoa = aoa
        self.model_size = model_size


    def run(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute lift, drag and momentum coefficients for the airfoil gemoetries defined by X.

        Parameters
        ----------
        X: np.ndarray
            Parameters defining the parametrized curve of the airfoil (n_samples, n_parameters).

        Returns
        -------
        Results: dictionary
            Dictionary containing the lift, drag, and moment coefficients for each airfoil geometry.
            Each coefficient array has shape (n_samples,).
        """

        n_samples = X.shape[0]

        cl = np.zeros(n_samples)
        cd = np.zeros(n_samples)
        cm = np.zeros(n_samples)

        for i in range(n_samples):
            coords = self.geom_patrametrization.generate_perturbed_geometry(X[i])

            aero = nf.get_aero_from_coordinates(
                coordinates=coords,
                alpha=self.aoa,
                Re=self.reynolds,
                model_size=self.model_size,
            )

            cl[i] = aero["CL"][0]
            cd[i] = aero["CD"][0]
            cm[i] = aero.get("CM", [0.0])[0]

        return {"Cl": cl, "Cd": cd, "Cm": cm}
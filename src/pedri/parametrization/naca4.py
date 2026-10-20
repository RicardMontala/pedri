import logging

import numpy as np

from pedri.parametrization.base import BaseParametrization
from pedri.parametrization.utils.naca4_utils import (
    naca4_surface,
)
from pedri.parametrization.utils.generic_utils import (
    construct_spline,
    order_airfoil,
    distribute_points,
)

logger = logging.getLogger(__name__)


class NACA4Parametrization(BaseParametrization):
    """
    Airfoil parametrization using the NACA 4-digit parametrization.

    The free parameters are the following 3 parameters (e.g NACA 2412):
        m: maximum camber as a percentage of chord (1st digit, e.g. 2%)
        p: position of maximum camber, in tenths of chord (2nd digit, e.g., 40%)
        t: Maximum thickness as a percentage of chord (Last two digits, e.g., 12%)
    """

    def __init__(
            self,
            n_points=100,
            relative_mode=False,
        ):
            super().__init__(n_points=n_points, relative_mode=relative_mode)

            self.n_parameters = 3

            self.reference_parameters = None
            self.reference_geometry = None
            self.reference_x = None


    def get_initial_parameters(self, target_airfoil:str) -> np.ndarray:
        """
        From a string describing the 4 digits of the NACA initial airfoil,
        return the parameters array to be optimized.

        Parameters
        ----------
        target_arifoil: string
            String containing four digits describing the initial NACA airfoil.

        Returns
        -------
        reference_parameters: np.ndarray
            Numpy array containing the 3 parameters describing the airfoil,
            i.e., [m, p, t]
        """

        if len(target_airfoil) != 4 or not target_airfoil.isdigit():
            raise ValueError(
                "A 4-digit NACA airfoil must be defined using exactly "
                f"4 digits. Instead, '{target_airfoil}' was provided."
            )

        m = int(target_airfoil[0])
        p = int(target_airfoil[1])
        t = int(target_airfoil[2:4])

        self.reference_parameters = np.array([m, p, t])

        beta = np.linspace(0.0, np.pi, self.n_points)
        self.reference_x = 0.5 * (1.0 - np.cos(beta)) # Cosinus x-distribution [0, 1]

        self.reference_geometry = self.generate_geometry(self.reference_parameters)

        return self.reference_parameters.copy()


    def generate_geometry(self, params, x_coords=None):
        """
        Generate geometry from 4 digits provided in params

        Parameters
        ----------
        params: np.ndarray 
            NACA 4-digits [m, p, t] parameters.

        x_coords : np.ndarray, optional
            Locations at which to evaluate the airfoil. By default,
            use the reference airfoil's x-coordinates.
        """

        if params.size != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} parameters, "
                f"got {params.size}."
            )

        if x_coords is None:
            if self.reference_x is None:
                raise RuntimeError(
                    "Call get_initial_parameters() first, or provide "
                    "x_coords explicitly."
                )
            x_coords = self.reference_x

        return naca4_surface(x_coords, params[0], params[1], params[2])
         
        

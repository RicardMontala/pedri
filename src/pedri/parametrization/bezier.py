import logging

import numpy as np
from scipy.optimize import minimize

from pedri.parametrization.base import BaseParametrization
from pedri.parametrization.utils.bezier_utils import (
    evaluate_surface,
)
from pedri.parametrization.utils.generic_utils import (
    construct_spline,
    order_airfoil,
    distribute_points,
)

logger = logging.getLogger(__name__)


class BezierParametrization(BaseParametrization):
    """
    Airfoil parametrization using piecewise quadratic Bezier curves.

    The free parameters are control-point coordinates, ordered as:

        [x_upper_1, y_upper_1, ..., x_lower_1, y_lower_1, ...]

    The leading edge and trailing edge are fixed at (0, 0) and (1, 0).
    """

    def __init__(
        self,
        n_ctlPts_upper=3,
        n_ctlPts_lower=3,
        n_points=100,
        relative_mode=True,
        bound_x=0.1,
        bound_y=0.025,
    ):
        super().__init__(n_points=n_points, relative_mode=relative_mode)

        if n_ctlPts_upper < 1 or n_ctlPts_lower < 1:
            raise ValueError(
                "At least one free control point is required on each surface."
            )

        self.n_ctlPts_upper = n_ctlPts_upper
        self.n_ctlPts_lower = n_ctlPts_lower
        self.n_parameters = 2 * (n_ctlPts_upper + n_ctlPts_lower)

        self.bound_x = bound_x
        self.bound_y = bound_y

        self.reference_parameters = None
        self.reference_geometry = None
        self.reference_x = None
        self.control_points = None


    def get_initial_parameters(self, target_airfoil,
                                split_points=np.array([[0.2,1.0],[1.0,0.0],[0.2,-1.0]]),
                                distribution='linear', max_iter=10000):
        '''
        From a list of xy points defining the airfoil geometry, create a mathching airfoil using Bezier curves.
        It generates the gemoetry using N = (NctePts_upper + NctePts_lower) control points, plus two extra control points:
        one at the LE (0,0) and the other at the TE (1,0).

        Parameters
        ----------
        target_airfoil: array (n_points, 2)
            List of n_points defining the (x,y) points of the airfoil.
            This is usually the .dat taken from airfoiltools.com.
            Points should go from x=[0,1].

        split_points: array (n_split, 2)
            Coordinates of the n_split points at which the spline is split
            into separate segments. If None, a single spline is constructed
            using all input points.
            The default points split the geometry into three curves: the upper
            surface, the leading-edge region, and the lower surface. This helps
            prevent undesirable geometries from being generated during the
            creation of the initial spline.

        distribution: string, optional
            How to distribute the intial control points to start optimizing.
            'cosine' or 'linear' available.

        max_iter: int, optional
            Maximum number of iterations allowed during the optimization process.

        Returns
        -------
        output 1: array (NctePts_upper + NctePts_lower + 2, 2)
            List of obtained control points
        
        ouutput 2: array(n_points, 2)
            xy points of the matched airfoil
        '''

        target_airfoil = np.asarray(target_airfoil, dtype=float)

        if (target_airfoil.shape[1] != 2):
            raise ValueError("target_airfoil must be an (N, 2) coordinate array.")

        # Create the spline of the airfoil
        xy = construct_spline(target_airfoil, n_points=self.n_points, split_points=split_points)

        # Order in the convenient order used in the following functions
        xy = order_airfoil(xy)

        # Initial guess provided to the optimizer (x1, y1, x2, y2, x3, y3, ..., x_N, y_N)
        initial_ctlPts = self._create_initial_distribution_ctlPts(xy, distribution)

        self.reference_x = np.clip(xy[:, 0], 0.0, 1.0)

        # Define the cost function to find the matching airfoil. We use the summ of squared error of the y-coordinate (is what we find)
        def shape_match(control_pts):

            '''
            Create cost function to evalute the matching between the created Bezier curve and the original (x,y) coordinates.

            control_pts: ndarray (n,)
                Contains the n parameters optimized (not considering the LE and TE)
            '''

            # Create Bezier airfoil surface with current parametrs "control_pts"
            try:
                text_xy = self.generate_geometry(control_pts, x_coords=self.reference_x)

                # Return cost function
                return np.sum((text_xy[:,1] - xy[:,1])**2) 
            
            except Exception:
                return 1e10 # When finding an invalid gemoetry, which breaks the logic of the Bezier curves, impose a high cost

        # Minimize the cost function to obtain the control points generating the matching airfoil
        result = minimize(shape_match, initial_ctlPts, method="L-BFGS-B", tol=1e-10, options={"maxiter": max_iter})
        logger.info("Bezier fitting completed.")
        logger.info("Optimization successful: %s", result.success)
        logger.info("Iterations: %d", result.nit)
        logger.info("Function evaluations: %d", result.nfev)
        logger.info("Final objective: %.6e", result.fun)

        self.reference_parameters = result.x.copy()
        self.control_points = self.transform_ctlpts(self.reference_parameters)
        self.reference_geometry = self.generate_geometry(self.reference_parameters)

        return self.reference_parameters.copy()


    def get_design_bounds(self):
        """Return bounds on control-point perturbations."""
        bounds = {}

        for i in range(self.n_ctlPts_upper + self.n_ctlPts_lower):
            bounds[f"Ax_{i}"] = (-self.bound_x, self.bound_x)
            bounds[f"Ay_{i}"] = (-self.bound_y, self.bound_y)

        # Restrict the x-displacement of the two control points
        # closest to the leading edge, following the original workflow.
        bounds[f"Ax_{self.n_ctlPts_upper - 1}"] = (0.0, self.bound_x)
        bounds[f"Ax_{self.n_ctlPts_upper}"] = (0.0, self.bound_x)

        return bounds


    def generate_geometry(self, params, x_coords=None):
        """
        Generate geometry from absolute control-point coordinates.

        Parameters
        ----------
        params : array_like
            Absolute free control-point coordinates, not perturbations.
        x_coords : array_like, optional
            Locations at which to evaluate the airfoil. By default,
            use the reference airfoil's x-coordinates.
        """
        #params = self.validate_parameters(params)

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

        control_points = self.transform_ctlpts(params)

        return self.bezier_airfoil(x_coords, control_points)


    def _create_initial_distribution_ctlPts(self, xy, distribution):
        '''
        Creates an initial set of N=(NctePts_upper + NctePts_lower) control points
        to start the optimization process of matching the airfoil shape.

        xy: array(n_points, 2)
            List of (x,y) points defining the airfoil geometry. The control points will fall within this curve (as an initial guess)
            Assumes that airfoil is already ordered uisng the convention TE -> upper -> LE -> lower -> TE

        distribution: string
            Distribution used to place the points along the airfoil xy surface.
            Currently available ['linear', 'cosine']

        output: ((Ncte_upper+Ncte_lower)*2,)
            List of control points already in the optimizer format [p01_x, p01_y, p02_x, p02_y, ...]
        '''

        # Find leading edge location
        id_le = np.argmin(xy[:,0])

        # Split into upper and lower surfaces
        upper = xy[:id_le+1]
        lower = xy[id_le:]

        upper_ids = distribute_points(upper, self.n_ctlPts_upper, distribution)
        lower_ids = distribute_points(lower, self.n_ctlPts_lower, distribution)

        return np.vstack((upper[upper_ids], lower[lower_ids])).flatten()


    def transform_ctlpts(self, ctlPts):
        """
        Transform the outputs from a given optimizer (list of params) into a form
        that's suitable for the bezier function input ctlPts.

        Parameters
        ----------
        ctl_pts = [px1, py1, px2, py2,...,pxn, pyn]

        Returns
        -------
        ctl_pts
        Ordering: TE -> upper -> LE -> lower -> TE.
        """
        ctlPts = np.asarray(ctlPts, dtype=float)

        expected = 2 * (self.n_ctlPts_upper + self.n_ctlPts_lower)

        if ctlPts.size != expected:
            raise ValueError(f"Expected {expected} parameters, got {ctlPts.size}.")

        free_pts = ctlPts.reshape(-1, 2)

        te = np.array([[1.0, 0.0]])
        le = np.array([[0.0, 0.0]])

        upper = free_pts[:self.n_ctlPts_upper]
        lower = free_pts[self.n_ctlPts_upper:]

        return np.vstack((te, upper, le, lower, te))


    def bezier_airfoil(self, x_coords, ctl_pts):
        """
        Geometry-generation function. Given a set of Bezier control points, generate the complete airfoil surface.
        The function uses te midpoints between control points to determine where one Bezier segment transitions into the next.

        x_coords: array (n_points, )
            List of x-coordinates where you want the airfoil surface to be evaluated (corresponding y coordinate will be found).
            x_coords must follow the convention ordering: TE -> upper -> LE -> lower -> TE. Use function "order_airfoil" for so.
            LE is (0,0) and TE is (1,0)

        ctlPts: array (N_ctePts, 2)
            List of N_ctePts control points (x,y) defining the airfoil shape.

        output: array (n_points, 2)
            List of xy-coordinates defining the airfoil surface using the control points defined in ctlPts
        """

        x_coords = np.asarray(x_coords, dtype=float)
        ctl_pts = np.asarray(ctl_pts, dtype=float)

        le_idx = np.argmin(x_coords)

        x_upper = x_coords[:le_idx + 1]
        x_lower = x_coords[le_idx:]

        n_ctl = len(ctl_pts)
        le_ctl_idx = (n_ctl - 1) // 2

        ctl_upper = ctl_pts[:le_ctl_idx + 1]
        ctl_lower = ctl_pts[le_ctl_idx:]

        y_upper = evaluate_surface(x_upper, ctl_upper)
        y_lower = evaluate_surface(x_lower, ctl_lower)

        y = np.concatenate((y_upper, y_lower[1:]))

        return np.column_stack((x_coords, y))
import numpy as np

import logging
logger = logging.getLogger(__name__)


def _ctePts_check_xordering(ctePts, NctePts_upper, NctePts_lower):
    ''''
    Return False if the control points violate their x-ordering.

    ctePts: array (NctePts_upper + NctePts_lower + 2, 2)
        Control points xy coordinates following the
        TE -> upper -> LE -> lower -> TE convention
        and containing the LE and TE.

    NctePts_upper: int
        Number of control points in the upper surface.

    NctePts_lower: int
        Number of control points in the lower surface.

    Output: bool
        True if the control points preserve their x-ordering,
        False otherwise.
    '''

    # Split into upper and lower (including LE and TE in each surface)
    ctePts_upper = ctePts[:NctePts_upper+2,:]
    ctePts_lower = ctePts[NctePts_upper+1:,:]

    # Upper surface: x must monotonically decrease from TE to LE.
    dx_upper = np.diff(ctePts_upper[:, 0])
    if np.any(dx_upper >= 0):
        return False

    # Lower surface: x must monotonically increase from LE to TE.
    dx_lower = np.diff(ctePts_lower[:, 0])
    if np.any(dx_lower <= 0):
        return False

    return True


def _check_thickness(xy):
    '''
    Check that the airfoil has positive thickness.

    xy: array (N, 2)
        Airfoil coordinates following the
        TE -> upper -> LE -> lower -> TE convention.

    output: bool
        True if the airfoil has positive thickness everywhere,
        False otherwise.
    '''

    # Split into upper and lower xy surface (including LE and TE)
    le_idx = np.argmin(xy[:, 0])
    xy_upper = xy[:le_idx+1,:]
    xy_lower = xy[le_idx:,:]

    # Interpolate at common x-locations
    x_check = np.linspace(0.01, 0.99, 100)

    y_upper = np.interp(x_check, xy_upper[:,0], xy_upper[:,1])
    y_lower = np.interp(x_check, xy_lower[:,0], xy_lower[:,1])

    # Compute thickness
    thickness = y_upper - y_lower

    # Require positive thickness everywhere
    if np.any(thickness <= 0.0):
        return False

    return True


def _curvature(xy):
    '''
    Compute the local curvature at each xy point of the airfoil.

    The airfoil is treated as a parametric curve (x(s), y(s)) following the
    TE -> upper -> LE -> lower -> TE convention. The curve
    parameter "s" is the accumulated Euclidean distance between
    consecutive points.

    curvature = |x'y'' - y'x''| / (x'2^2 + y'^2)^(3/2)

    xy: array (N, 2)
        Airfoil coordinates following the
        TE -> upper -> LE -> lower -> TE convention.

    output: array (N,)
        Curvature at each xy point           
    '''

    # Compute acumulated euclidian distance between points
    ds = np.sqrt(np.diff(xy[:, 0])**2 + np.diff(xy[:, 1])**2)
    s = np.concatenate(([0.0], np.cumsum(ds)))

    # Compute gradients
    dxds = np.gradient(xy[:, 0], s)
    dyds = np.gradient(xy[:, 1], s)

    # Compute second derivative
    d2xds2 = np.gradient(dxds, s)
    d2yds2 = np.gradient(dyds, s)

    # Compute curvature
    return np.abs(dxds * d2yds2 - dyds * d2xds2) / (dxds**2 + dyds**2)**1.5


def is_valid_airfoil(xy, ctlPts, NctePts_upper, NctePts_lower, curvature_limit=200, R_min=0.005):
    '''
    Check a list of requirements to decide weather the geometry is valid or not.

    xy: array (N, 2)
        Airfoil coordinates following the TE -> upper -> LE -> lower -> TE convention.
        This geometry has been constructed using the ctsPts provided.

    ctlPts: array (N_ctePts, 2)
        Coordinates of all the control points used to generate xy, including LE and TE points.

    NctePts_upper: int
        Number of control points considered in the upper surface.

    NctePts_lower: int
        Number of control points considered in the lower surface.

    curvature_limit: float
        Maximum allowed local curvature.

    R_min: float, optional
        Minimum allowed leading-edge radius (avoid sharp LE).
    '''

    # Check for NaN / Inf
    if not np.all(np.isfinite(xy)):
        logger.info(f"Geometry not valid: NaNs found in the xy coordinates.")
        return False

    # Check x range
    if np.min(xy[:,0]) < 0.0 or np.max(xy[:,0]) > 1.0:
        logger.info(f"Geometry not valid: x-values exceeding [0,1] limits.")
        return False

    # Check for x-ordering preservation of the control points
    if not _ctePts_check_xordering(ctlPts, NctePts_upper, NctePts_lower):
        logger.info(f"Geometry not valid: ctePts intersecting in the x direction.")
        return False

    # Check thickness
    if not _check_thickness(xy):
        logger.info(f"Geometry not valid: Negative thickness generated.")
        return False

    # Chech curvature
    xy_curvature = _curvature(xy)
    if not np.all(np.isfinite(xy_curvature)):
        logger.info("Geometry not valid: Non-finite curvature.")
        return False

    if np.max(xy_curvature) > curvature_limit:
        logger.info("Geometry not valid: Maximum curvature exceeded.")
        return False
    
    # Check LE rounding
    le_idx = np.argmin(xy[:,0])
    radius_LE = 1.0 / (xy_curvature[le_idx])
    if (not np.isfinite(radius_LE)) or (radius_LE < R_min):
        logger.info(f"Geometry not valid: Invalid or below minimum LE radius.")
        return False

    return True
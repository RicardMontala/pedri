import numpy as np

"""
Generic utils for NACA 4-digits airfoils
"""

# http://airfoiltools.com/airfoil/naca4digit
# Closed-trailing-edge thickness coefficients
a0 = 0.2969
a1 = -0.126
a2 = -0.3516
a3 = 0.2843
a4 = -0.1036


def naca4_surface(x_coord:np.ndarray, m: int, p: int, t: int):
    '''
    For a given list of x-coordinates and NACA parameters, find the
    corresponding y-coordinates describing the NACA airfoil surface.

    Parameters
    ----------
    x_coord : np.ndarray
        Chordwise coordinates normalized by the chord, x/c, in [0, 1].
        Must be sorted in ascending order.
    m: int
        NACA 1st digit M: Maximum camber devided by 100.
    p: int
        NACA 2nd digit P: Position of maximum camber devided by 10.
    t: int
        NACA 3rd and 4th digits tt: Maximum thickness devided by 100.

    Returns
    -------
    x : ndarray
        Chordwise coordinates of the closed airfoil contour.
    y : ndarray
        Normalized vertical coordinates of the airfoil contour.
    '''

    # Obtain physical parameters from NACA digits
    mm = m/100
    pp = p/10
    tt = t/100

    # Perform checks
    if x_coord.ndim != 1:
        raise ValueError("x_coord must be a one-dimensional array.")

    if np.any((x_coord < 0.0) | (x_coord > 1.0)):
        raise ValueError("x_coord must lie within [0, 1].")

    if not 0.0 <= pp < 1.0:
        raise ValueError("p must lie within [0, 1).")

    if mm < 0.0 or tt <= 0.0:
        raise ValueError("m must be non-negative and t must be positive.")

    if np.any(np.diff(x_coord) < 0.0):
        raise ValueError("x_coord must be sorted in ascending order.")

    # Mean camber line and its gradient
    if m == 0.0:
        # Symmetric airfoil: p is irrelevant
        yc = np.zeros_like(x_coord)
        dycdx = np.zeros_like(x_coord)

    else:
        # Cambered airfoil: p must be strictly between 0 and 1
        yc = np.empty_like(x_coord)
        dycdx = np.empty_like(x_coord)

        mask = x_coord < pp
        x_front = x_coord[mask]
        x_back = x_coord[~mask]

        yc[mask] = mm / pp**2 * (2.0*pp*x_front - x_front**2)
        yc[~mask] = mm / (1.0 - pp)**2 * (1.0 - 2.0*pp + 2.0*pp*x_back - x_back**2)

        dycdx[mask] = 2.0*mm/pp**2 * (pp - x_front)
        dycdx[~mask] = 2.0*mm/(1.0 - pp)**2 * (pp - x_back)

    # Camber-line angle
    theta = np.arctan(dycdx)

    # Thickness distribution
    yt = (tt/0.2)*(a0*x_coord**0.5 + a1*x_coord + a2*x_coord**2 + a3*x_coord**3 + a4*x_coord**4)

    # Upper and lower surfaces
    x_upper = x_coord - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)

    x_lower = x_coord + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)

    # Contour: trailing edge -> leading edge -> trailing edge
    x = np.concatenate((x_upper[::-1], x_lower[1:]))
    y = np.concatenate((y_upper[::-1], y_lower[1:]))

    return np.column_stack((x, y))

import numpy as np

"""Generic utils for Bezier curves"""

def quadratic_bezier(t, ctlPts):
    '''
    Evaluate xy-coordinates of a quadratic Bezier curve at given t values.
    In Bezier quadratic curves, P0 and P2 are the end points, and P1 the control point:
    B(t) = (1-t)[(1-t)P0 + tP1] + t[(1-t)P1 + tP2]

    Parameters
    ----------
    t: array (n_points,) (between 0 and 1 each value)
        Bezier parameter
    
    ctlPts: array (3,2)
        Control points P0, P1, and P2 to construct the Bezier curve. X and y coordinates.

    Returns
    -------
    x: array (n_points,)
        x-coordinate corresponding to parameter t

    y: array (n_points,)
        y-coordinate corresponding to parameter t
    '''

    x =(1-t)*((1-t)*ctlPts[0,0]+t*ctlPts[1,0])+t*((1-t)*ctlPts[1,0]+t*ctlPts[2,0])
    y =(1-t)*((1-t)*ctlPts[0,1]+t*ctlPts[1,1])+t*((1-t)*ctlPts[1,1]+t*ctlPts[2,1])

    return x, y


def get_t(x, ctlPts):
    '''
    For given x-coordinates, what Bezier parameters t corresponds to them.
    Valid for quadratic Bezier curves defined by three (x,y) control points. 

    Parameters
    ----------
    x: array (n_points,)
        Desired x-coordinate.
    
    ctrPts: array (3, 2)
        Three control points P0, P1, P2 defining the Bezier curve:
        ((P0_x, P0_y), (P1_x, P1_y), , (P2_x, P2_y)).

    Returns
    -------
    t: array (n_points,)
        t values corresponding to the x coordinates initially provided
    '''

    # Extract the three x-coordinates of the control points
    a = ctlPts[0,0]
    b = ctlPts[1,0]
    c = ctlPts[2,0]

    # Find the roots of the resulting equation for t: (a-2b+c)*t^2 + 2(b-a)*t + (a-x)=0
    # A*t^2 + B*t + C = 0 -> t = (-B +- sqrt(B^2-4AC))/2A
    A = c + a - 2*b
    B = 2*(b - a)
    C = a - x

    if abs(A) < 1e-12:
        return -C / B

    discriminant = B**2 - 4*A*C
    discriminant = np.maximum(discriminant, 0.0)

    sqrt_d = np.sqrt(discriminant)

    t0 = (-B + sqrt_d) / (2*A)
    t1 = (-B - sqrt_d) / (2*A)

    valid0 = (t0 >= -1e-10) & (t0 <= 1 + 1e-10) # Should be between 0 and 1
    
    return np.where(valid0, t0, t1) # Return t0 if it fall within [0,1], otherwise return t1


def evaluate_surface(x_coord, ctl_pts):
    '''
    For a given list of x-coordinates and control points defining a curve
    constructed concatenaiting bezier quadratic curve find the y value.
    '''

    y_coord = np.full_like(x_coord, np.nan)

    segments = []

    for j in range(len(ctl_pts) - 2):
        if j == 0:
            p0 = ctl_pts[0]
            p1 = ctl_pts[1]
            p2 = 0.5 * (ctl_pts[1] + ctl_pts[2])

        elif j == len(ctl_pts) - 3:
            p0 = 0.5 * (ctl_pts[-3] + ctl_pts[-2])
            p1 = ctl_pts[-2]
            p2 = ctl_pts[-1]

        else:
            p0 = 0.5 * (ctl_pts[j] + ctl_pts[j + 1])
            p1 = ctl_pts[j + 1]
            p2 = 0.5 * (ctl_pts[j + 1] + ctl_pts[j + 2])

        segments.append(np.array([p0, p1, p2]))

    for segment in segments:
        xmin = min(segment[0, 0], segment[2, 0])
        xmax = max(segment[0, 0], segment[2, 0])

        mask = ((x_coord >= xmin - 1e-10) & (x_coord <= xmax + 1e-10))

        if not np.any(mask):
            continue

        t = get_t(x_coord[mask], segment)
        _, y = quadratic_bezier(t, segment)

        # Preserve the original behavior at segment boundaries.
        y_coord[mask] = y

    if not np.all(np.isfinite(y_coord)):
        raise ValueError("Bezier segments did not cover all requested x-coordinates.")

    return y_coord

import math
import mpmath
import numpy as np


def normalize(v):
    """Return normalized numpy vector"""
    return(v / np.sqrt(np.sum(v**2)))


def magnitude(v):
    """Return magnitude for numpy vector"""
    return(np.sqrt(v ** 2).sum())


def calc_theta(v1, v2):
    """ Calculate theta of a function given v1 and v2.
        equation for this is theta = acos((v * vp) / (norm(v) * norm(vp)))

    Arguments:
        v1(np.array): The first vector in the form []
        v2(np.array): The second vector in the form

    Returns:
        theta:
            the theta between v1 and v2
    """
    theta = math.acos(np.dot(v1, v2) / (magnitude(v1) * magnitude(v2)))

    return(theta)


def dirichlet(pair):
    """ Calculate dirichlet between 2D and 3D triangle pair.


    Arguments:
        pair: This is a numpy array with pairs of 3d and 2d triangles

    Returns:
        dirichlet: returns the dirichlet energy between pair of tris
    """

    # Assign values in triangle to vertices
    fv1 = np.array(pair["2D"][0])
    fv2 = np.array(pair["2D"][1])
    fv3 = np.array(pair["2D"][2])
    v1 = pair["3D"][0]
    v2 = pair["3D"][1]
    v3 = pair["3D"][2]

    # Calc theta1, theta2, theta3
    theta1 = calc_theta(v3 - v1, v3 - v2)
    theta2 = calc_theta(v2 - v1, v2 - v3)
    theta3 = calc_theta(v1 - v3, v1 - v2)

    # Calc normalized euclidian vectors from points
    dpart1 = magnitude(fv1 - fv2) ** 2
    dpart2 = magnitude(fv1 - fv3) ** 2
    dpart3 = magnitude(fv2 - fv3) ** 2

    # Calc dirichlet
    dirichlet = mpmath.cot(theta3) * dpart1 + \
                mpmath.cot(theta2) * dpart2 + \
                mpmath.cot(theta1) * dpart3

    return(dirichlet)


def linearly_interpolate(a, b, M, m, v):
    """Interpolate between m and M with a range of [a, b] and input v"""
    return((b - a) * ((v - m)/(M - m)) + a)


def solve_quadratic(A, B, C):
    """Solve quadratic equation given A, B, and C"""
    if (A == 0):
        return([])

    discriminant = ((B ** 2) - 4*A*C)
    if discriminant > 0:
        s1 = (-B + math.sqrt(discriminant)) / (2*A)
        s2 = (-B - math.sqrt(discriminant)) / (2*A)
        return([s1, s2])

    elif (discriminant == 0):
        return((-B + math.sqrt(discriminant)) / (2*A))

    else:
        return([])


if __name__ == "__main__":
    pair = {
            "2D": ((0, 1), (0, 0), (1, 0)),
            "3D": (np.array([0, 1, 1]), np.array([0, 0, 0]),
                   np.array([1, 1, 0]))
            }
    print(dirichlet(pair))

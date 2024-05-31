import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

def itp(x, px, py, axis=-1):
    return CubicSpline(px, py, axis=axis)(x)

def itpmono(x, px, py, axis=-1, extrapolate=False):
    return PchipInterpolator(px, py, axis=axis, extrapolate=extrapolate)(x)

def cubic_poly_one_real_root(a, b, c, d):
    # ax^3 + bx^2 + cx + d = 0
    p = (3*a*c - b**2) / (3*(a**2))
    q = (27*(a**2)*d - 9*a*b*c + 2*(b**3)) / (27*(a**3))
    tmp = np.sqrt( (q/2)**2 + (p/3)**3 )
    return -b/(3*a) + np.cbrt(-q/2 + tmp) + np.cbrt(-q/2 - tmp)
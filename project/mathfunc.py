import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

def itp(x, px, py, axis=-1):
    return CubicSpline(px, py, axis=axis)(x)

def itpmono(x, px, py, axis=-1, extrapolate=False):
    return PchipInterpolator(px, py, axis=axis, extrapolate=extrapolate)(x)
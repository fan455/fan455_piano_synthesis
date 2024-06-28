import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from math import sin, cos, tan, sqrt

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

def area_of_triangle(x1, y1, x2, y2, x3, y3):
    return 0.5*((x1*y2+x2*y3+x3*y1)-(y1*x2+y2*x3+y3*x1))

def cross_product_2d(x1, y1, x2, y2):
    return x1*y2 - x2*y1

def dot_product_2d(x1, y1, x2, y2):
    return x1*x2 + y1*y2

def is_in_triangle(x, y, x1, y1, x2, y2, x3, y3):
    if cross_product_2d(x2-x1, y2-y1, x3-x1, y3-y1) < 0.: # clockwise
        x2, x3 = x3, x2
        y2, y3 = y3, y2 # ensure counterclock
    if cross_product_2d(x2-x1, y2-y1, x-x1, y-y1) < 0.:
        return False
    elif cross_product_2d(x3-x2, y3-y2, x-x2, y-y2) < 0.:
        return False
    elif cross_product_2d(x1-x3, y1-y3, x-x3, y-y3) < 0.:
        return False
    else:
        return True

def ensure_three_points_counterclock(x1, y1, x2, y2, x3, y3):
    if cross_product_2d(x2-x1, y2-y1, x3-x1, y3-y1) < 0.:
        return x1, y1, x3, y3, x2, y2
    else:
        return x1, y1, x2, y2, x3, y3

def line_eq_two_points(x1, y1, x2, y2):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k*x1
    return k, b # slope, constant

def intersect_of_two_lines(k1, b1, k2, b2):
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return x, y

def line_eq_point_slope(x0, y0, k):
    return y0 - k*x0 # b

def is_perfect_square(n):
    a = int(sqrt(n))
    return a * a == n, a

def point_between_two(x1, y1, x2, y2, r=0.5):
    # r is the ratio
    return (1-r)*x1+r*x2, (1-r)*y1+r*y2

def central_point_of_triangle(x1, y1, x2, y2, x3, y3):
    # r is the ratio
    return (x1+x2+x3)/3, (y1+y2+y3)/3

def poly2d_fit(x: np.ndarray, y: np.ndarray, f: np.ndarray):
    # x, y: (n,), f: (n, k)
    n = x.size
    n_is_perfect_square, nx = is_perfect_square(n)
    assert n_is_perfect_square
    A = np.zeros((n, n))
    i = 0
    for px in range(0, nx):
        for py in range(0, nx):
            A[:, i] = x**px * y**py
            i += 1
    return np.linalg.solve(A, f) # (n, k)

def get_poly2d_order(order):
    nx = int(order+1)
    sx = np.zeros(nx**2, dtype=int)
    sy = np.zeros(nx**2, dtype=int)
    i = 0
    for px in range(0, nx):
        for py in range(0, nx):
            sx[i] = px
            sy[i] = py
            i += 1
    return sx, sy

def shift_coord_sys(x, y, Ox_new, Oy_new):
    return x-Ox_new, y-Oy_new

def rotate_coord_sys(x, y, theta, clockwise=True):
    if clockwise:
        return x*cos(theta)-y*sin(theta), x*sin(theta)+y*cos(theta)
    else: # counterclock
        return x*cos(theta)+y*sin(theta), -x*sin(theta)+y*cos(theta)

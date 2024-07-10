#https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
import numpy as np
from plot import plot, plot_scatter
from scipy.interpolate import PchipInterpolator

eigval_path = "D:/self_research/piano_project/data/soundboard_eigenvalues.npy"
damp_path = "D:/self_research/piano_project/data/soundboard_damping.npy"


eigval = np.load(eigval_path)
assert eigval.ndim == 1
eig_n = eigval.size
print(f'eig_n = {eig_n}')

freq_init = np.sqrt(eigval)/(2*np.pi)

points = np.array([
    [0, 0], [100, 50], [1000, 100], [5000, 1000], [15000, 5000]
], dtype=np.float64)
#points[:, 1] = 5

freq_test = np.arange(0, 10000, dtype=np.float64)
damp_test = PchipInterpolator(points[:,0], points[:,1], axis=0, extrapolate=True)(freq_test)

damp = PchipInterpolator(points[:,0], points[:,1], axis=0, extrapolate=True)(freq_init)
np.save(damp_path, damp)

#plot(freq_init)
plot(damp_test, freq_test, title='frequency and damping relationship',
     xlabel='freq', ylabel='damp')

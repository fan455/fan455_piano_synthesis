# Audio Time-Frequency Analysis

import numpy as np
from scipy import fft, signal

# real discrete Fourier transform
def rfft_z(y, axis=-1):
    return fft.rfft(y, axis=axis, norm='backward')

def rfft_zf(y, sr, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return z, f

def rfft_m(y, axis=-1):
    # This returns frequency magnitudes (real positive numbers) instead of complex numbers.
    return np.abs(fft.rfft(y, axis=axis, norm='backward'))

def rfft_mf(y, sr, axis=-1):
    # This returns frequency magnitudes and the frequencies consistent with sr.
    m = np.abs(fft.rfft(y, axis=axis, norm='backward'))
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return m, f

def rfft_mp(y, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    return m, p

def rfft_mpf(y, sr, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return m, p, f

def irfft_z(z, time_size_is_even=True, axis=-1):
    if time_size_is_even:
        return fft.irfft(z, axis=axis, norm='backward')
    else:
        return fft.irfft(z, n=z.shape[axis]*2-1, axis=axis, norm='backward')

def irfft_mp(m, p, time_size_is_even=True, axis=-1):
    z = np.empty(m.shape, dtype=np.complex128)
    z.real, z.imag = m*np.cos(p), m*np.sin(p)
    if time_size_is_even:
        return fft.irfft(z, axis=axis, norm='backward')
    else:
        return fft.irfft(z, n=z.shape[axis]*2-1, axis=axis, norm='backward')

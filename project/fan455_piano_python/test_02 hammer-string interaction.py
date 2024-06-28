import numpy as np
from plot import plot, plot_itp
from loudness import norm_peak_mono
from tfa import rfft_mf
from mathfunc import itpmono
from scipy.signal import fftconvolve
import soundfile as sf


sr = 44100
du = 2.0
ns = int(sr*du)
t = np.arange(ns)/sr

folder_2 = 'D:/self_research/piano_project/audio_out'

# String harmonic vibration
x = np.zeros(ns)
freq0 = 880
n_freq = 6
i_freq = np.arange(1,n_freq+1)
freq = i_freq * freq0

damp0 = 7.5
damp = damp0 * i_freq

phase = np.zeros(n_freq)
phase[0::2] += np.pi

#ampli = np.ones(n_freq)
ampli = 1 / i_freq

x = np.sum(np.expand_dims(ampli,1) * np.exp(-np.expand_dims(damp,1)*np.expand_dims(t,0)) * \
    np.sin(2*np.pi*np.expand_dims(freq,1)*np.expand_dims(t,0) + np.expand_dims(phase,1)), axis=0)


# Hammer excitation force
do_hammer_excitation = True

if do_hammer_excitation:
    du_force=0.0007
    ns_force = int(sr*du_force)
    n_force = np.arange(ns_force)
    
    points = np.array([\
        [0,0], [int(ns_force*0.1),0.5], [int(ns_force*0.3),1], [int(ns_force*0.5),0.7], [int(ns_force*0.75),0.25], [ns_force*0.95,0]\
    ])
    force = itpmono(n_force, points[:,0], points[:,1], extrapolate=False)
    np.nan_to_num(force, copy=False, nan=0.0)

    points[:,0] /= sr
    force_m, force_f = rfft_mf(force, sr)
    #plot_itp(force, t[:ns_force], points, title='force')
    #plot(force_m, force_f, title='force magnitude')

    x = fftconvolve(force, x, mode='full')[:ns]

    out_p = f'{folder_2}/test_02 hammer force.wav'
    sf.write(out_p, force, sr, subtype='PCM_16')
    out_p = f'{folder_2}/test_02 with hammer.wav'

else:
    out_p = f'{folder_2}/test_02 without hammer.wav'

x = norm_peak_mono(x, -3)
x_m, x_f = rfft_mf(x, sr)

#print(sf.available_subtypes())
sf.write(out_p, x, sr, subtype='PCM_16')

#plot(x, t, title='x')
#plot(x_m, x_f, title='x magnitude')



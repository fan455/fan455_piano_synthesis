import numpy as np
from plot import plot
import soundfile as sf
from loudness import norm_peak_mono
from scipy.signal import fftconvolve
from tfa import rfft_mf

folder_1 = 'D:/self_research/piano_project/data/soundboard_response'
folder_2 = 'D:/self_research/piano_project/audio_out'
in_p1 = f'{folder_1}/soundboard_response_01_transverse.npy'
in_p11 = f'{folder_1}/soundboard_response_01_shear_x.npy'
in_p12 = f'{folder_1}/soundboard_response_01_shear_y.npy'
in_p2 = f'{folder_2}/test_02 without hammer.wav'
in_p3 = f'{folder_2}/test_02 hammer force.wav'
out_p1 = f'{folder_2}/soundboard_response_01_transverse.wav'
out_p11 = f'{folder_2}/soundboard_response_01_shear_x.wav'
out_p12 = f'{folder_2}/soundboard_response_01_shear_y.wav'
out_p2 = f'{folder_2}/piano_01.wav'
out_p3 = f'{folder_2}/piano_01_no_harmonic.wav'

sr = 44100
y1 = np.load(in_p1)
y1 = norm_peak_mono(y1, -3)

y11 = np.load(in_p11)
y11 = norm_peak_mono(y11, -3)

y12 = np.load(in_p12)
y12 = norm_peak_mono(y12, -3)

y2, _ = sf.read(in_p2)
y3, _ = sf.read(in_p3)

y = fftconvolve(y1, y2) #+ fftconvolve(y11, y2) + fftconvolve(y12, y2)
y = norm_peak_mono(y, -3)

y_ = fftconvolve(y11, y3) #+ fftconvolve(y11, y3) + fftconvolve(y12, y3)
y_ = norm_peak_mono(y_, -3)

assert sr == 44100
sf.write(out_p1, y1, 44100, 'PCM_16')
sf.write(out_p11, y11, 44100, 'PCM_16')
sf.write(out_p12, y12, 44100, 'PCM_16')
sf.write(out_p2, y, 44100, 'PCM_16')
sf.write(out_p3, y_, 44100, 'PCM_16')
#plot(y)

m, f = rfft_mf(y1, sr)
plot(m, f, title='rfft of soundboard response transverse')

m, f = rfft_mf(y11, sr)
plot(m, f, title='rfft of soundboard response shear x')

m, f = rfft_mf(y12, sr)
plot(m, f, title='rfft of soundboard response shear y')

m, f = rfft_mf(y_, sr)
plot(m, f, title='rfft of soundboard response with hammer')

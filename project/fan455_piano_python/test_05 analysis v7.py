import numpy as np
from plot import plot
import soundfile as sf
from loudness import norm_peak_mono
from scipy.signal import fftconvolve
from tfa import rfft_mf

folder_1 = 'D:/self_research/piano_project/data/soundboard_response'
folder_2 = 'D:/self_research/piano_project/audio_out'
in_p1 = f'{folder_1}/soundboard_response_01.npy'
in_p2 = f'{folder_2}/test_02 with hammer.wav'
#in_p3 = f'{folder_2}/test_02 hammer force.wav'
out_p1 = f'{folder_2}/soundboard_response_01_sum.wav'
out_p10 = f'{folder_2}/soundboard_response_01_disp_z.wav'
out_p11 = f'{folder_2}/soundboard_response_01_rot_x.wav'
out_p12 = f'{folder_2}/soundboard_response_01_rot_y.wav'
out_p13 = f'{folder_2}/soundboard_response_01_disp_x.wav'
out_p14 = f'{folder_2}/soundboard_response_01_disp_y.wav'
out_p2 = f'{folder_2}/piano_01.wav'
#out_p3 = f'{folder_2}/piano_01_no_harmonic.wav'

sr = 44100
y1 = np.load(in_p1)
if np.isfortran(y1):
    y1 = y1.T
y10 = norm_peak_mono(y1[0,:], -3)
y11 = norm_peak_mono(y1[1,:], -3)
y12 = norm_peak_mono(y1[2,:], -3)
y13 = norm_peak_mono(y1[3,:], -3)
y14 = norm_peak_mono(y1[4,:], -3)
y1_sum = norm_peak_mono(np.sum(y1, axis=0), -3)

y2, _ = sf.read(in_p2)
#y3, _ = sf.read(in_p3)

y = fftconvolve(y1_sum, y2)
y = norm_peak_mono(y, -3)

#y_ = fftconvolve(y11, y3) #+ fftconvolve(y11, y3) + fftconvolve(y12, y3)
#y_ = norm_peak_mono(y_, -3)

assert sr == 44100
sf.write(out_p1, y1_sum, sr, 'PCM_16')
sf.write(out_p10, y10, sr, 'PCM_16')
sf.write(out_p11, y11, sr, 'PCM_16')
sf.write(out_p12, y12, sr, 'PCM_16')
sf.write(out_p13, y13, sr, 'PCM_16')
sf.write(out_p14, y14, sr, 'PCM_16')
sf.write(out_p2, y, sr, 'PCM_16')
#sf.write(out_p3, y_, 44100, 'PCM_16')
#plot(y)

m, f = rfft_mf(y10, sr)
plot(m, f, title='rfft of soundboard response disp z')

m, f = rfft_mf(y, sr)
plot(m, f, title='rfft of piano sound')

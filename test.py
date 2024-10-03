import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile

sample_rate, audio_time_series = wavfile.read("Scene 1_Jorge.wav")
single_sample_data = audio_time_series[:sample_rate]

def fft_plot(audio, sample_rate):
  N = len(audio)    # Number of samples
  T = 1/sample_rate # Period
  y_freq = fft(audio)
  domain = len(y_freq) // 2
  x_freq = np.linspace(0, sample_rate//2, N//2)
  plt.plot(x_freq, abs(y_freq[:domain]))
  plt.xlabel("Frequency [Hz]")
  plt.ylabel("Frequency Amplitude |X(t)|")
  return plt.show()

fft_plot(single_sample_data, sample_rate)
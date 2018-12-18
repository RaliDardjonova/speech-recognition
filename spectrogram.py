#!/usr/bin/env python
import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

def get_windows(stream, window_size):
    X = []
    while True:
        x = read(stream, window_size)
        if len(x) != window_size:
            break
        X.extend(x)
        yield np.array(X)
        X = X[window_size:]

def read(stream, window_size):
    num_channels = stream.getnchannels()
    sample_width = stream.getsampwidth()

    x = stream.readframes(window_size)
    x = np.fromstring(x, dtype=np.int8 if sample_width == 1 else np.int16)

    x = np.reshape(x, (-1, num_channels))

    if num_channels > 1:
        x = (x[:,0] + x[:,1]) / 2
    else:
        x = x[:,0]

    return x

#w = wave.open('its-not-that-easy.wav', 'r')
w = wave.open('you-call-that-fun.wav', 'r')
window_size = 1024
sample_width = w.getsampwidth()
sample_rate  = w.getframerate()
hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(window_size)) / window_size)

Y = []
for x in get_windows(w, window_size):
    y = np.fft.rfft(x*hann)
    y = y[:window_size//2]
    Y.append(y)

Y = np.column_stack(Y)
Y = np.absolute(Y) * 2.0 / np.sum(hann)
Y = Y / np.power(2.0, (8 * sample_width - 1))
Y = (20.0 * np.log10(Y)).clip(-120)

print Y

t = np.arange(0, Y.shape[1], dtype=np.float) * window_size / sample_rate
f = np.arange(0, window_size / 2, dtype=np.float) * sample_rate / window_size
ax = plt.subplot(111)
plt.pcolormesh(t, f, Y, vmin=-120, vmax=0)

#plt.yscale('symlog', linthreshy=100, linscaley=0.25)
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.xlim(0, t[-1])
plt.ylim(0, f[-1])

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

cbar = plt.colorbar()
cbar.set_label("Intensity (dB)")

plt.show()

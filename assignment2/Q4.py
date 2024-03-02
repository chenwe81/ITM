import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Load the wave files
signals = []
for i in range(1, 6):
    rate, data = wavfile.read(f'/mnt/home/chenwe81/ITM/assignment2/mix_{i}.wav')
    signals.append(data)

# Stack the arrays vertically
X = np.vstack(signals)

# Unmix the signals
ica = FastICA(n_components=5)
S_ = ica.fit_transform(X.T)  # Reconstruct signals

# Rescale the unmixed signals
S_ = np.interp(S_, (S_.min(), S_.max()), (-1, 1))

# Write out the unmixed signals
for i, s in enumerate(S_.T):
    wavfile.write(f'unmixed_{i+1}.wav', rate, s.astype(np.float32))

# Plot the time courses of the different unmixed songs
time = np.arange(S_.shape[0]) / rate
for i in range(S_.shape[1]):
    plt.figure(figsize=(10, 2))
    plt.plot(time, S_[:, i])
    plt.xlabel('Time [s]')
    plt.title(f'Unmixed Signal {i+1}')
    plt.show()
import numpy, scipy, matplotlib.pyplot as plt
import librosa, librosa.display

x, sr = librosa.load('audio/drum_sound.wav')

plt.figure(figsize=(5, 5))
librosa.display.waveplot(x, sr, alpha=0.8)

# plt.suptitle('Drum (Time Domain)')
# plt.ylabel('Amplitude')
# plt.xlabel('Time (Seconds)')
# plt.ylim(-0.8, 0.8)

X = scipy.fft(x[:4096])
X_mag = numpy.absolute(X)        # spectral magnitude
f = numpy.linspace(0, sr, 4096)  # frequency variable
plt.figure(figsize=(5, 5))
plt.plot(f[:2000], X_mag[:2000]) # magnitude spectrum
plt.suptitle('Drum (Frequency Domain)')
plt.ylabel('Magnitude / Power')
plt.xlabel('Frequency (Hz)')
plt.ylim(10**-4, 10**3)
plt.yscale('log')


plt.show()
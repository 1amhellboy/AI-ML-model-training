import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_path = 'recordings_0_george_0.wav'
y, sr = librosa.load(audio_path, sr=None)

y = y / np.max(np.abs(y))

plt.figure(figsize=(12, 4))
plt.plot(np.linspace(0, len(y)/sr, num=len(y)), y)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Normalized Audio Waveform')
plt.show()

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print("MFCC feature shape:", mfccs.shape)
print("MFCCs (first 2 frames):\n", mfccs[:, :2])
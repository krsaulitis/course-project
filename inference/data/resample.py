import librosa
import scipy
import numpy as np

audio, sr = librosa.load('./output_0.wav', sr=16000)

waveform = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)

scipy.io.wavfile.write('./output_0_re.wav', rate=16000, data=waveform)

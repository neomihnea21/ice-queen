import numpy as np
import soundfile as sf

data1, _ = sf.read('assets/iris-noised.wav')
data2, _ = sf.read("assets/iris_denoised_ksvd.wav")

l1 = np.shape(data2)[0]
print(l1)
print(np.mean(data1[:l1] - data2))
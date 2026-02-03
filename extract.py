import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
import omp
import dictionary_builder as dic 
data, sr = sf.read("assets/semisonic.wav", dtype=np.float64)

print(np.shape(data))
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
from dictionary_builder import create_dict
from omp import find_sparse_vector
data, sr = sf.read("assets/semisonic.wav", dtype=np.float64)

data=data.T
CHUNK_SIZE=4410
new_signal=np.zeros(shape=np.shape(data)) 
#we will have a signal of size N, which we want to define in terms of a N * 2N dictionary
# But we won't compute that, we don't need it

learned_dict = create_dict(CHUNK_SIZE)
max_components = 10
for i in range(len(data)):
    chunks = np.reshape(data[i], (-1, CHUNK_SIZE))
    base_index=0
    for chunk in chunks:
        reformed_data = find_sparse_vector(learned_dict.T, chunk)
        new_signal[i][base_index : base_index+CHUNK_SIZE] = reformed_data
        base_index += CHUNK_SIZE
        #now we need to find out which components impact us the least and remove those
denoised_file = write("assets/semisonic_denoised.wav", 44100, new_signal)

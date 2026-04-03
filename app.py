import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
import omp, sys 
import dictionary_builder as dic 

def fix_broken_file():
    N = 350 # this is just some constant, picked so the dictionary can learn enough detail
    path = sys.argv[1]
    save_as = sys.argv[2]
    data, sr = sf.read(path)
    data = data.T
    new_data = np.zeros_like(data).astype(np.float32)
    song_size = np.shape(data)[1]

    learned_dict = dic.mixed(N, 'db4') # testing has showed the db4 wavelet matrix is the best at modeling songs, alongside sines 
    dict_norms = np.linalg.norm(learned_dict, axis=0)
    learned_dict /= dict_norms
    
    data = np.clip(data, -1, 1)
    for line in range(2):
        for j in range(0, song_size - N , N):
            section = data [line, j:j+N]
            new_section, _  = omp.fit_fast(learned_dict.T, section, 30)
            new_data [line, j:j+N] = new_section
    
    new_data = np.clip(new_data, -1, 1)
    new_data = new_data.T

    sf.write(save_as, new_data, sr)

if __name__ == "__main__":
    fix_broken_file()
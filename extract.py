import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
import omp, time, copy, fista, noises, sys 
import dictionary_builder as dic 
from test_noises import low_pass_filter


def find_snr(original, processed): 
   p1 = np.mean(original**2)
   p2 = np.mean((processed - original) ** 2)
   snr_db = 10 * np.log10(p1 / p2)
   return snr_db


paths = ["assets/pop-1.wav", "assets/iris.wav", "assets/cecil.wav", "assets/figaro.wav"]
N_small = 256
N_large = 1024
LCM = 1024

for path in paths: 
    data, sr = sf.read(path)
    orig_data = copy.deepcopy(data)
    data = data.T

    
    new_data_1 = np.zeros_like(data).astype(np.float32)
    new_data_2 = np.zeros_like(data).astype(np.float32)
    song_size = np.shape(data)[1]

    pink = noises.pink_noise(song_size, sr)
    data[0] += pink
    data[1] += pink
    data = np.clip(data, -1, 1)
    
    low = low_pass_filter(data, 1000, sr)
    high = data - low 

    # pad the data, so it's split as even chunks of length 256
    extra_cols = (LCM - (song_size % LCM)) % LCM
    data = np.pad(data, ((0, 0), (0, extra_cols)))

    sf.write(f"assets/{path[7:-4]}-noised.wav", data.T, sr)

    # we can assume this was normalized
    small_dict = np.load("dicts/trained-v3-SMALL.npy")
    large_dict = np.load("dicts/trained-v3-LARGE.npy")

    #dict_eigen = fista.precompute_step(learned_dict)

    t1 = time.time()
    for line in range(2):
        for j in range(0, song_size - N_small , N_small):
            section = data [line, j:j+N_small]
            new_section, _  = omp.fit_fast(small_dict.T, section, 35)
            new_data_1 [line, j:j+N_small] = new_section
    t2=time.time()
    
    omp_duration = t2-t1
    t1 = time.time()
    for line in range(2):
        for j in range(0, song_size - N_large , N_large):
            section = data [line, j:j+N_large]
            new_section, _  = omp.fit_fast(large_dict.T, section, 35)
            new_data_2 [line, j:j+N_large] = new_section
    t2=time.time()

    omp_duration += t2-t1
    new_data = new_data_1 + new_data_2
    print(f"Eroarea cu OMP: {np.mean(new_data-orig_data.T)**2}, filtrarea piesei {path[7:-4]} a durat {omp_duration} secunde")
    new_data = np.clip(new_data, -1, 1)
    new_data = new_data.T
    # compute SNR
    #snr_1 = find_snr(orig_data.T, data)
    #snr_2 = find_snr(orig_data, new_data)
    #print(f"SNR scazut cu: {snr_2 - snr_1:.2f} dB")
    sf.write(f"assets/{path[7:-4]}_denoised_ksvd.wav", new_data, sr)
    
    #t1 = time.time()
    #for line in range(2):
    #    for j in range(0, song_size - N , N):
    #        section = data [line, j:j+N]
    #        new_section  = fista.find_best_x(learned_dict, section, dict_eigen, 10, 7)
    #        new_data_fista [line, j:j+N] = learned_dict @ new_section
    #t2=time.time()
    #print(f"Eroarea cu FISTA: {np.mean(new_data_fista-orig_data.T)**2}, filtrarea piesei {path[7:-4]} a durat {t2-t1} secunde")
    #new_data_fista = np.clip(new_data_fista, -1, 1)
    #new_data_fista = new_data_fista.T
    #snr_1 = find_snr(orig_data.T, data)
    #snr_2 = find_snr(orig_data, new_data_fista)
    #print(f"SNR scazut cu: {snr_2 - snr_1:.2f} dB")
    #sf.write(f"assets/{path[7:-4]}_denoised_2.wav", new_data_fista, sr)

    


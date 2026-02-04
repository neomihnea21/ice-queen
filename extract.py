import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
import omp, time, copy, fista
import dictionary_builder as dic 

def pink_noise(duration, sr):
    n = int(duration)
    white = np.random.normal(0, 1, n) # start with white noise
    fft = np.fft.rfft(white) 

    freqs = np.fft.rfftfreq(n, 1 / sr) # and run a 1/f low-pass filter, this leads to 1/f spectral density
    freqs[0] = freqs[1]
    fft /= np.sqrt(freqs)

    pink = np.fft.irfft(fft, n)
    pink /= np.max(np.abs(pink))  # .wav works only with numbers from -1 to 1, so this is needed

    return pink

def find_snr(original, processed): 
   p1 = np.mean(original**2)
   p2 = np.mean((processed - original) ** 2)
   snr_db = 10 * np.log10(p1 / p2)
   return snr_db

paths = ["assets/semisonic.wav", "assets/iris.wav", "assets/cecil.wav", "assets/figaro.wav"]
N = 350
for path in paths: 
    data, sr = sf.read(path)
    orig_data = copy.deepcopy(data)
    data = data.T
    new_data = np.zeros_like(data).astype(np.float32)
    new_data_fista = np.zeros_like(data).astype(np.float32)
    song_size = np.shape(data)[1]

    pink = pink_noise(song_size, sr)
    data[0] += pink
    data[1] += pink
    data = np.clip(data, -1, 1)
    print(np.shape(data))
    sf.write(f"assets/{path[7:-4]}-noised.wav", data.T, sr)


    learned_dict = dic.mixed(N, 'db4')
    dict_norms = np.linalg.norm(learned_dict, axis=0)
    learned_dict /= dict_norms

    dict_eigen = fista.precompute_step(learned_dict)

    t1 = time.time()
    for line in range(2):
        for j in range(0, song_size - N , N):
            section = data [line, j:j+N]
            new_section  = omp.fit_fast(learned_dict.T, section, 30)
            new_data [line, j:j+N] = new_section
    t2=time.time()
    print(f"Eroarea cu OMP: {np.mean(new_data-orig_data.T)**2}, filtrarea piesei {path[7:-4]} a durat {t2-t1} secunde")
    new_data = np.clip(new_data, -1, 1)
    new_data = new_data.T
    # compute SNR
    snr_1 = find_snr(orig_data.T, data)
    snr_2 = find_snr(orig_data, new_data)
    print(f"SNR scazut cu: {snr_2 - snr_1:.2f} dB")
    sf.write(f"assets/{path[7:-4]}_denoised_1.wav", new_data, sr)
    
    t1 = time.time()
    for line in range(2):
        for j in range(0, song_size - N , N):
            section = data [line, j:j+N]
            new_section  = fista.find_best_x(learned_dict, section, dict_eigen, 10, 7)
            new_data_fista [line, j:j+N] = learned_dict @ new_section
    t2=time.time()
    print(f"Eroarea cu FISTA: {np.mean(new_data_fista-orig_data.T)**2}, filtrarea piesei {path[7:-4]} a durat {t2-t1} secunde")
    new_data_fista = np.clip(new_data_fista, -1, 1)
    new_data_fista = new_data_fista.T
    snr_1 = find_snr(orig_data.T, data)
    snr_2 = find_snr(orig_data, new_data_fista)
    print(f"SNR scazut cu: {snr_2 - snr_1:.2f} dB")
    sf.write(f"assets/{path[7:-4]}_denoised_2.wav", new_data_fista, sr)

    


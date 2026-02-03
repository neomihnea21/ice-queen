import numpy as np
import pywt
def create_random_dict(signal_length):
    dict  = np.random.uniform(-1, 1, (signal_length, 2*signal_length))
    np.save("dicts/random.npy", dict)
    return dict

def cosine(signal_length, atom_count):
    dict = np.zeros((signal_length, atom_count))
    for i in range (np.shape(dict)[0]):
        for j in range(np.shape(dict)[1]):
            dict[i, j]  = np.cos(np.pi * i * j / signal_length)
    return dict

def wavelet_matrix(signal_length, wavelet_name):
    wavelet = pywt.Wavelet(wavelet_name)
    wave_matrix = np.zeros((signal_length, signal_length))
    level = pywt.dwt_max_level(signal_length, wavelet.dec_len)
    # the trick to getting a wavelet matrix: run an identity operator over a full-scale DWT 
    test_signals = np.eye(signal_length)
    for i in range(signal_length):
        mapping = pywt.wavedec(test_signals[i], wavelet, mode = 'per', level = level)
        # mapping might be a tad longer 
        wave_matrix[:, i] = np.concatenate(mapping)
    return wave_matrix

def mixed(signal_length, wavelet_name):
    d1 = cosine(signal_length, signal_length)
    d2 = wavelet_matrix(signal_length, wavelet_name)
    return np.concatenate((d1, d2), axis=1)
# DB4 wavelets
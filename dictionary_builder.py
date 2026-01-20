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

def morlet(N, sigma=1.0, kx=5.0, ky=0.0):
    t = np.linspace(-2, 2, N)
    x, y = np.meshgrid(t, t)
    envelope = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    carrier = np.exp(1j * (kx * x + ky * y))
    morlet = envelope * carrier
    
    return morlet

def mixed(signal_length):
    d1 = cosine(signal_length, signal_length)
    d2 = morlet(signal_length)
    print(np.shape(d2))
    return np.dstack((d1, d2))
# DB4 wavelets
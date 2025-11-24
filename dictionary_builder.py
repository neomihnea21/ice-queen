import numpy as np

def create_dict(signal_length):
    dict  = np.random.uniform(-1, 1, (signal_length, 2*signal_length))
    np.save("dicts/random.npy", dict)
    return dict
CHUNK = 4410  #one second is 44100
create_dict(CHUNK)
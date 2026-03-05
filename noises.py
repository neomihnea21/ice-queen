import numpy as np

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

def rumble(duration):
    rumble = np.random.normal(0, 1, duration)
    # low
    THRESHOLD = 0.002
    filter = np.array([THRESHOLD] * 1/THRESHOLD)
    rumble = np.convolve(rumble, filter, mode='same')

    return rumble

def crack(duration, sr):
    ans = np.zeros(duration)
    CRACKS_PER_SECOND = 25
    # say the vinyl cracks 25 times a second
    num_clicks = int(duration * CRACKS_PER_SECOND / sr)
    
    click_positions = np.random.randint(0, duration, num_clicks)
    for pos in click_positions:
        length = np.random.randint(20, 200)
        weight = np.random.uniform(0.3, 1)
        ans[pos:pos+length] += weight * np.exp(-np.linspace(0, 5, length))
    return ans

def vinyl(duration, sr):
    crackle = crack(duration, sr) * 0.1 
    rumbling = rumble(duration) * 0.05
    hiss = pink_noise(duration, sr) * 0.02
    
    vinyl = crackle + rumbling + hiss 
    return vinyl
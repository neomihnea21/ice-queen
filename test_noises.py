import numpy as np 
import soundfile as sf 
import dictionary_builder as dic
import ksvd, time
from scipy.signal import filtfilt, butter 

# here's the genius idea: do TWO DICTIONARIES AT ONCE, 
def low_pass_filter(signal, cutoff, sr):
   nyquist = 0.5 * sr 
   cutoff /= nyquist 

   high, low = butter(5, cutoff, btype='low', analog=False)

   filtered_signal = filtfilt(high, low, signal)
   return filtered_signal

def main():
   SMALL_TO_LARGE_CUTOFF = 1000
   N_small = 256
   N_large = 1024
   LCM = 1024

   D_small = dic.mixed(N_small, 'db8')
   dict_norms = np.linalg.norm(D_small, axis=0)
   D_small /= dict_norms

   D_large = dic.mixed(N_large, 'db8')
   dict_norms = np.linalg.norm(D_large, axis=0)
   D_large /= dict_norms

   paths = ["assets/arabic.wav", "assets/pop-2.wav", "assets/smooth-operator.wav"]
   for path in paths: 
      data, sr = sf.read(path)
      data = data.T 
      song_size = np.shape(data)[1]

      # pad the data
      extra_cols = (LCM - (song_size % LCM)) % LCM
      data = np.pad(data, ((0, 0), (0, extra_cols)))
      
      print(np.shape(data))
      low_0 = low_pass_filter(data[0], SMALL_TO_LARGE_CUTOFF, sr)
      low_1 = low_pass_filter(data[1], SMALL_TO_LARGE_CUTOFF, sr)

      high_0, high_1 = data[0] - low_0, data[1] - low_1

      stack_low_0 = np.reshape(low_0, (N_large, -1))
      stack_low_1 = np.reshape(low_1, (N_large, -1))
      
      data = np.concat((stack_low_0, stack_low_1), axis=1)
      
      t1 = time.time()
      D_large = ksvd.ksvd(D_large, data)
      t2 = time.time()
      print(f"Antrenarea pe {path[7:-4]}, dictionarul mare, a durat {t2-t1} secunde")

      #acum sa facem si bucatile mici
      stack_high_0 = np.reshape(high_0, (N_small, -1))
      stack_high_1 = np.reshape(high_1, (N_small, -1))

      data = np.concat((stack_high_0, stack_high_1), axis=1)
      t1 = time.time()
      D_small = ksvd.ksvd(D_small, data)
      t2 = time.time()
      print(f"Antrenarea pe {path[7:-4]}, dictionarul mare, a durat {t2-t1} secunde")

   #TODO: normalize from the get-go
   np.save("dicts/trained-v3-SMALL.npy", D_small)
   np.save("dicts/trained-v3-LARGE.npy", D_large)

if __name__ == "main":
   main()
import numpy as np 
import soundfile as sf 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data, sr = sf.read("assets/cecil-noised.wav")
N = 512 

# pad the data so we can chunk it with a power of 2, the autoencoders will do better 
cols = np.shape(data)[1]
extra_length = N - (cols % N)
data = np.pad(data, ((0, 0), (0, extra_length)), mode = 'constant')

data = np.reshape(data, (N, -1))

scaler = StandardScaler()
data = scaler.fit_transform(data)

pca = PCA(n_components=0.85, svd_solver='full')
pca.fit(data)

print(pca.n_components_)
import numpy as np
import matplotlib.pyplot as plt
import scipy, fista, omp
import scipy.datasets
import dictionary_builder as dic

N = 1000
time = np.linspace(0, 8, N)

trend = np.array([5*x*x-2*x+4 for x in time])
season = np.array([10*np.sin(2*np.pi*x) + 12*np.sin(4*np.pi*x + np.pi/3) for x in time])
noise = np.random.normal(0, 5, N)

y = trend + season + noise


learned_dict = dic.cosine(N, 2*N)
dict_norms = np.linalg.norm(learned_dict, axis=0)

learned_dict /= dict_norms

print(np.shape(learned_dict))
x = omp.fit(learned_dict.T, y)
    

recovered_signal = learned_dict @ x
print(np.sum(np.abs(recovered_signal - y)))
plt.plot(time, recovered_signal)
plt.show()

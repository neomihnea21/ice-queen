import numpy as np
import matplotlib.pyplot as plt
import scipy, fista, omp, time
import scipy.datasets
import dictionary_builder as dic

N = 256

time_axis = np.linspace(0, 4, N)
trend = np.array([t*t for t in time_axis])
season = np.array([np.sin(4*np.pi*t) for t in time_axis])
noise = np.random.normal(0, 1, N)

y = trend + season + noise

learned_dict = dic.mixed(N, 'db4')
dict_norms = np.linalg.norm(learned_dict, axis=0)

learned_dict /= dict_norms


result = omp.fit_fast(learned_dict.T, y)
print(np.shape(result))
plt.plot(time_axis, y)
plt.plot(time_axis, result)
plt.show()

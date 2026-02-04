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
# this parameter is 
dict_eigen = fista.precompute_step(learned_dict)

t1 = time.time()
result_omp_1 = omp.fit(learned_dict.T, y)
t2 = time.time()
print(f"Timp OMP normal: {t2-t1}")
t1 = time.time()
result_omp_2 = omp.fit_fast(learned_dict.T, y)
t2 = time.time()
print(f"Timp OMP optimizat: {t2-t1}. MSE:{np.mean((result_omp_2 - (trend+season))**2)}")

plt.plot(time_axis, y, color = 'gray', label = 'Semnal cu zgomot')
plt.plot(time_axis, result_omp_2, color = 'aquamarine', label = 'Semnal recuperat')
plt.plot(time_axis, trend + season, color = 'gold', label = 'Semnal original')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Signal Recovery cu OMP")
plt.legend()
plt.savefig("signal-denoising-omp.pdf")
plt.clf()

t1 = time.time()
result_convex = fista.find_best_x(learned_dict, y, dict_eigen, 30, 40)
t2 = time.time()

print(f"MSE: {np.mean((learned_dict @ result_convex - (trend+season)) **2 )}, calculat cu FISTA in {t2-t1}")
plt.plot(time_axis, y, color = 'gray', label = 'Semnal cu zgomot')
plt.plot(time_axis, learned_dict @ result_convex, color = 'aquamarine', label = 'Semnal recuperat')
plt.plot(time_axis, trend + season, color = 'gold', label = 'Semnal original')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Signal Recovery cu FISTA")
plt.legend()
plt.savefig("signal-denoising-fista.pdf")
plt.clf()



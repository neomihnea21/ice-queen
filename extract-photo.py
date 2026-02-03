import numpy as np
import matplotlib.pyplot as plt 
import  omp, sys, scipy
import dictionary_builder as dic
import scipy.datasets

N = 256
# disclaimer: it doesn't work well on photos
learned_dict = dic.mixed(N, 'db4')
dict_norms = np.linalg.norm(learned_dict, axis=0)

learned_dict /= dict_norms

image = scipy.datasets.face(gray = True)
noise = np.random.normal(0, 4, np.shape(image))
image += np.astype(noise, np.uint8)
image = np.clip(image, 0, 255)
plt.imshow(image)
plt.show()
flat_image = np.ndarray.flatten(image)
new_flat = np.zeros_like(flat_image)

for i in range(len(flat_image) // N): 
   block = flat_image[i:i+N]
   denoised_block = omp.fit_fast(learned_dict.T, block, 50)
   new_flat[i:i+N] = denoised_block

new_image = np.reshape(new_flat, np.shape(image))
plt.imshow(new_image)
plt.show()
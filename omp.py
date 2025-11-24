import numpy as np 
import copy
#for now, dict is N * 2N and signal is a N, both np.arrays
def least_squares_fit(atoms, target): 
   left_term = atoms @ atoms.T
   print(np.shape(left_term))
   right_term = atoms @ target
   return np.linalg.solve(left_term, right_term)

def find_sparse_vector(dict, signal, iterations=10): 
    # we do matching pursuit to reconstruct the signal

   residual = copy.deepcopy(signal)
   chosen_dots = []
   for _ in range(iterations):
      #residual -= np.mean(residual) 
      dict -= np.mean(dict, axis=1, keepdims=True)
    
      absolute_values = np.matmul(dict, residual)
      corr_denominators = np.linalg.norm(dict, axis=1) * np.linalg.norm(residual)

      corrs = absolute_values/corr_denominators # there's no good way to not divide
      corrs[list(chosen_dots)] = -np.inf
      print(np.shape(corrs))
      max_pos = np.argmax(corrs)

      chosen_dots.append(max_pos)
      # and now, we must solve a least-squares problem to find the best-fitting coefficients for our atoms 
      # which has a well-known solution
      chosen_atoms = np.array([dict[i] for i in chosen_dots])
      coefs = least_squares_fit(chosen_atoms, signal)
      print(np.shape(coefs))
      
      residual = signal - (coefs @ chosen_atoms)
   return residual


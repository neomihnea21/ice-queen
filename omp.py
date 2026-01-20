import numpy as np
import copy
#for now, dict is N * 2N and signal is a N, both np.arrays
def least_squares_fit(atoms, target): 
   left_term = atoms @ atoms.T
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
      
      residual = signal - (coefs @ chosen_atoms)
   return residual

def fit(dict, y, iterations = 10):
   signal_length = np.shape(dict)[1]
   ans_length = np.shape(dict)[0]
   curr_atoms = []
   residual = copy.deepcopy(y)
   for _ in range(iterations):
      best_i = -1 
      best_quality = -3
      for i in range(signal_length):
         if i not in curr_atoms:
            # assume the dict is normalised
            quality = np.dot(residual, dict[i]) / np.linalg.norm(residual)
            print(np.shape(quality))
            if quality > best_quality:
               best_quality = quality
               best_i = i
      curr_atoms += [best_i]
      atoms = dict[curr_atoms, :]
      # we need to project on a subspace
      # this could be done by Gram-Schmidt, but it's easier to use least-squares
      curr_solution = least_squares_fit(atoms, y)

      curr_signal = np.zeros(signal_length)
      print(np.shape(curr_signal))
      for j in range(len(atoms)):
         curr_signal += atoms[j] * curr_solution[j]
      
      residual = y - curr_signal

   rebuilt_signal = np.zeros(ans_length)
   for j in range(len(curr_atoms)):
      rebuilt_signal[curr_atoms[j]] = curr_solution[j] 
   return rebuilt_signal

#OMP is slow, since it has to do a lot of least-squares fits with heavy overlap
# we might speed that up with reuse of past sols
# which works, based on Cholesky factorization

def fit_fast(D, y, iterations = 10):
   signal_length = np.shape(D)[1]
   L = np.array([1])
   residual = copy.deepcopy(y)
   right_side = []
   curr_atoms = []
   for _ in range(iterations):
      best_i = -1 
      best_quality = -3
      for i in range(signal_length):
         if i not in curr_atoms:
            quality = np.correlate(residual, D[i])
            if quality > best_quality:
               best_quality = quality
               best_i = i
      atoms = D[curr_atoms, :]
      right_side.append(np.dot(D[best_i, :], y))
      # instead of solving least-squares, we peel it back and look at its sol
      if len(curr_atoms) > 1:
         w = np.linalg.solve(L, atoms.T @ D[best_i, :])
         L = np.block([L, np.zeros((np.shape(L)[0], 1))], [w.T, np.ones(1)])
      curr_atoms += [best_i]
      curr_solution = np.linalg.solve(L @ L.T, right_side)

      curr_signal = np.zeros(signal_length)
      for j in range(len(curr_atoms)):
         curr_signal += atoms[j] * curr_solution[j]
      
   rebuilt_signal = np.zeros_like(y)
   for j in range(len(curr_atoms)):
      rebuilt_signal[curr_atoms[j]] = curr_solution[j]
   return rebuilt_signal
import numpy as np
import copy
#for now, dict is N * 2N and signal is a N, both np.arrays
def least_squares_fit(atoms, target): 
   left_term = atoms @ atoms.T
   right_term = atoms @ target
   return np.linalg.solve(left_term, right_term)

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
            quality = np.abs(np.dot(residual, dict[i]) / np.linalg.norm(residual))
            if quality > best_quality:
               best_quality = quality
               best_i = i
      if best_i != -1:         
         curr_atoms += [best_i]
         atoms = dict[curr_atoms, :]
      # we need to project on a subspace
      # this could be done by Gram-Schmidt, but it's easier to use least-squares
      curr_solution, _, _, _ = np.linalg.lstsq(atoms.T, y)

      curr_signal = np.zeros(signal_length)
      for j in range(len(atoms)):
         curr_signal += atoms[j] * curr_solution[j]
      
      residual = y - curr_signal
   return y - residual

#OMP is slow, since it has to do a lot of least-squares fits with heavy overlap
# we might speed that up with reuse of past sols
# which works, based on Cholesky factorization

def fit_fast(D, y, iterations = 10): 
   signal_length = np.shape(D)[1]
   L = None
   residual = copy.deepcopy(y)
   right_side = [] 
   curr_atoms = [] 
   for k in range(iterations):
       best_i = -1
       best_quality = -3
       for i in range(signal_length): 
          if i not in curr_atoms: 
             quality = np.abs(np.dot(residual, D[i]) / np.linalg.norm(residual))
             if quality > best_quality: 
               best_quality = quality 
               best_i = i 
       curr_atoms.append(best_i)
       atoms = D[curr_atoms, :] 
       right_side.append(np.dot(D[best_i, :], y)) 
       if k == 0:
          L = np.array([[1.0]])
       else:
          # remove the current atom, it technically "shouldn't be selected" at this point
          chosen_atoms = D[curr_atoms[:-1], :]
          curr_rhs = chosen_atoms @ D[best_i]
          w = np.linalg.solve(L, curr_rhs) # MAGIC: L is always lower triangular, so it'll take O(n^2), not O(n^3)
          
          # now, the new Cholesky matrix
          L_new = np.zeros((k+1, k+1))
          L_new[: -1, : -1] = L 
          L_new [k, :-1] = w 
          L_new[k, k] = 1

          L = L_new
       z = np.linalg.solve(L, right_side)
       x = np.linalg.solve(L.T, z)
       residual = y - atoms.T @ x
   return y - residual
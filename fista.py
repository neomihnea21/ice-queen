import numpy as np
import copy
# We cannot simply minimize manhattan(y-Dx), there are infinitely many x that make it o
# so we seek the least X, 
# this is best done via FISTA optimization
def soft_threshold(x, l):
   return np.sign(x) * np.maximum(np.abs(x) - l, 0)

def precompute_step(D):
    values, _ = np.linalg.eig(D.T @ D)
    # D.T @ D  is symmetric, but numpy isn't precise
    return np.max(np.abs(values))

# this is an implementation of FISTA with hyperparameters t and lambda
# we can tune them elsewhere
def find_best_x(D, y, l, t = 1, steps = 25):
   solution_size = np.shape(D)[1]
   signal_size = len(y)
   
   if signal_size != np.shape(D)[0]:
      raise ValueError("Nu se potrivesc semnalul si dictionarul")
   
   # first guess
   x = np.zeros(solution_size)
   z = x # this will be modified 
   
   # FISTA should converge fairly fast
   # but we won't run TOO MANY steps, or we recover what we started with
   old_tau = 1
   new_tau = 1
   for _ in range(steps):
      old_solution = copy.deepcopy(x)
      z += D.T @ (y - D @ z) / t 
      x = soft_threshold(z, l/t)
      old_tau = new_tau
      new_tau = (1+np.sqrt(4*(old_tau**2) + 1)) / 2
      
      ratio = (old_tau -1) / new_tau
      z = x + ratio * (x - old_solution)
   return x
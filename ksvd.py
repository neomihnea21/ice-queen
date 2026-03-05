import omp
import numpy as np
import dictionary_builder as dic
def get_representations(D, Y):
    reps = []
    signal_length, n_atoms = np.shape(D)
    _, train_size = np.shape(Y)
    for j in range(train_size):
       _, rep = omp.fit_fast(D.T, Y[:, j])
       reps.append(rep)
    X = np.stack(reps, axis = 0)
    atom_sets = []
    for _ in range(n_atoms):
        atom_sets.append([])
    for i in range(train_size):
       for j in range(n_atoms):
           if X[i, j] != 0: 
               atom_sets[j].append(i)
    return X, atom_sets

def ksvd(D, Y):
   X, atom_sets = get_representations(D, Y)
   _, n_atoms = np.shape(D)
   E = Y - D @ X.T
   E = E.T
   for j in range(n_atoms):
       omega = atom_sets[j]
       if len(omega) != 0:
           E_new = E[omega, :] + np.outer(X[omega, j], D[:, j])
           U, S, V = np.linalg.svd(E_new, full_matrices=False)

           sigma = S[0]
           u = U[:, 0]
           v = V[0]

           D[:, j] = v
           X[omega, j] = sigma * u 
            
           E[omega, :] = E_new -  np.outer(X[omega, j], D[:, j])
   return D

N = 16
 
D = dic.mixed(N, 'db8')
dict_norms = np.linalg.norm(D, axis=0)
D /= dict_norms

Y = np.random.normal(0, 2, (N, 120))

_ = ksvd(D, Y)

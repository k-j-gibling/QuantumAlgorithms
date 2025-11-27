import numpy as np
from fixed_hamiltonian import compute_effective_hamiltonian_normalized


N=5
H_list = [['I','X','Z'], ['I','Z','X'], ['X','I','Z'], ['X','Z','I'],['Z','X','I'],['Z','I','X']]

initial_state = np.array([1,1])/np.sqrt(2)

h_eff = compute_effective_hamiltonian_normalized(H_list, initial_state, N)


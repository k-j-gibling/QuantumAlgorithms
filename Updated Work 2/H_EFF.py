from final import compute_effective_hamiltonian
import numpy as np

H_list = [['I','X','Z'], ['I','Z','X'], ['X','I','Z'], ['X','Z','I'],['Z','X','I'],['Z','I','X']]

initial_state = np.array([1,1])/np.sqrt(2)

h_eff = compute_effective_hamiltonian(H_list,initial_state)


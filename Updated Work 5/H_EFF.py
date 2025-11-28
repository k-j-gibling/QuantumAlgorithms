import numpy as np
from fixed_hamiltonian import compute_effective_hamiltonian_normalized
from final import compute_effective_hamiltonian


N=12
H_list = [['I','X','Z'], ['I','Z','X'], ['X','I','Z'], ['X','Z','I'],['Z','X','I'],['Z','I','X']]

initial_state = np.array([1,1])/np.sqrt(2)

h_eff = compute_effective_hamiltonian_normalized(H_list, initial_state, N)

h_eff_A = compute_effective_hamiltonian(H_list, initial_state)

from specificHamiltonians import prepare_H_test1, prepare_H_test2

h_eff_B = compute_effective_hamiltonian_normalized(prepare_H_test1(N), initial_state, N)
h_eff_C = compute_effective_hamiltonian_normalized(prepare_H_test2(N), initial_state, N)

h_eff_D = compute_effective_hamiltonian(prepare_H_test1(N), initial_state)
h_eff_E = compute_effective_hamiltonian(prepare_H_test2(N), initial_state)
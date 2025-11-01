import numpy as np
from effectiveHamiltonian import compute_effective_hamiltonian
from runge_kutta_hamiltonian import runge_kutta_4

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


'''
Code to implement a hamiltonian, compute the effective hamoltonian,
and compute the time evolution for a given state... Here our
state will be denoted by \ket{\phi}
'''


phi_0 = np.array([1,1])/np.sqrt(2)

H_LIST = [['X','I','X'], ['X','Z', 'I'], ['Z','X','Z']]

H_eff = compute_effective_hamiltonian(H_LIST,phi_0)

# Evolve
t = 1.0
phi_t = runge_kutta_4(phi_0, H_eff, t, dt=0.001)


psi_0 = phi_0
from quantum_state_evolution_complete import *

N = 3


'''
For the given Hamiltonian list, H_LIST, compute
the corresponding Hamiltonian in matrix form.
'''

pauli_dict = dict()
pauli_dict['I'] = I
pauli_dict['X'] = X
pauli_dict['Y'] = Y
pauli_dict['Z'] = Z

H = 0
for element in H_LIST:
	tensor_product = pauli_dict[element[0]]

	for i in range(1,len(element)):
		current_matrix = pauli_dict[element[i]]
		tensor_product = np.kron(tensor_product,current_matrix)

	H += tensor_product



#TODO: Take n copies of this state.
r = evolve_and_trace(H, psi_0, N, t)


#TODO: Compute the density matrix for phi_t.


#TODO: Prepare the state \ket{\psi(0)}.

#TODO: Create N copies of this state.

#Evolve this state under e^{iHt}.

#Take the partial trace of this state.




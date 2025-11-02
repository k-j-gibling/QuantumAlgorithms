import numpy as np


def create_N_copy_state(N, single_qubit_state):
	productVector = single_qubit_state

	for i in range(N-1):
		productVector = np.kron(productVector, single_qubit_state)

	return productVector


v = np.array([1, 1])/np.sqrt(2)
v_N = create_N_copy_state(4,v)
import numpy as np


# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


"""
Expectation Value Computation
==============================

Compute ⟨φ|A|φ⟩ for a given operator A and state |φ⟩
"""

def expectation_value(A, phi):
    """
    Compute the expectation value ⟨φ|A|φ⟩
    
    Parameters:
    -----------
    A : array_like, shape (n, n)
        Operator matrix
    phi : array_like, shape (n,)
        State vector |φ⟩
    
    Returns:
    --------
    exp_val : complex
        The expectation value ⟨φ|A|φ⟩ = φ† A φ
    
    Examples:
    ---------
    >>> # Compute ⟨0|Z|0⟩
    >>> Z = np.array([[1, 0], [0, -1]])
    >>> phi = np.array([1, 0])
    >>> result = expectation_value(Z, phi)
    >>> print(result)  # Should be 1.0
    
    >>> # Compute ⟨+|X|+⟩
    >>> X = np.array([[0, 1], [1, 0]])
    >>> phi = np.array([1, 1]) / np.sqrt(2)
    >>> result = expectation_value(X, phi)
    >>> print(result)  # Should be 1.0
    """
    A = np.asarray(A, dtype=complex)
    phi = np.asarray(phi, dtype=complex)
    
    # Compute ⟨φ|A|φ⟩ = φ† A φ
    # where φ† is the conjugate transpose (bra)
    exp_val = np.dot(phi.conj(), np.dot(A, phi))
    
    return exp_val



'''
Function that computes the effective hamiltonian of a given
input hamiltonian, and returns the effective Hamiltonian.
'''
def compute_effective_hamiltonian(H_list, stateVector):
	H_eff_list = []

	for elmt in H_list:
		if elmt[0] != 'I':
			H_eff_list.append(elmt)


	H_eff_info_list = []

	for elmt in H_eff_list:
		elmt_operator = elmt[0]

		H_eff_info_list.append(dict())
		H_eff_info_list[-1]['operator'] = elmt_operator #For H_{eff}^1 the first element will be the operator.
		H_eff_info_list[-1]['operator_coefficient'] = 1 #Initialize to 1.
		H_eff_info_list[-1]['product_term_list'] = [] #Initially an empty list.

		for i in range(1,len(elmt)):
			if elmt[i] == elmt_operator:
				H_eff_info_list[-1]['operator_coefficient'] += 1
				continue

			if elmt[i] == 'I':
				continue

			H_eff_info_list[-1]['product_term_list'].append(elmt[i])


	matrices = []


	for elmt in H_eff_info_list:
		operator = elmt['operator']
		operator_coeff = elmt['operator_coefficient']
		product_terms = elmt['product_term_list']

		product_chain = 1


		for product_term in product_terms:
			#if product_term == 'X':
			#	expectationValue = expectation_value(X, stateVector)
			#	product_chain = product_chain*expectationValue
			expectationValue = expectation_value(pauliDict[product_term], stateVector)
			product_chain = product_chain*expectationValue

		currentMatrix = None

		currentMatrix = pauli_dict[operator]

		"""if operator == 'X':
			currentMatrix = X
		elif currentMatrix == 'Y':
			currentMatrix = Y
		else:
			currentMatrix = Z"""

		currentMatrix = currentMatrix*operator_coeff*product_chain
		matrices.append(currentMatrix)


	H_effective = matrices[0]

	for i in range(1,len(matrices)):
		H_effective += matrices[i]

	return H_effective



H_LIST = [['X','I','X'], ['X','Z', 'I'], ['Z','X','Z']]
psi = np.array([0,1])


H_eff = compute_effective_hamiltonian(H_LIST,psi)














	
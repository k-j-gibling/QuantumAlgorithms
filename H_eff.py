'''


Create the state phi with phi(0) = psi(0).
Compute the effective hamiltonian (todo: we need to know how to compute the effective Hamiltonian).
Estimate the state evolution of 
'''


'''
Program code to compute the effective hamiltonian, given an input hamiltonian in the form of a H_list.
'''

H_list = [['X','I','X'], ['X','Z', 'I'], ['Z','X','Z']]

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


"""
Expectation Value Computation
==============================

Compute ⟨φ|A|φ⟩ for a given operator A and state |φ⟩
"""

import numpy as np


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



psi = np.array([1,0])

# Ensure psi is a column vector
#if psi.ndim == 1:
#	psi = psi.reshape(-1, 1)


# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
# Test 1: ⟨0|Z|0⟩
phi_0 = np.array([1, 0])
result = expectation_value(Z, phi_0)
print(f"⟨0|Z|0⟩ = {result}")
print(f"Expected: 1.0")
print()
    
# Test 2: ⟨1|Z|1⟩
phi_1 = np.array([0, 1])
result = expectation_value(Z, phi_1)
print(f"⟨1|Z|1⟩ = {result}")
print(f"Expected: -1.0")
print()
    
# Test 3: ⟨+|X|+⟩
phi_plus = np.array([1, 1]) / np.sqrt(2)
result = expectation_value(X, phi_plus)
print(f"⟨+|X|+⟩ = {result}")
print(f"Expected: 1.0")
print()
    
# Test 4: ⟨+|Z|+⟩
result = expectation_value(Z, phi_plus)
print(f"⟨+|Z|+⟩ = {result}")
print(f"Expected: 0.0")
print()
    
# Test 5: Custom state
alpha, beta = 0.6, 0.8
phi_custom = np.array([alpha, beta])
result_X = expectation_value(X, phi_custom)
result_Z = expectation_value(Z, phi_custom)
print(f"Custom state |φ⟩ = {alpha}|0⟩ + {beta}|1⟩")
print(f"⟨φ|X|φ⟩ = {result_X}")
print(f"⟨φ|Z|φ⟩ = {result_Z}")
print(f"Expected ⟨Z⟩ = |α|² - |β|² = {alpha**2 - beta**2}")
print()
    
print("="*50)
print("All tests completed!")


matrices = []


for elmt in H_eff_info_list:
	operator = elmt['operator']
	operator_coeff = elmt['operator_coefficient']
	product_terms = elmt['product_term_list']

	product_chain = 1


	for product_term in product_terms:
		if product_term == 'X':
			expectationValue = expectation_value(X, psi)
			product_chain = product_chain*expectationValue

	currentMatrix = None

	if operator == 'X':
		currentMatrix = X
	elif currentMatrix == 'Y':
		currentMatrix = Y
	else:
		currentMatrix = Z

	currentMatrix = currentMatrix*operator_coeff*product_chain
	matrices.append(currentMatrix)


H_effective = matrices[0]

for i in range(1,len(matrices)):
	H_effective += matrices[i]




import numpy as np


def create_N_copy_state(N, single_qubit_state):
	productVector = single_qubit_state

	for i in range(N-1):
		productVector = np.kron(productVector, single_qubit_state)

	return productVector


v = np.array([1, 1])/np.sqrt(2)
v_N = create_N_copy_state(4,v)


def insert_bit(index, position, bit_value, total_bits):
    """
    Insert a bit at a specific position in binary representation.
    
    Parameters:
    -----------
    index : int
        Original index (without the bit)
    position : int
        Position to insert (0 = rightmost)
    bit_value : int
        Value of bit to insert (0 or 1)
    total_bits : int
        Total number of bits after insertion
    
    Returns:
    --------
    new_index : int
        Index with bit inserted
    
    Example:
    --------
    >>> insert_bit(0b101, 1, 0, 4)  # Insert 0 at position 1
    0b1001  # = 9
    """
    # Split index into parts before and after insertion point
    lower_mask = (1 << position) - 1
    lower_bits = index & lower_mask
    upper_bits = (index >> position) << (position + 1)
    
    # Insert the bit
    new_index = upper_bits | (bit_value << position) | lower_bits
    
    return new_index

def partial_trace(rho, N, keep_qubits):
    """
    Compute partial trace of density matrix.
    
    For N qubits, trace out all qubits except those in keep_qubits.
    
    Parameters:
    -----------
    rho : array_like, shape (2^N, 2^N)
        Density matrix for N qubits
    N : int
        Total number of qubits
    keep_qubits : list of int
        Indices of qubits to keep (0-indexed)
        Example: [0] keeps first qubit, [0,1] keeps first two
    
    Returns:
    --------
    rho_reduced : ndarray, shape (2^k, 2^k)
        Reduced density matrix, where k = len(keep_qubits)
    
    Examples:
    ---------
    >>> # 2-qubit system, trace out second qubit
    >>> rho_1 = partial_trace(rho, N=2, keep_qubits=[0])
    
    >>> # 4-qubit system, keep only first qubit
    >>> rho_1 = partial_trace(rho, N=4, keep_qubits=[0])
    
    >>> # 4-qubit system, keep first two qubits
    >>> rho_12 = partial_trace(rho, N=4, keep_qubits=[0, 1])
    """
    rho = np.asarray(rho, dtype=complex)
    
    keep_qubits = sorted(list(keep_qubits))
    trace_qubits = sorted([i for i in range(N) if i not in keep_qubits], reverse=True)
    
    # Work with a copy
    rho_current = rho.copy()
    N_current = N
    
    # Trace out qubits one by one
    for qubit in trace_qubits:
        dim = 2**N_current
        dim_reduced = 2**(N_current - 1)
        
        # Create reduced density matrix
        rho_new = np.zeros((dim_reduced, dim_reduced), dtype=complex)
        
        # Sum over the traced qubit basis
        for basis_state in range(2):  # 0 or 1
            # Create projector |basis_state⟩⟨basis_state| for the traced qubit
            # We need to identify which indices correspond to this qubit being in this state
            
            for i in range(dim_reduced):
                for j in range(dim_reduced):
                    # Insert basis_state at position 'qubit' in binary representation
                    i_full = insert_bit(i, qubit, basis_state, N_current)
                    j_full = insert_bit(j, qubit, basis_state, N_current)
                    rho_new[i, j] += rho_current[i_full, j_full]
        
        rho_current = rho_new
        N_current -= 1
    
    return rho_current


A = np.array([[2, 1,1,2],[2,3, 2,4], [4,3,2,5], [2,3,4,6]])

rho = partial_trace(A, N=2, keep_qubits=[0])


# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

pauli_dict = dict()
pauli_dict['I'] = I
pauli_dict['X'] = X
pauli_dict['Y'] = Y
pauli_dict['Z'] = Z

def H_list_to_H(H_list):
	H_ = 0
	for element in H_LIST:
		tensor_product = pauli_dict[element[0]]

		for i in range(1,len(element)):
			current_matrix = pauli_dict[element[i]]
			tensor_product = np.kron(tensor_product,current_matrix)

		H_ += tensor_product

	return H_





psi_0 = np.array([1,1])/np.sqrt(2)

N = 3

H_LIST = [['X','I','X'], ['X','Z', 'I'], ['Z','X','Z']] #Acts on a 3 qubit system (N=3).

#H_eff = compute_effective_hamiltonian(H_LIST,psi_0)

H = H_list_to_H(H_LIST)





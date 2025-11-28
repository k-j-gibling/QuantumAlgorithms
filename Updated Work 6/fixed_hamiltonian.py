import numpy as np

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

pauli_dict = {
    'I': I,
    'X': X,
    'Y': Y,
    'Z': Z
}

def expectation_value(A, phi):
    """Compute ⟨φ|A|φ⟩"""
    A = np.asarray(A, dtype=complex)
    phi = np.asarray(phi, dtype=complex)
    exp_val = np.dot(phi.conj(), np.dot(A, phi))
    return exp_val


def compute_effective_hamiltonian(H_list, stateVector, N):
    """
    Compute the effective Hamiltonian for mean-field theory.
    
    Parameters:
    -----------
    H_list : list of lists
        Each element is a list of Pauli operators representing a term in H
        E.g., [['X','Z'], ['Z','X']] represents X⊗Z + Z⊗X
    stateVector : np.ndarray
        Single-particle state |φ⟩
    N : int
        Total number of particles in the N-body system
    
    Returns:
    --------
    H_eff : np.ndarray
        Effective single-particle Hamiltonian
        
    Theory:
    -------
    For a k-body interaction term in the Hamiltonian, when computing the 
    mean-field effective Hamiltonian, we get a factor of:
    - Number of ways to choose the (k-1) other particles: binomial(N-1, k-1)
    - Expectation values of the (k-1) operators acting on other particles
    
    For 2-body: (N-1) * <operator>
    For 3-body: (N-1)(N-2) * <op1> * <op2>
    etc.
    """
    
    H_eff = np.zeros((2, 2), dtype=complex)
    
    for term in H_list:
        # Count how many non-identity operators in this term
        # This tells us it's a k-body interaction
        k_body = sum(1 for op in term if op != 'I')
        
        if k_body == 0:
            # All identity - just a constant, skip
            continue
        
        # Find the first non-identity operator - this will be our effective operator
        eff_operator = None
        eff_operator_idx = None
        for idx, op in enumerate(term):
            if op != 'I':
                eff_operator = op
                eff_operator_idx = idx
                break
        
        if eff_operator is None:
            continue
            
        # Compute the product of expectation values for all OTHER non-identity operators
        expectation_product = 1.0
        for idx, op in enumerate(term):
            if idx != eff_operator_idx and op != 'I':
                exp_val = expectation_value(pauli_dict[op], stateVector)
                expectation_product *= exp_val
        
        # Compute the combinatorial factor
        # For k-body interaction, we get product from i=0 to k-2 of (N-1-i)
        # For 2-body: (N-1)
        # For 3-body: (N-1)(N-2)
        # For 4-body: (N-1)(N-2)(N-3)
        combinatorial_factor = 1
        for i in range(k_body - 1):
            combinatorial_factor *= (N - 1 - i)
        
        # Add this term's contribution to H_eff
        contribution = combinatorial_factor * expectation_product * pauli_dict[eff_operator]
        H_eff += contribution
    
    return H_eff


def compute_effective_hamiltonian_normalized(H_list, stateVector, N):
    """
    Compute the effective Hamiltonian when the N-body Hamiltonian has 
    normalization factors.
    
    This version assumes your N-body Hamiltonian has the form:
    H_N = (1 / normalization_factor) * sum of k-body terms
    
    For example:
    - 2-body with 1/(N-1): H_N = (1/(N-1)) * sum_{i≠j} V_ij
    - 3-body with 1/((N-1)(N-2)): H_N = (1/((N-1)(N-2))) * sum_{i≠j≠k} V_ijk
    
    In these cases, the normalization factor cancels with the combinatorial
    factor, giving simpler expressions for H_eff.
    """
    
    H_eff = np.zeros((2, 2), dtype=complex)
    
    for term in H_list:
        # Count k-body interaction
        k_body = sum(1 for op in term if op != 'I')
        
        if k_body == 0:
            continue
        
        # Find the first non-identity operator
        eff_operator = None
        eff_operator_idx = None
        for idx, op in enumerate(term):
            if op != 'I':
                eff_operator = op
                eff_operator_idx = idx
                break
        
        if eff_operator is None:
            continue
            
        # Compute expectation value product
        expectation_product = 1.0
        for idx, op in enumerate(term):
            if idx != eff_operator_idx and op != 'I':
                exp_val = expectation_value(pauli_dict[op], stateVector)
                expectation_product *= exp_val
        
        # If your H_N already has 1/((N-1)(N-2)...) normalization,
        # then it cancels with the combinatorial factor and we just get:
        contribution = expectation_product * pauli_dict[eff_operator]
        H_eff += contribution
    
    return H_eff


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test case 1: 2-body Hamiltonian
    print("=" * 60)
    print("TEST 1: 2-body Hamiltonian")
    print("H_N = (1/(N-1)) * (X⊗Z + Z⊗X)")
    print("=" * 60)
    
    psi_0 = np.array([1, 1]) / np.sqrt(2)  # |+⟩ state
    N = 10
    
    H_list_2body = [['X', 'Z'], ['Z', 'X']]
    
    # Compute <X> and <Z>
    exp_X = expectation_value(X, psi_0)
    exp_Z = expectation_value(Z, psi_0)
    print(f"\n⟨X⟩ = {exp_X:.6f}")
    print(f"⟨Z⟩ = {exp_Z:.6f}")
    
    # For NORMALIZED Hamiltonian (with 1/(N-1) factor)
    H_eff_normalized = compute_effective_hamiltonian_normalized(H_list_2body, psi_0, N)
    print(f"\nH_eff (with normalization factor in H_N):")
    print("Expected: ⟨Z⟩X + ⟨X⟩Z")
    print(f"Expected matrix:\n{exp_Z * X + exp_X * Z}")
    print(f"\nComputed matrix:\n{H_eff_normalized}")
    print(f"Match: {np.allclose(H_eff_normalized, exp_Z * X + exp_X * Z)}")
    
    # For UNNORMALIZED Hamiltonian (without factor)
    H_eff_unnormalized = compute_effective_hamiltonian(H_list_2body, psi_0, N)
    print(f"\nH_eff (without normalization factor in H_N):")
    print(f"Expected: (N-1)(⟨Z⟩X + ⟨X⟩Z) = {N-1}(⟨Z⟩X + ⟨X⟩Z)")
    expected_unnorm = (N-1) * (exp_Z * X + exp_X * Z)
    print(f"Expected matrix:\n{expected_unnorm}")
    print(f"\nComputed matrix:\n{H_eff_unnormalized}")
    print(f"Match: {np.allclose(H_eff_unnormalized, expected_unnorm)}")
    
    # Test case 2: 3-body Hamiltonian
    print("\n" + "=" * 60)
    print("TEST 2: 3-body Hamiltonian")
    print("H_N = (1/((N-1)(N-2))) * (X⊗Z⊗Z + Z⊗X⊗Z + Z⊗Z⊗X)")
    print("=" * 60)
    
    H_list_3body = [['X', 'Z', 'Z'], ['Z', 'X', 'Z'], ['Z', 'Z', 'X']]
    
    # For NORMALIZED Hamiltonian
    H_eff_3body_norm = compute_effective_hamiltonian_normalized(H_list_3body, psi_0, N)
    print(f"\nH_eff (with normalization factor in H_N):")
    print("Expected: ⟨Z⟩²X + 2⟨X⟩⟨Z⟩Z")
    expected_3body = exp_Z**2 * X + 2 * exp_X * exp_Z * Z
    print(f"Expected matrix:\n{expected_3body}")
    print(f"\nComputed matrix:\n{H_eff_3body_norm}")
    print(f"Match: {np.allclose(H_eff_3body_norm, expected_3body)}")
    
    # For UNNORMALIZED Hamiltonian
    H_eff_3body_unnorm = compute_effective_hamiltonian(H_list_3body, psi_0, N)
    print(f"\nH_eff (without normalization factor in H_N):")
    print(f"Expected: (N-1)(N-2)(⟨Z⟩²X + 2⟨X⟩⟨Z⟩Z)")
    expected_3body_unnorm = (N-1) * (N-2) * expected_3body
    print(f"Expected matrix:\n{expected_3body_unnorm}")
    print(f"\nComputed matrix:\n{H_eff_3body_unnorm}")
    print(f"Match: {np.allclose(H_eff_3body_unnorm, expected_3body_unnorm)}")

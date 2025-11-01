"""
Quantum State Evolution for Permutation Symmetric Hamiltonians
================================================================

This module provides a complete workflow for:
1. Constructing permutation symmetric Hamiltonians
2. Evolving quantum states using e^{-iHt}
3. Computing density matrices
4. Taking partial traces to extract single-copy states

For research on mean-field quantum systems with N identical qubits.

Date: October 31, 2025
"""

import numpy as np
from scipy.linalg import eig, expm
from itertools import combinations
import warnings


# ============================================================================
# PART 1: HAMILTONIAN CONSTRUCTION
# ============================================================================

# Pauli operators
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(*ops):
    """Compute tensor product of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def add_single_Z(N, coefficient):
    """Add h * Σᵢ Zᵢ term (longitudinal field)."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        ops = [I]*N
        ops[i] = Z
        H += coefficient * tensor(*ops)
    return H


def add_single_X(N, coefficient):
    """Add h * Σᵢ Xᵢ term (transverse field)."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        ops = [I]*N
        ops[i] = X
        H += coefficient * tensor(*ops)
    return H


def add_ZZ_interaction(N, J):
    """Add J * Σᵢ<ⱼ Zᵢ ⊗ Zⱼ term (Ising interaction)."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i, j in combinations(range(N), 2):
        ops = [I]*N
        ops[i] = Z
        ops[j] = Z
        H += J * tensor(*ops)
    return H


def add_XX_interaction(N, J):
    """Add J * Σᵢ<ⱼ Xᵢ ⊗ Xⱼ term (XX interaction)."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i, j in combinations(range(N), 2):
        ops = [I]*N
        ops[i] = X
        ops[j] = X
        H += J * tensor(*ops)
    return H


# ============================================================================
# PART 2: STATE EVOLUTION
# ============================================================================

def evolve_state(psi0, H, t, method='auto'):
    """
    Evolve quantum state: |ψ(t)⟩ = e^{-iHt} |ψ(0)⟩
    
    Parameters:
    -----------
    psi0 : array_like, shape (d,)
        Initial state vector
    H : array_like, shape (d, d)
        Hamiltonian matrix
    t : float
        Evolution time
    method : str, optional
        'auto', 'diagonalize', or 'expm'
    
    Returns:
    --------
    psi_t : ndarray
        Evolved state
    info : dict
        Information about the evolution
    """
    psi0 = np.asarray(psi0, dtype=complex)
    H = np.asarray(H, dtype=complex)
    
    # Normalize initial state
    psi0 = psi0 / np.linalg.norm(psi0)
    
    # Check Hermiticity
    if not np.allclose(H, H.conj().T, atol=1e-10):
        warnings.warn("Hamiltonian is not Hermitian", UserWarning)
    
    # Choose method
    if method == 'auto' or method == 'diagonalize':
        # Diagonalize H
        eigenvalues, U = eig(H)
        exp_diag = np.exp(-1j * eigenvalues * t)
        psi_t = U @ (exp_diag * (U.conj().T @ psi0))
    elif method == 'expm':
        # Direct matrix exponential
        U = expm(-1j * H * t)
        psi_t = U @ psi0
    else:
        raise ValueError(f"Unknown method: {method}")
    
    info = {
        'method': method,
        'norm_initial': np.linalg.norm(psi0),
        'norm_final': np.linalg.norm(psi_t),
        'unitarity_error': abs(np.linalg.norm(psi_t) - 1.0)
    }
    
    return psi_t, info


# ============================================================================
# PART 3: DENSITY MATRIX COMPUTATION
# ============================================================================

def state_to_density_matrix(psi):
    """
    Convert pure state to density matrix: ρ = |ψ⟩⟨ψ|
    
    Parameters:
    -----------
    psi : array_like, shape (d,)
        State vector
    
    Returns:
    --------
    rho : ndarray, shape (d, d)
        Density matrix
    """
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)  # Normalize
    rho = np.outer(psi, psi.conj())
    return rho


def verify_density_matrix(rho, tol=1e-10):
    """
    Verify that ρ is a valid density matrix.
    
    Checks:
    1. Hermitian: ρ = ρ†
    2. Trace one: Tr(ρ) = 1
    3. Positive semi-definite: eigenvalues ≥ 0
    
    Returns:
    --------
    is_valid : bool
    errors : dict
    """
    errors = {}
    
    # Check Hermitian
    hermitian_error = np.max(np.abs(rho - rho.conj().T))
    errors['hermitian_error'] = hermitian_error
    
    # Check trace
    trace = np.trace(rho)
    errors['trace_error'] = abs(trace - 1.0)
    
    # Check positive semi-definite
    eigenvalues = np.linalg.eigvalsh(rho)
    min_eigenvalue = np.min(eigenvalues)
    errors['min_eigenvalue'] = min_eigenvalue
    
    is_valid = (hermitian_error < tol and 
                abs(trace - 1.0) < tol and 
                min_eigenvalue > -tol)
    
    return is_valid, errors


# ============================================================================
# PART 4: PARTIAL TRACE (THE KEY OPERATION!)
# ============================================================================

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


def get_single_qubit_state(rho, N, qubit_index=0):
    """
    Extract single-qubit reduced density matrix.
    
    Convenience function for partial_trace when keeping only one qubit.
    
    Parameters:
    -----------
    rho : array_like, shape (2^N, 2^N)
        Full density matrix
    N : int
        Total number of qubits
    qubit_index : int, optional
        Which qubit to extract (default: 0, the first qubit)
    
    Returns:
    --------
    rho_1 : ndarray, shape (2, 2)
        Single-qubit density matrix
    """
    return partial_trace(rho, N, keep_qubits=[qubit_index])


# ============================================================================
# PART 5: COMPLETE WORKFLOW FUNCTION
# ============================================================================

def evolve_and_trace(H, single_qubit_state, N_copies, t, keep_qubits=[0], method='auto', verbose=True):
    """
    Complete workflow: Create N copies, evolve, and compute partial trace.
    
    This is the main function for your research workflow!
    
    Workflow:
    1. Create N copies: |ψ(0)⟩ = |ψ_single⟩^⊗N
    2. Evolve: |ψ(t)⟩ = e^{-iHt}|ψ(0)⟩
    3. Compute density matrix: ρ(t) = |ψ(t)⟩⟨ψ(t)|
    4. Partial trace to get single-copy state back
    
    Parameters:
    -----------
    H : array_like, shape (2^N, 2^N)
        Permutation symmetric Hamiltonian for N qubits
    single_qubit_state : array_like, shape (2,)
        Single-qubit initial state |ψ_single⟩ = [α, β]
        This will be copied N times: |ψ⟩^⊗N
    N_copies : int
        Number of copies to create
    t : float
        Evolution time
    keep_qubits : list of int, optional
        Which qubits to keep after tracing (default: [0])
        For single-qubit extraction, use [0]
    method : str, optional
        Evolution method: 'auto', 'diagonalize', or 'expm'
    verbose : bool, optional
        Print progress information
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'single_qubit_input': Original single-qubit state
        - 'N_copies': Number of copies
        - 'psi_0': Initial N-copy state |ψ⟩^⊗N
        - 'psi_t': Evolved state
        - 'rho_full': Full density matrix ρ(t)
        - 'rho_reduced': Reduced density matrix after partial trace
        - 'evolution_info': Information about evolution
        - 'density_info': Information about density matrix
    
    Examples:
    ---------
    >>> # Single qubit in |0⟩, create 4 copies, evolve, extract back
    >>> H = construct_TFIM(N=4, J=1.0, h=0.5)
    >>> single_state = [1, 0]  # |0⟩
    >>> results = evolve_and_trace(H, single_state, N_copies=4, t=1.0)
    >>> rho_1 = results['rho_reduced']  # Single-qubit state after evolution
    
    >>> # Single qubit in |+⟩, create 6 copies
    >>> single_state = [1, 1]/np.sqrt(2)  # |+⟩
    >>> results = evolve_and_trace(H, single_state, N_copies=6, t=2.0)
    """
    
    if verbose:
        print("="*70)
        print("QUANTUM STATE EVOLUTION AND PARTIAL TRACE")
        print("="*70)
        print(f"Single-qubit input state: {single_qubit_state}")
        print(f"Number of copies: N = {N_copies}")
        print(f"Hilbert space dimension: {2**N_copies}")
        print(f"Evolution time: t = {t}")
        print(f"Keeping qubits: {keep_qubits}")
        print()
    
    # Step 0: Create N copies of the single-qubit state
    if verbose:
        print("Step 0: Creating N copies of single-qubit state...")
    psi0 = create_product_state(N_copies, single_qubit_state)
    if verbose:
        print(f"  ✓ Created product state |ψ⟩^⊗{N_copies}")
        print(f"  Dimension: {len(psi0)}")
        print()
    
    # Step 1: Evolve state
    if verbose:
        print("Step 1: Evolving N-copy state...")
    psi_t, evolution_info = evolve_state(psi0, H, t, method=method)
    
    if verbose:
        print(f"  ✓ Evolution complete")
        print(f"  Method: {evolution_info['method']}")
        print(f"  Unitarity error: {evolution_info['unitarity_error']:.2e}")
        print()
    
    # Step 2: Compute density matrix
    if verbose:
        print("Step 2: Computing density matrix...")
    rho_full = state_to_density_matrix(psi_t)
    is_valid, density_errors = verify_density_matrix(rho_full)
    
    if verbose:
        print(f"  ✓ Density matrix computed")
        print(f"  Valid: {is_valid}")
        print(f"  Trace: {np.trace(rho_full):.6f}")
        print(f"  Purity: {np.trace(rho_full @ rho_full).real:.6f}")
        print()
    
    # Step 3: Partial trace
    if verbose:
        print("Step 3: Computing partial trace to extract single-copy state...")
    rho_reduced = partial_trace(rho_full, N_copies, keep_qubits)
    is_valid_reduced, reduced_errors = verify_density_matrix(rho_reduced)
    
    if verbose:
        print(f"  ✓ Partial trace computed")
        print(f"  Reduced dimension: {rho_reduced.shape[0]}x{rho_reduced.shape[0]}")
        print(f"  Valid: {is_valid_reduced}")
        print(f"  Trace: {np.trace(rho_reduced):.6f}")
        print(f"  Purity: {np.trace(rho_reduced @ rho_reduced).real:.6f}")
        print()
    
    # Compile results
    results = {
        'single_qubit_input': np.asarray(single_qubit_state),
        'N_copies': N_copies,
        'psi_0': psi0,
        'psi_t': psi_t,
        'rho_full': rho_full,
        'rho_reduced': rho_reduced,
        'evolution_info': evolution_info,
        'density_info': {
            'full_valid': is_valid,
            'full_errors': density_errors,
            'reduced_valid': is_valid_reduced,
            'reduced_errors': reduced_errors,
            'full_purity': np.trace(rho_full @ rho_full).real,
            'reduced_purity': np.trace(rho_reduced @ rho_reduced).real
        }
    }
    
    if verbose:
        print("="*70)
        print("COMPLETE!")
        print("="*70)
        print()
    
    return results


# ============================================================================
# PART 6: PRE-BUILT HAMILTONIANS
# ============================================================================

def construct_TFIM(N, J=1.0, h=0.5):
    """
    Transverse Field Ising Model.
    H = -J Σᵢ<ⱼ Zᵢ⊗Zⱼ - h Σᵢ Xᵢ
    """
    H = -J * add_ZZ_interaction(N, 1.0)
    H -= h * add_single_X(N, 1.0)
    return H


def construct_XY_model(N, J=0.5):
    """
    XY Model (X-only approximation).
    H = J Σᵢ<ⱼ Xᵢ⊗Xⱼ
    """
    return J * add_XX_interaction(N, 1.0)


def construct_general_MF(N, h_x=0.3, h_z=0.1, J_xx=0.2, J_zz=0.4):
    """
    General Mean-Field Hamiltonian.
    H = Σᵢ (hₓXᵢ + hzZᵢ) + Σᵢ<ⱼ (JₓₓXᵢ⊗Xⱼ + JzzZᵢ⊗Zⱼ)
    """
    H = h_x * add_single_X(N, 1.0)
    H += h_z * add_single_Z(N, 1.0)
    H += J_xx * add_XX_interaction(N, 1.0)
    H += J_zz * add_ZZ_interaction(N, 1.0)
    return H


# ============================================================================
# PART 7: UTILITY FUNCTIONS
# ============================================================================

def create_product_state(N, single_qubit_state):
    """
    Create product state |ψ⟩^⊗N from single-qubit state.
    
    Parameters:
    -----------
    N : int
        Number of copies
    single_qubit_state : array_like, shape (2,)
        Single-qubit state [α, β]
    
    Returns:
    --------
    psi : ndarray, shape (2^N,)
        Product state
    
    Examples:
    ---------
    >>> # Create |0⟩^⊗4
    >>> psi = create_product_state(4, [1, 0])
    
    >>> # Create |+⟩^⊗4
    >>> psi = create_product_state(4, [1, 1]/np.sqrt(2))
    """
    single_qubit_state = np.asarray(single_qubit_state, dtype=complex)
    single_qubit_state = single_qubit_state / np.linalg.norm(single_qubit_state)
    
    psi = single_qubit_state
    for _ in range(N-1):
        psi = np.kron(psi, single_qubit_state)
    
    return psi


def bloch_vector(rho_1):
    """
    Compute Bloch vector from single-qubit density matrix.
    
    For ρ = (I + r·σ)/2, returns r = (r_x, r_y, r_z)
    
    Parameters:
    -----------
    rho_1 : array_like, shape (2, 2)
        Single-qubit density matrix
    
    Returns:
    --------
    r : ndarray, shape (3,)
        Bloch vector [r_x, r_y, r_z]
    """
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    r_x = 2 * np.real(rho_1[0, 1])
    r_y = 2 * np.real(-1j * rho_1[0, 1])
    r_z = np.real(rho_1[0, 0] - rho_1[1, 1])
    
    return np.array([r_x, r_y, r_z])


# ============================================================================
# PART 8: DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEMONSTRATION: Quantum State Evolution with Partial Trace")
    print("="*70)
    print()
    
    # Parameters
    N_copies = 4  # Number of copies
    t = 1.0  # Evolution time
    
    # Example 1: TFIM with |0⟩ initial state
    print("EXAMPLE 1: Transverse Field Ising Model")
    print("-"*70)
    print("Single-qubit input: |0⟩")
    print()
    
    H1 = construct_TFIM(N_copies, J=1.0, h=0.5)
    single_state_1 = [1, 0]  # |0⟩
    
    results1 = evolve_and_trace(H1, single_state_1, N_copies, t, keep_qubits=[0])
    
    rho_1 = results1['rho_reduced']
    r = bloch_vector(rho_1)
    print(f"Single-qubit density matrix after evolution:")
    print(rho_1)
    print(f"\nBloch vector: r = {r}")
    print(f"|r| = {np.linalg.norm(r):.4f} (purity)")
    print()
    
    # Example 2: Different initial state
    print("EXAMPLE 2: Initial state |+⟩")
    print("-"*70)
    print("Single-qubit input: |+⟩ = (|0⟩ + |1⟩)/√2")
    print()
    
    single_state_2 = np.array([1, 1]) / np.sqrt(2)  # |+⟩
    
    results2 = evolve_and_trace(H1, single_state_2, N_copies, t, keep_qubits=[0], verbose=False)
    
    rho_2 = results2['rho_reduced']
    r2 = bloch_vector(rho_2)
    print(f"Single-qubit density matrix after evolution:")
    print(rho_2)
    print(f"\nBloch vector: r = {r2}")
    print(f"|r| = {np.linalg.norm(r2):.4f}")
    print()
    
    # Example 3: Extract 2-qubit state
    print("EXAMPLE 3: Extract 2-qubit reduced state")
    print("-"*70)
    print("Keeping first two qubits after evolution")
    print()
    
    results3 = evolve_and_trace(H1, single_state_1, N_copies, t, keep_qubits=[0, 1], verbose=False)
    
    rho_12 = results3['rho_reduced']
    print(f"Two-qubit density matrix shape: {rho_12.shape}")
    print(f"Trace: {np.trace(rho_12):.6f}")
    print(f"Purity: {np.trace(rho_12 @ rho_12).real:.6f}")
    print()
    
    # Example 4: Compare different Hamiltonians
    print("EXAMPLE 4: Comparing different Hamiltonians")
    print("-"*70)
    print("Same initial state |0⟩, different Hamiltonians")
    print()
    
    # XY model
    H_xy = construct_XY_model(N_copies, J=0.5)
    results_xy = evolve_and_trace(H_xy, single_state_1, N_copies, t, keep_qubits=[0], verbose=False)
    rho_xy = results_xy['rho_reduced']
    r_xy = bloch_vector(rho_xy)
    
    # General MF
    H_mf = construct_general_MF(N_copies, h_x=0.3, h_z=0.1, J_xx=0.2, J_zz=0.4)
    results_mf = evolve_and_trace(H_mf, single_state_1, N_copies, t, keep_qubits=[0], verbose=False)
    rho_mf = results_mf['rho_reduced']
    r_mf = bloch_vector(rho_mf)
    
    print(f"TFIM:     Bloch vector = {r}")
    print(f"XY Model: Bloch vector = {r_xy}")
    print(f"General:  Bloch vector = {r_mf}")
    print()
    
    # Example 5: Custom single-qubit state
    print("EXAMPLE 5: Custom single-qubit initial state")
    print("-"*70)
    alpha, beta = 0.8, 0.6
    single_state_custom = np.array([alpha, beta])
    print(f"Single-qubit input: α={alpha}, β={beta}")
    print()
    
    results_custom = evolve_and_trace(H1, single_state_custom, N_copies, t, keep_qubits=[0], verbose=False)
    rho_custom = results_custom['rho_reduced']
    r_custom = bloch_vector(rho_custom)
    
    print(f"Bloch vector after evolution: {r_custom}")
    print(f"Purity: {np.linalg.norm(r_custom)**2:.4f}")
    print()
    
    print("="*70)
    print("KEY WORKFLOW")
    print("="*70)
    print("""
The complete workflow is:

1. Input: Single-qubit state [α, β] (dimension 2)
2. Create N copies: |ψ⟩^⊗N (dimension 2^N)
3. Evolve: |ψ(t)⟩ = e^{-iHt} |ψ⟩^⊗N
4. Form density matrix: ρ(t) = |ψ(t)⟩⟨ψ(t)|
5. Partial trace: Extract single-copy state back

This allows studying how mean-field interactions affect individual qubits!
    """)
    
    print("="*70)
    print("All examples completed successfully!")
    print("="*70)

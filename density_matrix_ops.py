"""
Density Matrix Operations Module

This module provides functions for:
1. Computing density matrices from state vectors: ρ = |ψ⟩⟨ψ|
2. Computing partial traces to obtain reduced density matrices
3. Various density matrix analysis tools
"""

import numpy as np
from typing import List, Union


def state_to_density_matrix(psi):
    """
    Compute the density matrix from a state vector.
    
    For a pure state |ψ⟩, the density matrix is:
    ρ = |ψ⟩⟨ψ|
    
    Parameters:
    -----------
    psi : numpy.ndarray
        State vector (1D array of complex numbers)
    
    Returns:
    --------
    rho : numpy.ndarray
        Density matrix (2D array)
    
    Example:
    --------
    >>> psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    >>> rho = state_to_density_matrix(psi)
    >>> print(rho)
    [[0.5 0.5]
     [0.5 0.5]]
    """
    psi = np.array(psi, dtype=complex)
    
    # Ensure psi is a column vector
    if psi.ndim == 1:
        psi = psi.reshape(-1, 1)
    
    # Compute |ψ⟩⟨ψ| = ψ * ψ†
    rho = psi @ psi.conj().T
    
    return rho


def partial_trace(rho, dims, trace_out):
    """
    Compute the partial trace of a density matrix.
    
    For a bipartite system A⊗B with density matrix ρ_AB:
    - Trace out B: ρ_A = Tr_B(ρ_AB)
    - Trace out A: ρ_B = Tr_A(ρ_AB)
    
    Parameters:
    -----------
    rho : numpy.ndarray
        Density matrix of the full system
    dims : list of int
        Dimensions of each subsystem. For n qubits, dims = [2, 2, ..., 2]
        Example: [2, 2] for two qubits, [2, 2, 2] for three qubits
    trace_out : int or list of int
        Index or indices of subsystems to trace out (0-indexed)
        Example: 1 means trace out subsystem 1
                [0, 2] means trace out subsystems 0 and 2
    
    Returns:
    --------
    rho_reduced : numpy.ndarray
        Reduced density matrix after tracing out specified subsystems
    
    Examples:
    ---------
    >>> # Two-qubit system, trace out second qubit
    >>> rho = state_to_density_matrix(psi)
    >>> rho_A = partial_trace(rho, dims=[2, 2], trace_out=1)
    
    >>> # Three-qubit system, trace out first and third qubits
    >>> rho_B = partial_trace(rho, dims=[2, 2, 2], trace_out=[0, 2])
    """
    rho = np.array(rho, dtype=complex)
    
    # Convert trace_out to list if it's a single integer
    if isinstance(trace_out, int):
        trace_out = [trace_out]
    
    # Validate inputs
    n_subsystems = len(dims)
    total_dim = np.prod(dims)
    
    if rho.shape != (total_dim, total_dim):
        raise ValueError(f"Density matrix shape {rho.shape} doesn't match total dimension {total_dim}")
    
    for idx in trace_out:
        if idx < 0 or idx >= n_subsystems:
            raise ValueError(f"trace_out index {idx} out of range [0, {n_subsystems-1}]")
    
    # Determine which subsystems to keep
    keep = [i for i in range(n_subsystems) if i not in trace_out]
    
    if not keep:
        raise ValueError("Cannot trace out all subsystems")
    
    # Reshape density matrix for partial trace computation
    # Shape: (dim_0, dim_1, ..., dim_{n-1}, dim_0, dim_1, ..., dim_{n-1})
    shape = dims + dims
    rho_reshaped = rho.reshape(shape)
    
    # Trace out subsystems by summing over their indices
    # We trace from the end to avoid index shifting issues
    for idx in sorted(trace_out, reverse=True):
        # Sum over the diagonal elements of the subsystem to trace out
        # Axes to sum: idx (from first set) and idx + n_subsystems (from second set)
        rho_reshaped = np.trace(rho_reshaped, axis1=idx, axis2=idx + len(dims))
        # Remove traced subsystem from dims for next iteration
        dims = dims[:idx] + dims[idx+1:]
    
    # Reshape back to 2D matrix
    reduced_dim = np.prod([dims[i] for i in range(len(dims))])
    rho_reduced = rho_reshaped.reshape(reduced_dim, reduced_dim)
    
    return rho_reduced


def partial_trace_qubit_system(rho, n_qubits, keep_qubits):
    """
    Convenience function for partial trace of qubit systems.
    
    Instead of specifying which qubits to trace out, specify which to keep.
    
    Parameters:
    -----------
    rho : numpy.ndarray
        Density matrix of n-qubit system
    n_qubits : int
        Total number of qubits
    keep_qubits : int or list of int
        Qubit index or indices to keep (0-indexed)
    
    Returns:
    --------
    rho_reduced : numpy.ndarray
        Reduced density matrix
    
    Example:
    --------
    >>> # Three qubits, keep only the middle qubit
    >>> rho_1 = partial_trace_qubit_system(rho, n_qubits=3, keep_qubits=1)
    
    >>> # Three qubits, keep qubits 0 and 2
    >>> rho_02 = partial_trace_qubit_system(rho, n_qubits=3, keep_qubits=[0, 2])
    """
    if isinstance(keep_qubits, int):
        keep_qubits = [keep_qubits]
    
    # Determine which qubits to trace out
    trace_out = [i for i in range(n_qubits) if i not in keep_qubits]
    
    # Call the general partial trace function
    dims = [2] * n_qubits
    return partial_trace(rho, dims, trace_out)


def purity(rho):
    """
    Compute the purity of a density matrix.
    
    Purity is defined as Tr(ρ²).
    - For pure states: purity = 1
    - For maximally mixed states: purity = 1/d (where d is dimension)
    
    Parameters:
    -----------
    rho : numpy.ndarray
        Density matrix
    
    Returns:
    --------
    p : float
        Purity value
    """
    return np.real(np.trace(rho @ rho))


def von_neumann_entropy(rho, base=2):
    """
    Compute the von Neumann entropy of a density matrix.
    
    S(ρ) = -Tr(ρ log ρ) = -∑_i λ_i log λ_i
    
    where λ_i are the eigenvalues of ρ.
    
    Parameters:
    -----------
    rho : numpy.ndarray
        Density matrix
    base : float
        Logarithm base (default: 2 for bits)
    
    Returns:
    --------
    S : float
        Von Neumann entropy
    """
    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(rho)
    
    # Remove negative eigenvalues (numerical errors)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    
    # Compute entropy
    if base == 2:
        S = -np.sum(eigenvalues * np.log2(eigenvalues))
    else:
        S = -np.sum(eigenvalues * np.log(eigenvalues) / np.log(base))
    
    return np.real(S)


def is_pure_state(rho, tol=1e-10):
    """
    Check if a density matrix represents a pure state.
    
    A state is pure if Tr(ρ²) = 1.
    
    Parameters:
    -----------
    rho : numpy.ndarray
        Density matrix
    tol : float
        Tolerance for purity check
    
    Returns:
    --------
    bool
        True if pure state, False otherwise
    """
    p = purity(rho)
    return np.abs(p - 1.0) < tol


def fidelity(rho1, rho2):
    """
    Compute the fidelity between two density matrices.
    
    F(ρ1, ρ2) = Tr(√(√ρ1 ρ2 √ρ1))²
    
    For pure states |ψ⟩ and |φ⟩: F = |⟨ψ|φ⟩|²
    
    Parameters:
    -----------
    rho1, rho2 : numpy.ndarray
        Density matrices
    
    Returns:
    --------
    F : float
        Fidelity (between 0 and 1)
    """
    # Compute sqrt(rho1)
    eigvals1, eigvecs1 = np.linalg.eigh(rho1)
    sqrt_rho1 = eigvecs1 @ np.diag(np.sqrt(np.maximum(eigvals1, 0))) @ eigvecs1.conj().T
    
    # Compute sqrt(rho1) @ rho2 @ sqrt(rho1)
    M = sqrt_rho1 @ rho2 @ sqrt_rho1
    
    # Compute sqrt of M
    eigvals_M, eigvecs_M = np.linalg.eigh(M)
    sqrt_M = eigvecs_M @ np.diag(np.sqrt(np.maximum(eigvals_M, 0))) @ eigvecs_M.conj().T
    
    # Fidelity is trace squared
    F = np.real(np.trace(sqrt_M))**2
    
    return F


# Examples and testing
if __name__ == "__main__":
    print("=" * 80)
    print("DENSITY MATRIX OPERATIONS - Examples")
    print("=" * 80)
    
    # Example 1: Single qubit pure state
    print("\n--- Example 1: Single qubit |+⟩ state ---")
    psi_plus = np.array([1, 1]) / np.sqrt(2)
    rho_plus = state_to_density_matrix(psi_plus)
    print(f"State: |+⟩ = (|0⟩ + |1⟩)/√2")
    print(f"Density matrix:\n{rho_plus}")
    print(f"Purity: {purity(rho_plus):.6f} (should be 1 for pure state)")
    print(f"Von Neumann entropy: {von_neumann_entropy(rho_plus):.6f} (should be 0 for pure state)")
    print(f"Is pure state: {is_pure_state(rho_plus)}")
    
    # Example 2: Two-qubit Bell state
    print("\n--- Example 2: Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 ---")
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_bell = state_to_density_matrix(bell_state)
    print(f"Full system density matrix shape: {rho_bell.shape}")
    print(f"Full system density matrix:\n{rho_bell}")
    print(f"\nPurity of full system: {purity(rho_bell):.6f}")
    
    # Partial trace: trace out qubit 1, keep qubit 0
    print("\n--- Tracing out qubit 1 (keeping qubit 0) ---")
    rho_A = partial_trace(rho_bell, dims=[2, 2], trace_out=1)
    print(f"Reduced density matrix of qubit 0:\n{rho_A}")
    print(f"Purity: {purity(rho_A):.6f} (< 1 indicates entanglement)")
    print(f"Von Neumann entropy: {von_neumann_entropy(rho_A):.6f} (> 0 indicates entanglement)")
    
    # Trace out qubit 0, keep qubit 1
    print("\n--- Tracing out qubit 0 (keeping qubit 1) ---")
    rho_B = partial_trace(rho_bell, dims=[2, 2], trace_out=0)
    print(f"Reduced density matrix of qubit 1:\n{rho_B}")
    print(f"Purity: {purity(rho_B):.6f}")
    print(f"Von Neumann entropy: {von_neumann_entropy(rho_B):.6f}")
    
    # Example 3: Three-qubit GHZ state
    print("\n--- Example 3: Three-qubit GHZ state (|000⟩ + |111⟩)/√2 ---")
    ghz_state = np.zeros(8, dtype=complex)
    ghz_state[0] = 1/np.sqrt(2)  # |000⟩
    ghz_state[7] = 1/np.sqrt(2)  # |111⟩
    rho_ghz = state_to_density_matrix(ghz_state)
    print(f"Full system density matrix shape: {rho_ghz.shape}")
    
    # Trace out qubits 1 and 2, keep qubit 0
    print("\n--- Tracing out qubits 1 and 2 (keeping qubit 0) ---")
    rho_ghz_0 = partial_trace(rho_ghz, dims=[2, 2, 2], trace_out=[1, 2])
    print(f"Reduced density matrix:\n{rho_ghz_0}")
    print(f"Purity: {purity(rho_ghz_0):.6f}")
    
    # Keep only qubit 1
    print("\n--- Using convenience function: keep only qubit 1 ---")
    rho_ghz_1 = partial_trace_qubit_system(rho_ghz, n_qubits=3, keep_qubits=1)
    print(f"Reduced density matrix:\n{rho_ghz_1}")
    print(f"Purity: {purity(rho_ghz_1):.6f}")
    
    # Keep qubits 0 and 2
    print("\n--- Keep qubits 0 and 2 (trace out qubit 1) ---")
    rho_ghz_02 = partial_trace_qubit_system(rho_ghz, n_qubits=3, keep_qubits=[0, 2])
    print(f"Reduced density matrix shape: {rho_ghz_02.shape}")
    print(f"Reduced density matrix:\n{rho_ghz_02}")
    print(f"Purity: {purity(rho_ghz_02):.6f}")
    
    # Example 4: Product state (not entangled)
    print("\n--- Example 4: Product state |0⟩⊗|+⟩ (not entangled) ---")
    product_state = np.array([1, 1, 0, 0]) / np.sqrt(2)  # |0⟩⊗|+⟩
    rho_product = state_to_density_matrix(product_state)
    
    rho_product_A = partial_trace(rho_product, dims=[2, 2], trace_out=1)
    print(f"Reduced density matrix of qubit 0:\n{rho_product_A}")
    print(f"Purity: {purity(rho_product_A):.6f} (should be 1, no entanglement)")
    print(f"Von Neumann entropy: {von_neumann_entropy(rho_product_A):.6f} (should be 0)")
    
    # Example 5: Fidelity between states
    print("\n--- Example 5: Fidelity between states ---")
    psi1 = np.array([1, 0], dtype=complex)  # |0⟩
    psi2 = np.array([0, 1], dtype=complex)  # |1⟩
    psi3 = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+⟩
    
    rho1 = state_to_density_matrix(psi1)
    rho2 = state_to_density_matrix(psi2)
    rho3 = state_to_density_matrix(psi3)
    
    print(f"Fidelity between |0⟩ and |1⟩: {fidelity(rho1, rho2):.6f} (should be 0)")
    print(f"Fidelity between |0⟩ and |+⟩: {fidelity(rho1, rho3):.6f} (should be 0.5)")
    print(f"Fidelity between |0⟩ and |0⟩: {fidelity(rho1, rho1):.6f} (should be 1)")
    
    # Example 6: Maximally mixed state
    print("\n--- Example 6: Maximally mixed state ---")
    rho_mixed = np.eye(2) / 2  # I/2 for a qubit
    print(f"Maximally mixed state:\n{rho_mixed}")
    print(f"Purity: {purity(rho_mixed):.6f} (should be 0.5 for qubit)")
    print(f"Von Neumann entropy: {von_neumann_entropy(rho_mixed):.6f} (should be 1 bit)")
    print(f"Is pure state: {is_pure_state(rho_mixed)}")
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)

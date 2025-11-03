"""
Quantum State Evolution using Matrix Exponential e^{iHt}

This module provides functions to evolve quantum states under a Hamiltonian
by computing the matrix exponential e^{iHt}. It handles both general 
diagonalizable Hamiltonians and already-diagonal Hamiltonians efficiently.


Date: 2025
"""

import numpy as np
from scipy.linalg import eig, expm
import warnings


def evolve_state(psi0, H, t, method='auto', check_diagonal_tol=1e-10):
    """
    Evolve a quantum state by the Hamiltonian H for time t.
    
    Computes: |ψ(t)⟩ = e^{-iHt} |ψ(0)⟩
    
    Parameters:
    -----------
    psi0 : array_like, shape (n,)
        Initial state vector |ψ(0)⟩ (complex or real)
        
    H : array_like, shape (n, n)
        Hamiltonian matrix (should be Hermitian)
        
    t : float
        Evolution time
        
    method : str, optional
        Method to use for computing e^{-iHt}:
        - 'auto' : Automatically detect if H is diagonal (default)
        - 'diagonal' : Assume H is diagonal (fastest)
        - 'diagonalize' : Diagonalize H first (exact for diagonalizable H)
        - 'expm' : Use scipy's matrix exponential (general, slower)
        
    check_diagonal_tol : float, optional
        Tolerance for checking if matrix is diagonal (default: 1e-10)
        Only used when method='auto'
    
    Returns:
    --------
    psi_t : ndarray, shape (n,)
        Evolved state |ψ(t)⟩ = e^{-iHt} |ψ(0)⟩
        
    info : dict
        Dictionary containing:
        - 'method_used': The actual method used
        - 'is_diagonal': Whether H was detected as diagonal
        - 'eigenvalues': Eigenvalues (if diagonalization was used)
        - 'norm_initial': ||ψ(0)||
        - 'norm_final': ||ψ(t)||
    
    Examples:
    ---------
    >>> # Example 1: Diagonal Hamiltonian
    >>> H = np.diag([1.0, -1.0])
    >>> psi0 = np.array([1, 1]) / np.sqrt(2)
    >>> psi_t, info = evolve_state(psi0, H, t=1.0)
    
    >>> # Example 2: General Hamiltonian (will be diagonalized)
    >>> H = np.array([[0, 1], [1, 0]])  # Pauli X
    >>> psi0 = np.array([1, 0])
    >>> psi_t, info = evolve_state(psi0, H, t=np.pi/2)
    
    >>> # Example 3: Force specific method
    >>> psi_t, info = evolve_state(psi0, H, t=1.0, method='diagonalize')
    
    Notes:
    ------
    - The Hamiltonian H should be Hermitian for physical evolution
    - Unitarity is preserved: ||ψ(t)|| = ||ψ(0)||
    - Time evolution uses the convention: e^{-iHt} (not e^{iHt})
    """
    
    # Convert to numpy arrays
    psi0 = np.asarray(psi0, dtype=complex)
    H = np.asarray(H, dtype=complex)
    
    # Validate inputs
    if psi0.ndim != 1:
        raise ValueError(f"psi0 must be a 1D array, got shape {psi0.shape}")
    
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square matrix, got shape {H.shape}")
    
    if psi0.shape[0] != H.shape[0]:
        raise ValueError(f"Dimension mismatch: psi0 has length {psi0.shape[0]}, "
                        f"H has shape {H.shape}")
    
    # Check if H is Hermitian (warning only)
    if not np.allclose(H, H.conj().T, atol=1e-10):
        warnings.warn("Hamiltonian is not Hermitian. Evolution may not be unitary.",
                     UserWarning)
    
    # Store initial norm
    norm_initial = np.linalg.norm(psi0)
    
    # Determine method to use
    is_diagonal = False
    if method == 'auto':
        # Check if H is diagonal
        is_diagonal = _is_diagonal(H, tol=check_diagonal_tol)
        method_used = 'diagonal' if is_diagonal else 'diagonalize'
    else:
        method_used = method
        if method == 'diagonal':
            is_diagonal = True
    
    # Compute evolution based on method
    eigenvalues = None
    
    if method_used == 'diagonal':
        # H is diagonal: e^{-iHt} is just e^{-iλ_i t} on diagonal
        psi_t = _evolve_diagonal(psi0, H, t)
        eigenvalues = np.diag(H)
        
    elif method_used == 'diagonalize':
        # Diagonalize H and use e^{-iHt} = U e^{-iDt} U†
        psi_t, eigenvalues = _evolve_via_diagonalization(psi0, H, t)
        
    elif method_used == 'expm':
        # Use scipy's matrix exponential directly
        psi_t = _evolve_via_expm(psi0, H, t)
        
    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Choose from 'auto', 'diagonal', 'diagonalize', 'expm'")
    
    # Store final norm
    norm_final = np.linalg.norm(psi_t)
    
    # Prepare info dictionary
    info = {
        'method_used': method_used,
        'is_diagonal': is_diagonal,
        'eigenvalues': eigenvalues,
        'norm_initial': norm_initial,
        'norm_final': norm_final,
        'unitarity_error': abs(norm_final - norm_initial)
    }
    
    return psi_t, info


def _is_diagonal(H, tol=1e-10):
    """Check if matrix H is diagonal within tolerance."""
    # Get off-diagonal elements
    off_diag = H - np.diag(np.diag(H))
    return np.allclose(off_diag, 0, atol=tol)


def _evolve_diagonal(psi0, H, t):
    """
    Evolve state when H is diagonal.
    
    For diagonal H = diag(λ₁, λ₂, ..., λₙ):
    e^{-iHt} = diag(e^{-iλ₁t}, e^{-iλ₂t}, ..., e^{-iλₙt})
    
    So: (e^{-iHt}ψ)ᵢ = e^{-iλᵢt} ψᵢ
    """
    eigenvalues = np.diag(H)
    phases = np.exp(-1j * eigenvalues * t)
    return phases * psi0


def _evolve_via_diagonalization(psi0, H, t):
    """
    Evolve state by diagonalizing H.
    
    H = U D U†
    e^{-iHt} = U e^{-iDt} U†
    
    Returns:
    --------
    psi_t : evolved state
    eigenvalues : eigenvalues of H
    """
    # Diagonalize H
    eigenvalues, U = eig(H)
    
    # eigenvalues is the array of eigenvalues
    # U is the matrix of eigenvectors (columns)
    
    # Check that eigenvalues are real (for Hermitian matrices)
    if not np.allclose(eigenvalues.imag, 0, atol=1e-10):
        warnings.warn("Eigenvalues have imaginary parts. "
                     "H may not be Hermitian.", UserWarning)
    
    # Compute e^{-iDt} where D = diag(eigenvalues)
    exp_diag = np.exp(-1j * eigenvalues * t)
    
    # Method 1: Direct formula
    # |ψ(t)⟩ = U @ diag(e^{-iλt}) @ U† @ |ψ(0)⟩
    psi_t = U @ (exp_diag * (U.conj().T @ psi0))
    
    return psi_t, eigenvalues


def _evolve_via_expm(psi0, H, t):
    """
    Evolve state using scipy's matrix exponential.
    
    Directly computes: |ψ(t)⟩ = e^{-iHt} |ψ(0)⟩
    """
    U = expm(-1j * H * t)
    return U @ psi0


def check_hermiticity(H, tol=1e-10):
    """
    Check if a matrix is Hermitian.
    
    Parameters:
    -----------
    H : array_like
        Matrix to check
    tol : float
        Tolerance for comparison
    
    Returns:
    --------
    is_hermitian : bool
        True if H = H†
    max_error : float
        Maximum deviation from Hermiticity
    """
    H = np.asarray(H, dtype=complex)
    diff = H - H.conj().T
    max_error = np.max(np.abs(diff))
    is_hermitian = max_error < tol
    return is_hermitian, max_error


def verify_unitarity(psi0, psi_t, tol=1e-10):
    """
    Verify that evolution preserves norm (unitarity).
    
    Parameters:
    -----------
    psi0 : array_like
        Initial state
    psi_t : array_like
        Final state
    tol : float
        Tolerance
    
    Returns:
    --------
    is_unitary : bool
        True if ||ψ(t)|| ≈ ||ψ(0)||
    error : float
        |  ||ψ(t)|| - ||ψ(0)|| |
    """
    norm0 = np.linalg.norm(psi0)
    normt = np.linalg.norm(psi_t)
    error = abs(normt - norm0)
    is_unitary = error < tol
    return is_unitary, error


# ============================================================================
# DEMONSTRATION AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("QUANTUM STATE EVOLUTION: e^{-iHt} Demonstration")
    print("="*70)
    print()
    
    # Example 1: Diagonal Hamiltonian
    print("EXAMPLE 1: Diagonal Hamiltonian")
    print("-"*70)
    H_diag = np.diag([1.0, -1.0])
    psi0 = np.array([1, 1]) / np.sqrt(2)
    t = 1.0
    
    print(f"H = \n{H_diag}")
    print(f"|ψ(0)⟩ = {psi0}")
    print(f"t = {t}")
    print()
    
    psi_t, info = evolve_state(psi0, H_diag, t)
    
    print(f"Method used: {info['method_used']}")
    print(f"Is diagonal: {info['is_diagonal']}")
    print(f"Eigenvalues: {info['eigenvalues']}")
    print(f"|ψ(t)⟩ = {psi_t}")
    print(f"||ψ(0)|| = {info['norm_initial']:.6f}")
    print(f"||ψ(t)|| = {info['norm_final']:.6f}")
    print(f"Unitarity error: {info['unitarity_error']:.2e}")
    print()
    
    # Example 2: Non-diagonal Hamiltonian (Pauli X)
    print("EXAMPLE 2: Non-diagonal Hamiltonian (Pauli X)")
    print("-"*70)
    H_x = np.array([[0, 1], [1, 0]], dtype=complex)
    psi0 = np.array([1, 0], dtype=complex)
    t = np.pi / 2
    
    print(f"H = \n{H_x}")
    print(f"|ψ(0)⟩ = {psi0}")
    print(f"t = π/2 = {t:.4f}")
    print()
    
    psi_t, info = evolve_state(psi0, H_x, t)
    
    print(f"Method used: {info['method_used']}")
    print(f"Is diagonal: {info['is_diagonal']}")
    print(f"Eigenvalues: {info['eigenvalues']}")
    print(f"|ψ(t)⟩ = {psi_t}")
    print(f"Unitarity error: {info['unitarity_error']:.2e}")
    print()
    
    # Example 3: 2×2 Mean-field Hamiltonian
    print("EXAMPLE 3: Mean-Field Hamiltonian H = h_x X + h_z Z")
    print("-"*70)
    h_x = 0.5
    h_z = 1.0
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H_mf = h_x * X + h_z * Z
    
    psi0 = np.array([1, 1]) / np.sqrt(2)
    t = 2.0
    
    print(f"h_x = {h_x}, h_z = {h_z}")
    print(f"H = h_x*X + h_z*Z = \n{H_mf}")
    print(f"|ψ(0)⟩ = {psi0}")
    print(f"t = {t}")
    print()
    
    psi_t, info = evolve_state(psi0, H_mf, t)
    
    print(f"Method used: {info['method_used']}")
    print(f"Eigenvalues: {info['eigenvalues']}")
    print(f"|ψ(t)⟩ = {psi_t}")
    print()
    
    # Compare methods
    print("EXAMPLE 4: Comparing Methods")
    print("-"*70)
    methods = ['diagonalize', 'expm']
    
    H = H_mf
    psi0 = np.array([1, 0], dtype=complex)
    t = 1.0
    
    print(f"H = \n{H}")
    print(f"|ψ(0)⟩ = {psi0}")
    print(f"t = {t}")
    print()
    
    results = {}
    for method in methods:
        psi_t, info = evolve_state(psi0, H, t, method=method)
        results[method] = psi_t
        print(f"Method '{method}':")
        print(f"  |ψ(t)⟩ = {psi_t}")
    
    print()
    print("Difference between methods:")
    diff = np.linalg.norm(results['diagonalize'] - results['expm'])
    print(f"  ||ψ_diag - ψ_expm|| = {diff:.2e}")
    print()
    
    # Example 5: Large system
    print("EXAMPLE 5: Larger System (4×4)")
    print("-"*70)
    np.random.seed(42)
    n = 4
    # Create a random Hermitian matrix
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H_large = (A + A.conj().T) / 2  # Make Hermitian
    
    psi0 = np.random.randn(n) + 1j * np.random.randn(n)
    psi0 = psi0 / np.linalg.norm(psi0)
    t = 0.5
    
    print(f"Dimension: {n}×{n}")
    print(f"t = {t}")
    print()
    
    psi_t, info = evolve_state(psi0, H_large, t)
    
    print(f"Method used: {info['method_used']}")
    print(f"Eigenvalues: {info['eigenvalues']}")
    print(f"Unitarity error: {info['unitarity_error']:.2e}")
    print()
    
    print("="*70)
    print("All examples completed successfully!")
    print("="*70)

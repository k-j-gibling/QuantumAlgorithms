import numpy as np
from functools import reduce
import re

# Define Pauli matrices and identity
PAULI_MATRICES = {
    'I': np.array([[1, 0], 
                   [0, 1]], dtype=complex),
    'X': np.array([[0, 1], 
                   [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], 
                   [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], 
                   [0, -1]], dtype=complex)
}


def tensor_product(*matrices):
    """
    Compute the tensor product (Kronecker product) of multiple matrices.
    
    Parameters:
    -----------
    *matrices : numpy.ndarray
        Variable number of matrices to compute tensor product
    
    Returns:
    --------
    result : numpy.ndarray
        Tensor product of all input matrices
    """
    return reduce(np.kron, matrices)


def parse_pauli_string(pauli_str):
    """
    Parse a string of Pauli operators and return the corresponding matrix.
    
    Examples:
    ---------
    'X' -> X matrix
    'X⊗Z' or 'X*Z' or 'XZ' -> X ⊗ Z
    'X⊗I⊗Z' -> X ⊗ I ⊗ Z
    
    Parameters:
    -----------
    pauli_str : str
        String representing tensor product of Pauli matrices
        Can use ⊗, *, or just concatenation as tensor product symbol
    
    Returns:
    --------
    matrix : numpy.ndarray
        The resulting matrix from the tensor product
    """
    # Remove whitespace
    pauli_str = pauli_str.strip()
    
    # Split by tensor product symbols (⊗, *, or just parse character by character)
    # First, try splitting by explicit symbols
    if '⊗' in pauli_str:
        operators = pauli_str.split('⊗')
    elif '*' in pauli_str:
        operators = pauli_str.split('*')
    else:
        # Parse character by character
        operators = list(pauli_str)
    
    # Clean up operators (remove whitespace)
    operators = [op.strip() for op in operators if op.strip()]
    
    # Convert to matrices
    matrices = []
    for op in operators:
        if op not in PAULI_MATRICES:
            raise ValueError(f"Unknown Pauli operator: '{op}'. Must be one of I, X, Y, Z")
        matrices.append(PAULI_MATRICES[op])
    
    if not matrices:
        raise ValueError("No valid Pauli operators found in string")
    
    # Compute tensor product
    return tensor_product(*matrices)


def parse_coefficient(coef_str):
    """
    Parse a coefficient string (can be integer, float, or complex).
    
    Examples: '2', '1.5', '2j', '1+2j', '-3.5'
    """
    coef_str = coef_str.strip()
    if not coef_str or coef_str == '+':
        return 1.0
    if coef_str == '-':
        return -1.0
    
    try:
        # Try to evaluate as a number (handles int, float, complex)
        return complex(coef_str)
    except ValueError:
        raise ValueError(f"Invalid coefficient: '{coef_str}'")


def hamiltonian_from_string(hamiltonian_str):
    """
    Parse a Hamiltonian string and return the corresponding matrix.
    
    The Hamiltonian should be specified as a sum of tensor products of Pauli matrices.
    
    Examples:
    ---------
    'X⊗Z' or 'X*Z' or 'XZ'
    'X⊗I⊗Z + Z⊗X⊗I'
    '0.5*X⊗Z + 0.5*Z⊗X'
    '2*XII + 3*IXI + 4*IIX'
    'Z⊗I⊗X⊗Z + X⊗X⊗I⊗I'
    
    Parameters:
    -----------
    hamiltonian_str : str
        String representation of the Hamiltonian
    
    Returns:
    --------
    H : numpy.ndarray
        The Hamiltonian matrix
    """
    # Remove all whitespace for easier parsing
    hamiltonian_str = hamiltonian_str.replace(' ', '')
    
    # Split by + and - while keeping the signs
    # Use regex to split but keep delimiters
    terms = re.split(r'(?=[+-])', hamiltonian_str)
    terms = [t for t in terms if t]  # Remove empty strings
    
    H = None
    
    for term in terms:
        term = term.strip()
        if not term:
            continue
        
        # Check if term starts with + or -
        sign = 1.0
        if term[0] == '+':
            term = term[1:]
            sign = 1.0
        elif term[0] == '-':
            term = term[1:]
            sign = -1.0
        
        # Split coefficient from Pauli string
        # Look for * that separates coefficient from operators
        if '*' in term:
            # Find the last * that's part of coefficient*operators pattern
            # We need to be careful: 2*X*Z means coefficient is 2 and operators are X*Z
            parts = term.split('*')
            
            # Try to parse the first part as a coefficient
            try:
                coef = parse_coefficient(parts[0])
                pauli_part = '*'.join(parts[1:])
            except ValueError:
                # If first part is not a number, treat the whole thing as Pauli operators
                coef = 1.0
                pauli_part = term
        else:
            # No explicit coefficient
            coef = 1.0
            pauli_part = term
        
        # Apply sign to coefficient
        coef *= sign
        
        # Parse the Pauli string
        term_matrix = parse_pauli_string(pauli_part)
        
        # Add to Hamiltonian
        if H is None:
            H = coef * term_matrix
        else:
            if H.shape != term_matrix.shape:
                raise ValueError(f"Incompatible term dimensions: {H.shape} vs {term_matrix.shape}")
            H = H + coef * term_matrix
    
    if H is None:
        raise ValueError("No valid terms found in Hamiltonian string")
    
    return H


class HamiltonianBuilder:
    """
    A class for building Hamiltonians from Pauli operators with a fluent interface.
    
    Example:
    --------
    H = HamiltonianBuilder()
    H.add_term(['X', 'I', 'Z'], coefficient=0.5)
    H.add_term(['Z', 'X', 'I'], coefficient=0.5)
    matrix = H.build()
    """
    
    def __init__(self):
        self.terms = []
    
    def add_term(self, operators, coefficient=1.0):
        """
        Add a term to the Hamiltonian.
        
        Parameters:
        -----------
        operators : list of str
            List of Pauli operators (e.g., ['X', 'I', 'Z'])
        coefficient : float or complex
            Coefficient for this term
        
        Returns:
        --------
        self : HamiltonianBuilder
            Returns self for method chaining
        """
        self.terms.append((operators, coefficient))
        return self
    
    def build(self):
        """
        Build and return the Hamiltonian matrix.
        
        Returns:
        --------
        H : numpy.ndarray
            The Hamiltonian matrix
        """
        if not self.terms:
            raise ValueError("No terms added to Hamiltonian")
        
        H = None
        for operators, coefficient in self.terms:
            # Convert operators to matrices
            matrices = [PAULI_MATRICES[op] for op in operators]
            term_matrix = tensor_product(*matrices)
            
            if H is None:
                H = coefficient * term_matrix
            else:
                H = H + coefficient * term_matrix
        
        return H
    
    def __str__(self):
        """String representation of the Hamiltonian."""
        terms_str = []
        for operators, coef in self.terms:
            op_str = '⊗'.join(operators)
            if coef == 1.0:
                terms_str.append(op_str)
            elif coef == -1.0:
                terms_str.append(f"-{op_str}")
            else:
                terms_str.append(f"{coef}*{op_str}")
        return " + ".join(terms_str)


# Convenience functions for common Hamiltonians
def heisenberg_hamiltonian(n_sites, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Create a Heisenberg Hamiltonian for a 1D chain.
    H = sum_i [Jx*X_i*X_{i+1} + Jy*Y_i*Y_{i+1} + Jz*Z_i*Z_{i+1}]
    
    Parameters:
    -----------
    n_sites : int
        Number of sites in the chain
    Jx, Jy, Jz : float
        Coupling constants for X, Y, Z interactions
    
    Returns:
    --------
    H : numpy.ndarray
        The Hamiltonian matrix
    """
    builder = HamiltonianBuilder()
    
    for i in range(n_sites - 1):
        # X_i X_{i+1}
        ops_x = ['I'] * n_sites
        ops_x[i] = 'X'
        ops_x[i+1] = 'X'
        builder.add_term(ops_x, Jx)
        
        # Y_i Y_{i+1}
        ops_y = ['I'] * n_sites
        ops_y[i] = 'Y'
        ops_y[i+1] = 'Y'
        builder.add_term(ops_y, Jy)
        
        # Z_i Z_{i+1}
        ops_z = ['I'] * n_sites
        ops_z[i] = 'Z'
        ops_z[i+1] = 'Z'
        builder.add_term(ops_z, Jz)
    
    return builder.build()


def ising_hamiltonian(n_sites, J=1.0, h=0.0):
    """
    Create a transverse field Ising Hamiltonian.
    H = -J * sum_i Z_i*Z_{i+1} - h * sum_i X_i
    
    Parameters:
    -----------
    n_sites : int
        Number of sites
    J : float
        Coupling constant
    h : float
        Transverse field strength
    
    Returns:
    --------
    H : numpy.ndarray
        The Hamiltonian matrix
    """
    builder = HamiltonianBuilder()
    
    # ZZ interaction
    for i in range(n_sites - 1):
        ops = ['I'] * n_sites
        ops[i] = 'Z'
        ops[i+1] = 'Z'
        builder.add_term(ops, -J)
    
    # Transverse field
    for i in range(n_sites):
        ops = ['I'] * n_sites
        ops[i] = 'X'
        builder.add_term(ops, -h)
    
    return builder.build()


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("HAMILTONIAN BUILDER - Examples and Tests")
    print("=" * 70)
    
    # Example 1: Simple string parsing
    print("\nExample 1: Simple two-qubit Hamiltonian")
    print("-" * 70)
    H1_str = "X⊗Z"
    H1 = hamiltonian_from_string(H1_str)
    print(f"H = {H1_str}")
    print(f"Matrix:\n{H1}\n")
    
    # Example 2: Sum of terms
    print("Example 2: Sum of tensor products")
    print("-" * 70)
    H2_str = "X⊗I + I⊗X"
    H2 = hamiltonian_from_string(H2_str)
    print(f"H = {H2_str}")
    print(f"Matrix:\n{H2}\n")
    
    # Example 3: With coefficients
    print("Example 3: Hamiltonian with coefficients")
    print("-" * 70)
    H3_str = "0.5*X⊗Z + 0.3*Z⊗X + 0.2*Y⊗Y"
    H3 = hamiltonian_from_string(H3_str)
    print(f"H = {H3_str}")
    print(f"Matrix:\n{H3}\n")
    
    # Example 4: Four-qubit system as requested
    print("Example 4: Four-qubit Hamiltonian (Z⊗I⊗X⊗Z + ...)")
    print("-" * 70)
    H4_str = "Z⊗I⊗X⊗Z + X⊗X⊗I⊗I + 0.5*I⊗Y⊗Y⊗I"
    H4 = hamiltonian_from_string(H4_str)
    print(f"H = {H4_str}")
    print(f"Matrix shape: {H4.shape}")
    print(f"Matrix (first 8x8 block):\n{H4[:8, :8]}\n")
    
    # Example 5: Alternative notation (concatenation)
    print("Example 5: Compact notation (no tensor symbols)")
    print("-" * 70)
    H5_str = "XII + IXI + IIX"
    H5 = hamiltonian_from_string(H5_str)
    print(f"H = {H5_str}")
    print(f"Matrix:\n{H5}\n")
    
    # Example 6: Using HamiltonianBuilder class
    print("Example 6: Using HamiltonianBuilder class")
    print("-" * 70)
    builder = HamiltonianBuilder()
    builder.add_term(['X', 'I', 'Z'], coefficient=1.0)
    builder.add_term(['Z', 'X', 'I'], coefficient=1.0)
    builder.add_term(['I', 'Z', 'X'], coefficient=1.0)
    H6 = builder.build()
    print(f"H = {builder}")
    print(f"Matrix:\n{H6}\n")
    
    # Example 7: Heisenberg model
    print("Example 7: 3-site Heisenberg Hamiltonian")
    print("-" * 70)
    H7 = heisenberg_hamiltonian(3, Jx=1.0, Jy=1.0, Jz=1.0)
    print("H = Heisenberg(3 sites, Jx=Jy=Jz=1)")
    print(f"Matrix shape: {H7.shape}")
    print(f"Matrix:\n{H7}\n")
    
    # Example 8: Transverse field Ising model
    print("Example 8: 3-site Transverse Field Ising Hamiltonian")
    print("-" * 70)
    H8 = ising_hamiltonian(3, J=1.0, h=0.5)
    print("H = Ising(3 sites, J=1.0, h=0.5)")
    print(f"Matrix shape: {H8.shape}")
    print(f"Matrix:\n{H8}\n")
    
    # Example 9: Verify Hermiticity
    print("Example 9: Verify that Hamiltonians are Hermitian")
    print("-" * 70)
    for i, H in enumerate([H1, H2, H3, H4, H5, H6, H7, H8], 1):
        is_hermitian = np.allclose(H, H.conj().T)
        print(f"H{i} is Hermitian: {is_hermitian}")
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

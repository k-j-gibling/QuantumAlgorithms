import numpy as np
from scipy.linalg import eig, expm
from itertools import combinations
import warnings


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


def prepare_H_test1(N):
    """
    Blue Hamiltonian (2-body, FIXED - no double counting):
    H_N = (1/(N-1)) * Σ_{i≠j} (X_i ⊗ Z_j + Z_i ⊗ X_j)
    
    IMPORTANT: The sum Σ_{i≠j} means sum over all ORDERED pairs (i,j) with i≠j.
    To avoid double-counting, we only loop over i < j and add both terms.
    
    N: Number of copies (qubits)
    Returns: H_list - list of terms in the Hamiltonian
    """
    H_list = []
    
    # Only loop over unordered pairs (i < j) to avoid double counting
    for i in range(N):
        for j in range(i+1, N):
            # Term 1: X_i ⊗ Z_j
            term1 = ['I'] * N
            term1[i] = 'X'
            term1[j] = 'Z'
            
            # Term 2: Z_i ⊗ X_j
            term2 = ['I'] * N
            term2[i] = 'Z'
            term2[j] = 'X'
            
            H_list.append(term1)
            H_list.append(term2)
    
    return H_list


def prepare_H_test2(N):
    """
    Red Hamiltonian (3-body, FIXED - no double counting):
    H_N = (1/((N-1)(N-2))) * Σ_{i≠j≠k} (X_i⊗Z_j⊗Z_k + Z_i⊗X_j⊗Z_k + Z_i⊗Z_j⊗X_k)
    
    IMPORTANT: The sum Σ_{i≠j≠k} means sum over all ORDERED triples (i,j,k).
    To avoid overcounting, we only loop over i < j < k and add all three terms.
    
    N: Number of copies (qubits)
    Returns: H_list - list of terms in the Hamiltonian
    """
    H_list = []
    
    # Only loop over unordered triples (i < j < k) to avoid overcounting
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                # Term 1: X_i ⊗ Z_j ⊗ Z_k
                term1 = ['I'] * N
                term1[i] = 'X'
                term1[j] = 'Z'
                term1[k] = 'Z'
                
                # Term 2: Z_i ⊗ X_j ⊗ Z_k
                term2 = ['I'] * N
                term2[i] = 'Z'
                term2[j] = 'X'
                term2[k] = 'Z'
                
                # Term 3: Z_i ⊗ Z_j ⊗ X_k
                term3 = ['I'] * N
                term3[i] = 'Z'
                term3[j] = 'Z'
                term3[k] = 'X'
                
                H_list.append(term1)
                H_list.append(term2)
                H_list.append(term3)
    
    return H_list


def prepare_H_test3(N):
    """
    Alternative Hamiltonian with X and Y (FIXED - no double counting):
    H_N = (1/(N-1)) * Σ_{i≠j} (X_i ⊗ Y_j + Y_i ⊗ X_j)
    
    N: Number of copies (qubits)
    Returns: H_list - list of terms in the Hamiltonian
    """
    H_list = []
    
    # Only loop over unordered pairs (i < j) to avoid double counting
    for i in range(N):
        for j in range(i+1, N):
            # Term 1: X_i ⊗ Y_j
            term1 = ['I'] * N
            term1[i] = 'X'
            term1[j] = 'Y'
            
            # Term 2: Y_i ⊗ X_j
            term2 = ['I'] * N
            term2[i] = 'Y'
            term2[j] = 'X'
            
            H_list.append(term1)
            H_list.append(term2)
    
    return H_list


# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_no_duplicates(H_list):
    """Check if H_list contains duplicate terms"""
    terms_as_strings = [''.join(term) for term in H_list]
    unique_terms = set(terms_as_strings)
    
    has_duplicates = len(terms_as_strings) != len(unique_terms)
    
    if has_duplicates:
        from collections import Counter
        term_counts = Counter(terms_as_strings)
        print("WARNING: Duplicate terms found!")
        for term, count in term_counts.items():
            if count > 1:
                print(f"  {term}: appears {count} times")
    else:
        print("✓ No duplicate terms found")
    
    return not has_duplicates


def count_expected_terms(N, k_body):
    """
    Calculate expected number of terms for a k-body Hamiltonian.
    
    For k-body: We have C(N,k) ways to choose k qubits.
    Each choice gives k terms (one X, rest Z).
    So total is k * C(N,k)
    """
    from math import comb
    return k_body * comb(N, k_body)


if __name__ == "__main__":
    print("="*70)
    print("HAMILTONIAN VERIFICATION")
    print("="*70)
    
    # Test Blue Hamiltonian
    N = 5
    print(f"\nTesting Blue Hamiltonian (2-body) with N={N}")
    print("-"*70)
    H_blue = prepare_H_test1(N)
    print(f"Number of terms: {len(H_blue)}")
    print(f"Expected: {count_expected_terms(N, 2)}")
    verify_no_duplicates(H_blue)
    
    # Test Red Hamiltonian
    print(f"\nTesting Red Hamiltonian (3-body) with N={N}")
    print("-"*70)
    H_red = prepare_H_test2(N)
    print(f"Number of terms: {len(H_red)}")
    print(f"Expected: {count_expected_terms(N, 3)}")
    verify_no_duplicates(H_red)
    
    # Show a few example terms
    print(f"\nExample terms from Blue Hamiltonian:")
    for i, term in enumerate(H_blue[:6]):
        print(f"  Term {i}: {''.join(term)}")
    
    print(f"\nExample terms from Red Hamiltonian:")
    for i, term in enumerate(H_red[:6]):
        print(f"  Term {i}: {''.join(term)}")
    
    print("\n" + "="*70)
    print("All tests complete!")
    print("="*70)

'''
Placing everything into one workflow...
This will be a functional program; that is, consisting of functions.

INPUTS
- The input state.
- The Hamiltonian.
- The The number of copies, N.
- The time t.

PROCESS
- Take N copies of the pure state.
- Evolve the pure state.
- Take the partial trace of the N-body system.
- 
'''



import numpy as np
from scipy.linalg import eig, expm
from itertools import combinations
import warnings
from sympy import Matrix #Used for testing if a matrix is diagonal.
from qiskit.quantum_info import partial_trace, DensityMatrix


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



"""
=====================
=====================
RUNGE KUTTA METHODS
=====================
=====================
"""

def runge_kutta_4(psi0, H, t, dt=0.001):
    """
    4th order Runge-Kutta method for solving the Schrödinger equation:
    i * d|psi>/dt = H|psi>
    
    Parameters:
    -----------
    psi0 : numpy.ndarray
        Initial state vector at t=0 (complex vector)
    H : numpy.ndarray
        Hamiltonian matrix (hermitian matrix)
    t : float
        Final time at which to compute the evolved state
    dt : float, optional
        Time step for integration (default: 0.001)
        Smaller dt gives more accurate results but takes longer
    
    Returns:
    --------
    psi : numpy.ndarray
        Evolved state vector at time t
    """
    print(H.shape)
    
    # Ensure inputs are numpy arrays with complex dtype
    psi = np.array(psi0, dtype=complex)
    H = np.array(H, dtype=complex)
    
    # Number of time steps
    n_steps = int(t / dt)
    actual_dt = t / n_steps  # Adjust dt to hit exactly time t
    
    # Derivative function: d|psi>/dt = -i * H|psi>
    def dpsi_dt(psi_current):
        return -1j * H @ psi_current
    
    # Runge-Kutta 4th order integration
    for step in range(n_steps):
        k1 = dpsi_dt(psi)
        k2 = dpsi_dt(psi + 0.5 * actual_dt * k1)
        k3 = dpsi_dt(psi + 0.5 * actual_dt * k2)
        k4 = dpsi_dt(psi + actual_dt * k3)
        
        psi = psi + (actual_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return psi


def runge_kutta_adaptive(psi0, H, t, tol=1e-8, dt_init=0.01, dt_min=1e-10, dt_max=0.1):
    """
    Adaptive step-size Runge-Kutta method for the Schrödinger equation.
    
    Parameters:
    -----------
    psi0 : numpy.ndarray
        Initial state vector at t=0
    H : numpy.ndarray
        Hamiltonian matrix
    t : float
        Final time
    tol : float, optional
        Error tolerance for adaptive step size (default: 1e-8)
    dt_init : float, optional
        Initial time step (default: 0.01)
    dt_min : float, optional
        Minimum allowed time step (default: 1e-10)
    dt_max : float, optional
        Maximum allowed time step (default: 0.1)
    
    Returns:
    --------
    psi : numpy.ndarray
        Evolved state vector at time t
    """
    
    psi = np.array(psi0, dtype=complex)
    H = np.array(H, dtype=complex)
    
    t_current = 0.0
    dt = dt_init
    
    def dpsi_dt(psi_current):
        return -1j * H @ psi_current
    
    while t_current < t:
        # Adjust dt if we would overshoot
        if t_current + dt > t:
            dt = t - t_current
        
        # RK4 step with current dt
        k1 = dpsi_dt(psi)
        k2 = dpsi_dt(psi + 0.5 * dt * k1)
        k3 = dpsi_dt(psi + 0.5 * dt * k2)
        k4 = dpsi_dt(psi + dt * k3)
        psi_new = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # RK4 step with half dt for error estimation
        k1_half = dpsi_dt(psi)
        k2_half = dpsi_dt(psi + 0.25 * dt * k1_half)
        k3_half = dpsi_dt(psi + 0.25 * dt * k2_half)
        k4_half = dpsi_dt(psi + 0.5 * dt * k3_half)
        psi_half = psi + (dt / 12.0) * (k1_half + 2*k2_half + 2*k3_half + k4_half)
        
        k1_half2 = dpsi_dt(psi_half)
        k2_half2 = dpsi_dt(psi_half + 0.25 * dt * k1_half2)
        k3_half2 = dpsi_dt(psi_half + 0.25 * dt * k2_half2)
        k4_half2 = dpsi_dt(psi_half + 0.5 * dt * k3_half2)
        psi_half2 = psi_half + (dt / 12.0) * (k1_half2 + 2*k2_half2 + 2*k3_half2 + k4_half2)
        
        # Estimate error
        error = np.linalg.norm(psi_new - psi_half2)
        
        if error < tol or dt <= dt_min:
            # Accept step
            psi = psi_new
            t_current += dt
            
            # Adjust step size for next iteration
            if error > 0:
                dt = dt * min(2.0, 0.9 * (tol / error) ** 0.2)
            else:
                dt = dt * 2.0
            dt = min(dt, dt_max)
        else:
            # Reject step and reduce dt
            dt = dt * max(0.5, 0.9 * (tol / error) ** 0.2)
            dt = max(dt, dt_min)
    
    return psi



"""
==============================
==============================
Expectation Value Computation
==============================
==============================
'''

'''
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
==================================
==================================
Effective Hamiltonian Computation
==================================
==================================
'''

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
			expectationValue = expectation_value(pauli_dict[product_term], stateVector)
			product_chain = product_chain*expectationValue


		currentMatrix = None

		currentMatrix = pauli_dict[operator]

		"""if operator == 'X':
			currentMatrix = X
		elif currentMatrix == 'Y':
			currentMatrix = Y
		elif currentMatrix == 'I':
			currentMatrix = I
		else:
			currentMatrix = Z"""

		currentMatrix = currentMatrix*operator_coeff*product_chain
		matrices.append(currentMatrix)


	H_effective = matrices[0]

	for i in range(1,len(matrices)):
		H_effective += matrices[i]

	return H_effective





def create_N_copy_state(single_qubit_state, N):
	productVector = single_qubit_state

	for i in range(N-1):
		productVector = np.kron(productVector, single_qubit_state)

	return productVector

#def evolve_state(psi_0, H, t):
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
    """



'''
============================
============================
Density Matrix Computation
============================
============================
'''
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


# ============================================================================
# PART 4: PARTIAL TRACE (THE KEY OPERATION!)
# ============================================================================

"""
def partial_trace(rho, N, keep_qubits):

	return partial_trace(rho, keep_qubits)
"""




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



def _is_diagonal(H, tol=1e-10):
    """Check if matrix H is diagonal within tolerance."""
    # Get off-diagonal elements
    off_diag = H - np.diag(np.diag(H))
    return np.allclose(off_diag, 0, atol=tol)

#Function that determines whether or not a given matrix is diagonalizable.
#def _is_diagonalizable(H):



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
    
    #return psi_t, eigenvalues
    return psi_t

def _evolve_via_expm(psi0, H, t):
    """
    Evolve state using scipy's matrix exponential.
    
    Directly computes: |ψ(t)⟩ = e^{-iHt} |ψ(0)⟩
    """
    U = expm(-1j * H * t)
    return U @ psi0

def _evolve_via_diagonal_or_diagonalizable(psi0,H, t, diagonal):
	"""
	Evolve a state whereby H is either diagonal or diagonalizable.
	
	diagonal: a boolean variable.

	"""

	if diagonal:
		return _evolve_diagonal(psi0, H, t)

	return _evolve_via_diagonalization(psi0, H, t)


	#TODO: Check if diagonal.

	#TODO: Elif check if diagonalizable.

	#TODO: Else if neither diagonal nor diagonalizable then raise an error.


def matrix_is_diagonal(H):
	M = Matrix(H)

	return M.is_diagonal()

def matrix_is_diagonalizable(H):
	M = Matrix(H)

	return M.is_diagonalizable()


def _evolve(psi0, H, t):
	"""
		First check if H is diagonal. If diagonal then evolve via diagonalization.
		Else check if H is diagonalizable. If diagonalizable then evolve via diagonal.
		Else evolve via exponentiation.
	"""

	if matrix_is_diagonal(H):
		return _evolve_diagonal(psi0, H, t)
	elif matrix_is_diagonalizable(H):
		return _evolve_via_diagonalization(psi0, H, t)


	return _evolve_via_expm(psi0, H, t)



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



def H_list_to_H(H_list):
	H_ = 0
	for element in H_list:
		tensor_product = pauli_dict[element[0]]

		for i in range(1,len(element)):
			current_matrix = pauli_dict[element[i]]
			tensor_product = np.kron(tensor_product,current_matrix)

		H_ += tensor_product

	return H_



"""
==================
==================
Main Workflow
==================
==================
"""

psi_0 = np.array([1,1])/np.sqrt(2)

N = 3

H_LIST = [['X','I','X'], ['X','Z', 'I'], ['Z','X','Z']] #Acts on a 3 qubit system (N=3).

H_eff = compute_effective_hamiltonian(H_LIST,psi_0)

H = H_list_to_H(H_LIST)

#Input state.
#psi_0 = 
phi_0 = psi_0 #This is reserved as the single copy state that will evolve via the non-linear Schrodinger equation.


psi_0_N = create_N_copy_state(psi_0, N)



#Define the hamiltonian. Start from H_list, then build H_matrix, and H_eff
#H_list = []
H = H_list_to_H(H_LIST)
H_eff = 0
t = 1

H_eff = compute_effective_hamiltonian(H_LIST, psi_0)

#For the time being we will only deal with Hamiltonians that are either
#diagonal or diagonalizable.
diagonal = True

#Evolve psi_0_N
psi_t_N = _evolve_via_diagonal_or_diagonalizable(psi_0_N, H, t, diagonal)

#Compute the density operator of this N-body system.
psi_t_N_rho = state_to_density_matrix(psi_t_N)

#Take the partial trace of this N-body system.
rho_1 = partial_trace(psi_t_N_rho, N=N, keep_qubits=[0])



#Evolve phi_0.
phi_t = runge_kutta_4(phi_0, H_eff, t, dt=0.001)

#Compute the corresponding density matrix of phi_t.
phi_t_rho = state_to_density_matrix(phi_t)

#Compute error metrics between rho_1 and phi_t_rho.
#TODO













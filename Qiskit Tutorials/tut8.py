"""
Quantum State Evolution with PROPER Tomography
Includes correct quantum state tomography reconstruction
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import product


class QuantumStateEvolver:
    """
    Quantum state evolver with PROPER quantum state tomography.
    """
    
    def __init__(self, initial_state, num_copies, hamiltonian, evolution_time, shots=1000):
        """
        Initialize the Quantum State Evolver.
        
        Parameters:
        -----------
        initial_state : list or np.ndarray
            Initial quantum state (will be normalized)
        num_copies : int
            Number of tensor copies (N)
        hamiltonian : np.ndarray
            Hamiltonian matrix (must be Hermitian)
        evolution_time : float
            Evolution time parameter
        shots : int
            Measurements per basis (default: 1000)
        """
        # Normalize initial state
        self.initial_state = np.array(initial_state, dtype=complex)
        self.initial_state = self.initial_state / np.linalg.norm(self.initial_state)
        
        self.num_copies = num_copies
        self.num_qubits = int(np.log2(len(self.initial_state)))
        self.total_qubits = self.num_qubits * num_copies
        self.shots = shots
        
        # Set Hamiltonian
        self.hamiltonian = np.array(hamiltonian, dtype=complex)
        self.evolution_time = evolution_time
        self.hamiltonian = (self.hamiltonian + self.hamiltonian.conj().T) / 2
        
        # Pre-compute evolution operator
        self.U_evolution = self._compute_evolution_operator()
        
        # Simulator
        self.simulator = AerSimulator(method='statevector')
        
        # Results
        self.evolved_state = None
        self.density_matrix = None
        self.reconstructed_state = None
        self.pauli_expectations = {}
        self.fidelity = None
        self.purity = None
    
    def _compute_evolution_operator(self):
        """Compute U = e^(-iHt)"""
        U = expm(-1j * self.hamiltonian * self.evolution_time)
        error = np.linalg.norm(U @ U.conj().T - np.eye(len(U)))
        
        if error > 1e-10:
            print(f"⚠ Warning: Unitarity error {error:.2e}, correcting...")
            from scipy.linalg import sqrtm
            U = U @ np.linalg.inv(sqrtm(U.conj().T @ U))
        
        return U
    
    def run(self, compute_exact_state=True, use_mle=False):
        """
        Run complete evolution and tomography.
        
        Parameters:
        -----------
        compute_exact_state : bool
            Compute exact state for comparison
        use_mle : bool
            Use maximum likelihood estimation (slower but more accurate)
        """
        print(f"\n{'='*70}")
        print("QUANTUM STATE EVOLUTION WITH PROPER TOMOGRAPHY")
        print(f"{'='*70}")
        print(f"System: {self.total_qubits} qubits")
        print(f"Hilbert space: {2**self.total_qubits} dimensions")
        print(f"Shots per basis: {self.shots}")
        
        # Optional: exact state
        if compute_exact_state:
            print("\n→ Computing exact evolved state...")
            self._compute_exact_evolved_state()
        
        # Main: Proper tomography
        print("\n" + "="*70)
        print("PROPER QUANTUM STATE TOMOGRAPHY")
        print("="*70)
        self._perform_proper_tomography(use_mle=use_mle)
        
        # Extract state
        print("\n→ Extracting state vector...")
        self._extract_state_from_density_matrix()
        
        # Metrics
        print("→ Calculating metrics...")
        self._calculate_metrics()
        
        self._print_results()
        
        return self
    
    def _compute_exact_evolved_state(self):
        """Compute exact evolved state"""
        tensor_state = self.initial_state
        for _ in range(self.num_copies - 1):
            tensor_state = np.kron(tensor_state, self.initial_state)
        
        evolved = self.U_evolution @ tensor_state
        self.evolved_state = Statevector(evolved)
        print(f"  ✓ Exact state computed")
    
    def _perform_proper_tomography(self, use_mle=False):
        """
        PROPER quantum state tomography reconstruction.
        
        Process:
        1. Measure in all Pauli bases
        2. Calculate Pauli expectation values
        3. Reconstruct density matrix using linear inversion or MLE
        """
        print("\n→ Step 1: Measuring Pauli expectation values...")
        
        # Get Pauli strings and measure
        pauli_strings = self._get_pauli_strings()
        print(f"  ✓ Number of Pauli operators: {len(pauli_strings)}")
        
        self.pauli_expectations = {}
        for i, pauli_string in enumerate(pauli_strings):
            expectation = self._measure_pauli_expectation(pauli_string)
            self.pauli_expectations[pauli_string] = expectation
            
            if (i + 1) % 10 == 0:
                print(f"  • Measured {i+1}/{len(pauli_strings)} Pauli operators")
        
        print(f"  ✓ All Pauli expectations measured")
        
        # Reconstruct density matrix
        print("\n→ Step 2: Reconstructing density matrix...")
        
        if use_mle:
            print("  Using Maximum Likelihood Estimation...")
            self._reconstruct_density_matrix_mle()
        else:
            print("  Using Linear Inversion...")
            self._reconstruct_density_matrix_linear_inversion()
        
        print(f"  ✓ Density matrix reconstructed")
    
    def _get_pauli_strings(self):
        """
        Generate all Pauli strings for n qubits.
        For n qubits: 4^n Pauli operators (I, X, Y, Z for each qubit)
        
        For practical systems, we use a subset if n > 3
        """
        if self.total_qubits <= 3:
            # Full tomography for small systems
            pauli_basis = ['I', 'X', 'Y', 'Z']
            all_paulis = [''.join(p) for p in product(pauli_basis, repeat=self.total_qubits)]
            return all_paulis
        else:
            # Use important Pauli strings for larger systems
            pauli_strings = []
            
            # Single-qubit Paulis
            for i in range(self.total_qubits):
                for pauli in ['X', 'Y', 'Z']:
                    pauli_string = ['I'] * self.total_qubits
                    pauli_string[i] = pauli
                    pauli_strings.append(''.join(pauli_string))
            
            # Two-qubit correlations (important for entanglement)
            for i in range(min(4, self.total_qubits)):
                for j in range(i + 1, min(4, self.total_qubits)):
                    for p1 in ['X', 'Y', 'Z']:
                        for p2 in ['X', 'Y', 'Z']:
                            pauli_string = ['I'] * self.total_qubits
                            pauli_string[i] = p1
                            pauli_string[j] = p2
                            pauli_strings.append(''.join(pauli_string))
            
            # Add identity
            pauli_strings.append('I' * self.total_qubits)
            
            return pauli_strings
    
    def _measure_pauli_expectation(self, pauli_string):
        """
        Measure expectation value ⟨Ψ|P|Ψ⟩ for Pauli operator P.
        
        This is done by:
        1. Rotating to the eigenbasis of P
        2. Measuring in computational basis
        3. Computing expectation from measurement statistics
        """
        qc = QuantumCircuit(self.total_qubits, self.total_qubits)
        
        # Prepare and evolve state
        for i in range(self.num_copies):
            start = i * self.num_qubits
            end = start + self.num_qubits
            qc.initialize(self.initial_state, range(start, end))
        
        gate = UnitaryGate(self.U_evolution, label='U(t)')
        qc.append(gate, range(self.total_qubits))
        
        # Apply basis rotations for Pauli measurement
        for qubit, pauli in enumerate(pauli_string):
            if pauli == 'X':
                qc.h(qubit)
            elif pauli == 'Y':
                qc.sdg(qubit)
                qc.h(qubit)
            # I and Z: no rotation needed
        
        # Measure
        qc.measure(range(self.total_qubits), range(self.total_qubits))
        
        # Run circuit
        result = self.simulator.run(qc, shots=self.shots).result()
        counts = result.get_counts()
        
        # Calculate expectation value
        expectation = 0.0
        for bitstring, count in counts.items():
            # Compute parity: (-1)^(number of 1s where Pauli is not I)
            parity = 1
            clean_bitstring = bitstring.replace(' ', '')
            for qubit, pauli in enumerate(pauli_string):
                if pauli != 'I' and clean_bitstring[qubit] == '1':
                    parity *= -1
            
            expectation += parity * count / self.shots
        
        return expectation
    
    def _reconstruct_density_matrix_linear_inversion(self):
        """
        Reconstruct density matrix using linear inversion.
        
        Theory: ρ = (1/2^n) Σ_P Tr(ρP) P
        where P are Pauli operators and Tr(ρP) are measured expectations
        """
        dim = 2 ** self.total_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        
        # Build density matrix from Pauli expectations
        for pauli_string, expectation in self.pauli_expectations.items():
            # Build Pauli operator matrix
            P = self._pauli_string_to_matrix(pauli_string)
            
            # Add to density matrix: ρ += (expectation / 2^n) * P
            rho += (expectation / dim) * P
        
        # Ensure proper density matrix
        rho = (rho + rho.conj().T) / 2  # Hermiticity
        
        # Project to physical density matrix (positive semidefinite, trace 1)
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        eigenvalues = np.maximum(eigenvalues, 0)  # Remove negative eigenvalues
        eigenvalues = eigenvalues / np.sum(eigenvalues)  # Normalize
        
        rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        
        self.density_matrix = DensityMatrix(rho)
    
    def _reconstruct_density_matrix_mle(self):
        """
        Reconstruct density matrix using Maximum Likelihood Estimation.
        This is more accurate but computationally expensive.
        """
        dim = 2 ** self.total_qubits
        
        # Start with linear inversion as initial guess
        self._reconstruct_density_matrix_linear_inversion()
        rho_init = self.density_matrix.data
        
        # Parameterize density matrix using Cholesky decomposition
        # ρ = T T† where T is lower triangular
        # This ensures positive semidefiniteness
        
        def cholesky_to_density_matrix(params):
            """Convert Cholesky parameters to density matrix"""
            T = np.zeros((dim, dim), dtype=complex)
            idx = 0
            for i in range(dim):
                for j in range(i + 1):
                    if i == j:
                        T[i, j] = params[idx]  # Diagonal: real
                        idx += 1
                    else:
                        T[i, j] = params[idx] + 1j * params[idx + 1]  # Off-diagonal: complex
                        idx += 2
            
            rho = T @ T.conj().T
            rho = rho / np.trace(rho)  # Normalize
            return rho
        
        def negative_log_likelihood(params):
            """Negative log-likelihood to minimize"""
            rho = cholesky_to_density_matrix(params)
            
            # Calculate log-likelihood from Pauli expectations
            log_likelihood = 0
            for pauli_string, measured_exp in self.pauli_expectations.items():
                P = self._pauli_string_to_matrix(pauli_string)
                predicted_exp = np.real(np.trace(rho @ P))
                
                # Likelihood based on measurement statistics
                log_likelihood -= (measured_exp - predicted_exp) ** 2
            
            return -log_likelihood
        
        # Initial parameters from linear inversion
        L = np.linalg.cholesky(rho_init + 1e-10 * np.eye(dim))
        params_init = []
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    params_init.append(np.real(L[i, j]))
                else:
                    params_init.append(np.real(L[i, j]))
                    params_init.append(np.imag(L[i, j]))
        
        # Optimize
        print("  Running optimization...")
        result = minimize(negative_log_likelihood, params_init, method='BFGS', 
                         options={'maxiter': 100, 'disp': False})
        
        # Get final density matrix
        rho_mle = cholesky_to_density_matrix(result.x)
        self.density_matrix = DensityMatrix(rho_mle)
        
        print(f"  ✓ MLE converged (iterations: {result.nit})")
    
    def _pauli_string_to_matrix(self, pauli_string):
        """Convert Pauli string like 'XYZ' to matrix"""
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        # Build tensor product
        matrix = np.array([[1]], dtype=complex)
        for pauli in pauli_string:
            matrix = np.kron(matrix, pauli_dict[pauli])
        
        return matrix
    
    def _extract_state_from_density_matrix(self):
        """Extract state from density matrix"""
        eigenvalues, eigenvectors = np.linalg.eigh(self.density_matrix.data)
        max_idx = np.argmax(np.abs(eigenvalues))
        state_vector = eigenvectors[:, max_idx]
        
        self.reconstructed_state = Statevector(state_vector)
        print(f"  ✓ Largest eigenvalue: {eigenvalues[max_idx]:.6f}")
    
    def _calculate_metrics(self):
        """Calculate fidelity and purity"""
        self.purity = np.real(np.trace(self.density_matrix.data @ self.density_matrix.data))
        
        if self.evolved_state is not None:
            overlap = np.vdot(self.evolved_state.data, self.reconstructed_state.data)
            self.fidelity = np.abs(overlap) ** 2
    
    def _print_results(self):
        """Print results"""
        print(f"\n{'='*70}")
        print("TOMOGRAPHY RESULTS")
        print(f"{'='*70}")
        
        print(f"\nPauli Measurements:")
        print(f"  • Number of Pauli operators measured: {len(self.pauli_expectations)}")
        print(f"  • Shots per Pauli: {self.shots}")
        
        print(f"\nReconstructed Density Matrix:")
        print(f"  • Trace: {np.trace(self.density_matrix.data):.6f}")
        print(f"  • Purity: {self.purity:.6f}")
        
        if self.fidelity is not None:
            print(f"\nFidelity with Exact State:")
            print(f"  • Fidelity: {self.fidelity:.6f}")
        
        print(f"{'='*70}\n")
    
    def plot(self, figsize=(14, 10)):
        """Visualize results"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Density matrix real
        rho_real = np.real(self.density_matrix.data)
        im1 = axes[0, 0].imshow(rho_real, cmap='RdBu', 
                                vmin=-np.max(np.abs(rho_real)), 
                                vmax=np.max(np.abs(rho_real)))
        axes[0, 0].set_title('Density Matrix (Real)', fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Density matrix imaginary
        rho_imag = np.imag(self.density_matrix.data)
        im2 = axes[0, 1].imshow(rho_imag, cmap='RdBu',
                                vmin=-np.max(np.abs(rho_imag)), 
                                vmax=np.max(np.abs(rho_imag)))
        axes[0, 1].set_title('Density Matrix (Imaginary)', fontweight='bold')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Populations
        populations = np.real(np.diag(self.density_matrix.data))
        axes[1, 0].bar(range(len(populations)), populations, color='steelblue', alpha=0.7)
        axes[1, 0].set_title('State Populations', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(self.density_matrix.data)
        eigenvalues = np.sort(eigenvalues)[::-1]
        axes[1, 1].bar(range(len(eigenvalues)), eigenvalues, color='coral', alpha=0.7)
        axes[1, 1].set_title('Eigenvalue Spectrum', fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_pauli_expectations(self, num_show=10):
        """Print sample Pauli expectation values"""
        print(f"\n{'='*70}")
        print("PAULI EXPECTATION VALUES")
        print(f"{'='*70}")
        
        for i, (pauli, expectation) in enumerate(list(self.pauli_expectations.items())[:num_show]):
            print(f"  ⟨{pauli}⟩ = {expectation:+.6f}")
        
        if len(self.pauli_expectations) > num_show:
            print(f"  ... and {len(self.pauli_expectations) - num_show} more")
        
        print(f"{'='*70}\n")


# =============================================================================
# EXAMPLE
# =============================================================================

def example_proper_tomography():
    """Example with PROPER tomography"""
    print("\n" + "="*70)
    print("PROPER QUANTUM STATE TOMOGRAPHY EXAMPLE")
    print("="*70)
    
    # Bell state
    initial_state = [1, 0, 0, 1]
    num_copies = 1
    
    # Simple Hamiltonian
    dim = 4
    hamiltonian = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        hamiltonian[i, i] = 0.1 * i
    for i in range(dim - 1):
        hamiltonian[i, i+1] = 0.3
        hamiltonian[i+1, i] = 0.3
    
    evolution_time = 1.0
    
    # Run with proper tomography
    evolver = QuantumStateEvolver(
        initial_state=initial_state,
        num_copies=num_copies,
        hamiltonian=hamiltonian,
        evolution_time=evolution_time,
        shots=1000
    )
    
    evolver.run(compute_exact_state=True, use_mle=False)
    evolver.print_pauli_expectations()
    evolver.plot()
    
    return evolver


if __name__ == "__main__":
    evolver = example_proper_tomography()

    # Get the reconstructed state vector
    reconstructed_state = evolver.reconstructed_state

    # Access the state vector data (numpy array)
    state_vector = reconstructed_state.data

    # Print it
    print("\nReconstructed State Vector:")
    print(state_vector)

    # Or print with nice formatting
    print("\nReconstructed State Vector (formatted):")
    for i, amplitude in enumerate(state_vector):
        basis = format(i, f'0{evolver.total_qubits}b')
        real_part = np.real(amplitude)
        imag_part = np.imag(amplitude)
        print(f"|{basis}⟩: {real_part:+.6f} {imag_part:+.6f}i")
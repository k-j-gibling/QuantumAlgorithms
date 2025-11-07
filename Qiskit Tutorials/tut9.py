"""
Quantum State Evolution and Tomography System
==============================================

A comprehensive system for:
1. Creating N tensor copies of a quantum state
2. Evolving under a user-specified Hamiltonian
3. Performing quantum state tomography
4. Reconstructing the evolved state and density matrix

Author: Research Implementation
Version: 1.0
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import expm
import matplotlib.pyplot as plt
from itertools import product
import json
from datetime import datetime


class QuantumTomographySystem:
    """
    Complete system for quantum state evolution and tomography.
    
    This class provides a full pipeline for:
    - Creating tensor product states
    - Hamiltonian evolution
    - Quantum state tomography
    - Density matrix reconstruction
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the Quantum Tomography System.
        
        Parameters:
        -----------
        verbose : bool
            If True, print detailed progress information
        """
        self.verbose = verbose
        self.simulator = AerSimulator(method='statevector')
        
        # Configuration
        self.initial_state = None
        self.num_copies = None
        self.hamiltonian = None
        self.evolution_time = None
        self.shots = None
        
        # Derived parameters
        self.num_qubits = None
        self.total_qubits = None
        self.U_evolution = None
        
        # Results
        self.evolved_state_exact = None
        self.density_matrix = None
        self.reconstructed_state = None
        self.pauli_expectations = {}
        self.fidelity = None
        self.purity = None
        
    def configure(self, initial_state, num_copies, hamiltonian, evolution_time, shots=1000):
        """
        Configure the system with user parameters.
        
        Parameters:
        -----------
        initial_state : list, np.ndarray
            Initial quantum state vector
            Example: [1, 0, 0, 1] for Bell state
        num_copies : int
            Number of tensor product copies (N)
            Total qubits = (qubits in initial_state) × N
        hamiltonian : np.ndarray
            Hamiltonian matrix (must be Hermitian)
            Dimension: 2^(total_qubits) × 2^(total_qubits)
        evolution_time : float
            Evolution time parameter t in U(t) = e^(-iHt)
        shots : int, optional
            Number of measurements per Pauli basis (default: 1000)
        
        Returns:
        --------
        self : QuantumTomographySystem
            Returns self for method chaining
        """
        # Store initial state
        self.initial_state = np.array(initial_state, dtype=complex)
        norm = np.linalg.norm(self.initial_state)
        self.initial_state = self.initial_state / norm
        
        # Store configuration
        self.num_copies = num_copies
        self.num_qubits = int(np.log2(len(self.initial_state)))
        self.total_qubits = self.num_qubits * num_copies
        self.shots = shots
        
        # Validate and store Hamiltonian
        self.hamiltonian = np.array(hamiltonian, dtype=complex)
        self._validate_hamiltonian()
        
        # Store evolution time
        self.evolution_time = evolution_time
        
        # Compute evolution operator
        self._compute_evolution_operator()
        
        if self.verbose:
            self._print_configuration()
        
        return self
    
    def _validate_hamiltonian(self):
        """Validate Hamiltonian properties"""
        # Make Hermitian
        self.hamiltonian = (self.hamiltonian + self.hamiltonian.conj().T) / 2
        
        # Check dimensions
        expected_dim = 2 ** self.total_qubits
        if self.hamiltonian.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Hamiltonian dimension {self.hamiltonian.shape} doesn't match "
                f"expected dimension {expected_dim}×{expected_dim} for {self.total_qubits} qubits"
            )
        
        # Check Hermiticity
        error = np.linalg.norm(self.hamiltonian - self.hamiltonian.conj().T)
        if error > 1e-10:
            print(f"⚠ Warning: Hamiltonian Hermiticity error {error:.2e}")
    
    def _compute_evolution_operator(self):
        """Compute unitary evolution operator U(t) = e^(-iHt)"""
        self.U_evolution = expm(-1j * self.hamiltonian * self.evolution_time)
        
        # Verify unitarity
        error = np.linalg.norm(self.U_evolution @ self.U_evolution.conj().T - np.eye(len(self.U_evolution)))
        
        if error > 1e-10:
            if self.verbose:
                print(f"⚠ Warning: Unitarity error {error:.2e}, correcting...")
            from scipy.linalg import sqrtm
            self.U_evolution = self.U_evolution @ np.linalg.inv(sqrtm(self.U_evolution.conj().T @ self.U_evolution))
    
    def _print_configuration(self):
        """Print system configuration"""
        print("\n" + "="*70)
        print("QUANTUM TOMOGRAPHY SYSTEM - CONFIGURATION")
        print("="*70)
        print(f"\nInitial State:")
        print(f"  • Dimension: {len(self.initial_state)}")
        print(f"  • Qubits: {self.num_qubits}")
        print(f"  • Norm: {np.linalg.norm(self.initial_state):.6f}")
        
        # Show initial state amplitudes
        print(f"  • Non-zero amplitudes:")
        for i, amp in enumerate(self.initial_state):
            if np.abs(amp) > 1e-6:
                basis = format(i, f'0{self.num_qubits}b')
                print(f"      |{basis}⟩: {np.real(amp):+.6f} {np.imag(amp):+.6f}i")
        
        print(f"\nSystem Configuration:")
        print(f"  • Number of copies (N): {self.num_copies}")
        print(f"  • Total qubits: {self.total_qubits}")
        print(f"  • Hilbert space dimension: {2**self.total_qubits}")
        
        print(f"\nHamiltonian:")
        eigenvalues = np.linalg.eigvalsh(self.hamiltonian)
        print(f"  • Dimension: {self.hamiltonian.shape[0]}*{self.hamiltonian.shape[1]}")
        print(f"  • Eigenvalue range: [{np.min(eigenvalues):.6f}, {np.max(eigenvalues):.6f}]")
        print(f"  • Spectral gap: {eigenvalues[1] - eigenvalues[0]:.6f}")
        
        print(f"\nEvolution:")
        print(f"  • Evolution time (t): {self.evolution_time}")
        
        print(f"\nMeasurement:")
        print(f"  • Shots per Pauli basis: {self.shots}")
        
        print("="*70 + "\n")
    
    def run_tomography(self, compute_exact_state=True):
        """
        Execute the complete tomography process.
        
        Process:
        1. (Optional) Compute exact evolved state for validation
        2. Measure Pauli expectation values
        3. Reconstruct density matrix via linear inversion
        4. Extract state vector from density matrix
        5. Calculate metrics (fidelity, purity)
        
        Parameters:
        -----------
        compute_exact_state : bool, optional
            If True, compute exact evolved state for comparison (default: True)
        
        Returns:
        --------
        results : dict
            Dictionary containing all results
        """
        if self.initial_state is None:
            raise ValueError("System not configured. Call configure() first.")
        
        print("\n" + "="*70)
        print("QUANTUM STATE TOMOGRAPHY - EXECUTION")
        print("="*70)
        
        # Step 1: Compute exact state (optional, for validation)
        if compute_exact_state:
            print("\n→ Step 1: Computing exact evolved state (for validation)...")
            self._compute_exact_evolved_state()
        
        # Step 2: Perform tomography
        print("\n→ Step 2: Performing quantum state tomography...")
        self._perform_tomography()
        
        # Step 3: Extract state vector
        print("\n→ Step 3: Extracting state vector from density matrix...")
        self._extract_state_vector()
        
        # Step 4: Calculate metrics
        print("\n→ Step 4: Calculating quality metrics...")
        self._calculate_metrics()
        
        # Step 5: Print results
        self._print_results()
        
        return self.get_results()
    
    def _compute_exact_evolved_state(self):
        """Compute exact evolved state (for validation only)"""
        # Build tensor product state
        tensor_state = self.initial_state
        for _ in range(self.num_copies - 1):
            tensor_state = np.kron(tensor_state, self.initial_state)
        
        # Apply evolution
        evolved = self.U_evolution @ tensor_state
        self.evolved_state_exact = Statevector(evolved)
        
        if self.verbose:
            print(f"  ✓ Exact evolved state computed")
    
    def _perform_tomography(self):
        """
        Perform quantum state tomography.
        
        Measures Pauli expectation values and reconstructs density matrix.
        """
        print("\n  → Generating Pauli measurement bases...")
        pauli_strings = self._generate_pauli_strings()
        
        if self.verbose:
            print(f"    • Number of Pauli operators: {len(pauli_strings)}")
            print(f"    • Total measurements: {len(pauli_strings) * self.shots}")
        
        print("\n  → Measuring Pauli expectation values...")
        self.pauli_expectations = {}
        
        for i, pauli_string in enumerate(pauli_strings):
            expectation = self._measure_pauli_expectation(pauli_string)
            self.pauli_expectations[pauli_string] = expectation
            
            if self.verbose and ((i + 1) % 10 == 0 or i == 0):
                print(f"    • Measured {i+1}/{len(pauli_strings)}: {pauli_string} = {expectation:+.6f}")
        
        if self.verbose:
            print(f"  ✓ All Pauli expectations measured")
        
        print("\n  → Reconstructing density matrix from measurements...")
        self._reconstruct_density_matrix()
        
        if self.verbose:
            print(f"  ✓ Density matrix reconstructed")
    
    def _generate_pauli_strings(self):
        """Generate Pauli strings for tomography"""
        if self.total_qubits <= 3:
            # Full tomography for small systems
            pauli_basis = ['I', 'X', 'Y', 'Z']
            return [''.join(p) for p in product(pauli_basis, repeat=self.total_qubits)]
        else:
            # Partial tomography for larger systems
            pauli_strings = []
            
            # Single-qubit Paulis
            for i in range(self.total_qubits):
                for pauli in ['X', 'Y', 'Z']:
                    ps = ['I'] * self.total_qubits
                    ps[i] = pauli
                    pauli_strings.append(''.join(ps))
            
            # Two-qubit correlations
            for i in range(min(4, self.total_qubits)):
                for j in range(i + 1, min(4, self.total_qubits)):
                    for p1 in ['X', 'Y', 'Z']:
                        for p2 in ['X', 'Y', 'Z']:
                            ps = ['I'] * self.total_qubits
                            ps[i] = p1
                            ps[j] = p2
                            pauli_strings.append(''.join(ps))
            
            # Identity
            pauli_strings.append('I' * self.total_qubits)
            
            return pauli_strings
    
    def _measure_pauli_expectation(self, pauli_string):
        """Measure expectation value ⟨P⟩ for Pauli operator P"""
        qc = QuantumCircuit(self.total_qubits, self.total_qubits)
        
        # Prepare initial state (tensor product)
        for i in range(self.num_copies):
            start = i * self.num_qubits
            end = start + self.num_qubits
            qc.initialize(self.initial_state, range(start, end))
        
        # Apply evolution
        gate = UnitaryGate(self.U_evolution, label='U(t)')
        qc.append(gate, range(self.total_qubits))
        
        # Apply Pauli basis rotations
        for qubit, pauli in enumerate(pauli_string):
            if pauli == 'X':
                qc.h(qubit)
            elif pauli == 'Y':
                qc.sdg(qubit)
                qc.h(qubit)
        
        # Measure
        qc.measure(range(self.total_qubits), range(self.total_qubits))
        
        # Execute
        result = self.simulator.run(qc, shots=self.shots).result()
        counts = result.get_counts()
        
        # Calculate expectation value
        expectation = 0.0
        for bitstring, count in counts.items():
            parity = 1
            clean_bitstring = bitstring.replace(' ', '')
            for qubit, pauli in enumerate(pauli_string):
                if pauli != 'I' and clean_bitstring[qubit] == '1':
                    parity *= -1
            expectation += parity * count / self.shots
        
        return expectation
    
    def _reconstruct_density_matrix(self):
        """Reconstruct density matrix using linear inversion"""
        dim = 2 ** self.total_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        
        # Build density matrix: ρ = (1/dim) Σ_P ⟨P⟩ P
        for pauli_string, expectation in self.pauli_expectations.items():
            P = self._pauli_string_to_matrix(pauli_string)
            rho += (expectation / dim) * P
        
        # Ensure physical density matrix
        rho = (rho + rho.conj().T) / 2  # Hermiticity
        
        # Project to positive semidefinite
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        eigenvalues = np.maximum(eigenvalues, 0)  # Remove negative eigenvalues
        eigenvalues = eigenvalues / np.sum(eigenvalues)  # Normalize
        
        rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        
        self.density_matrix = DensityMatrix(rho)
    
    def _pauli_string_to_matrix(self, pauli_string):
        """Convert Pauli string to matrix"""
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        matrix = np.array([[1]], dtype=complex)
        for pauli in pauli_string:
            matrix = np.kron(matrix, pauli_dict[pauli])
        
        return matrix
    
    def _extract_state_vector(self):
        """Extract state vector from density matrix"""
        eigenvalues, eigenvectors = np.linalg.eigh(self.density_matrix.data)
        max_idx = np.argmax(np.abs(eigenvalues))
        state_vector = eigenvectors[:, max_idx]
        
        self.reconstructed_state = Statevector(state_vector)
        
        if self.verbose:
            print(f"  ✓ Largest eigenvalue: {eigenvalues[max_idx]:.6f}")
    
    def _calculate_metrics(self):
        """Calculate quality metrics"""
        # Purity
        self.purity = np.real(np.trace(self.density_matrix.data @ self.density_matrix.data))
        
        # Fidelity (if exact state available)
        if self.evolved_state_exact is not None:
            overlap = np.vdot(self.evolved_state_exact.data, self.reconstructed_state.data)
            self.fidelity = np.abs(overlap) ** 2
        
        if self.verbose:
            print(f"  ✓ Purity: {self.purity:.6f}")
            if self.fidelity is not None:
                print(f"  ✓ Fidelity: {self.fidelity:.6f}")
    
    def _print_results(self):
        """Print comprehensive results"""
        print("\n" + "="*70)
        print("TOMOGRAPHY RESULTS")
        print("="*70)
        
        print(f"\nMeasurements:")
        print(f"  • Pauli operators measured: {len(self.pauli_expectations)}")
        print(f"  • Total measurement shots: {len(self.pauli_expectations) * self.shots}")
        
        print(f"\nDensity Matrix:")
        print(f"  • Dimension: {self.density_matrix.data.shape[0]}×{self.density_matrix.data.shape[0]}")
        print(f"  • Trace: {np.trace(self.density_matrix.data):.6f}")
        print(f"  • Purity: {self.purity:.6f}")
        
        if self.fidelity is not None:
            print(f"\nValidation:")
            print(f"  • Fidelity with exact state: {self.fidelity:.6f}")
            quality = "Excellent ✓" if self.fidelity > 0.99 else "Good" if self.fidelity > 0.95 else "Fair"
            print(f"  • Reconstruction quality: {quality}")
        
        print("="*70 + "\n")
    
    def get_results(self):
        """
        Get all results as a dictionary.
        
        Returns:
        --------
        results : dict
            Dictionary containing:
            - density_matrix: Reconstructed density matrix
            - reconstructed_state: Reconstructed state vector
            - evolved_state_exact: Exact evolved state (if computed)
            - pauli_expectations: All measured Pauli expectation values
            - fidelity: Fidelity with exact state (if available)
            - purity: Purity of reconstructed state
        """
        return {
            'density_matrix': self.density_matrix,
            'reconstructed_state': self.reconstructed_state,
            'evolved_state_exact': self.evolved_state_exact,
            'pauli_expectations': self.pauli_expectations,
            'fidelity': self.fidelity,
            'purity': self.purity,
            'configuration': {
                'num_copies': self.num_copies,
                'total_qubits': self.total_qubits,
                'evolution_time': self.evolution_time,
                'shots': self.shots
            }
        }
    
    def get_density_matrix(self):
        """Get the reconstructed density matrix"""
        if self.density_matrix is None:
            raise ValueError("No density matrix. Run tomography first.")
        return self.density_matrix
    
    def get_reconstructed_state(self):
        """Get the reconstructed state vector"""
        if self.reconstructed_state is None:
            raise ValueError("No reconstructed state. Run tomography first.")
        return self.reconstructed_state
    
    def print_state_vector(self, threshold=1e-6):
        """
        Print the reconstructed state vector in readable format.
        
        Parameters:
        -----------
        threshold : float
            Minimum amplitude magnitude to display
        """
        if self.reconstructed_state is None:
            raise ValueError("No reconstructed state. Run tomography first.")
        
        print("\n" + "="*70)
        print("RECONSTRUCTED STATE VECTOR")
        print("="*70)
        
        state_vector = self.reconstructed_state.data
        print(f"Dimension: {len(state_vector)}")
        print(f"Norm: {np.linalg.norm(state_vector):.6f}")
        
        print("\nAmplitudes (|amplitude| > {:.0e}):".format(threshold))
        for i, amp in enumerate(state_vector):
            if np.abs(amp) > threshold:
                basis = format(i, f'0{self.total_qubits}b')
                real = np.real(amp)
                imag = np.imag(amp)
                magnitude = np.abs(amp)
                phase = np.angle(amp)
                
                print(f"  |{basis}⟩: {real:+.8f} {imag:+.8f}i")
                print(f"           magnitude = {magnitude:.8f}, phase = {phase:+.6f} rad")
        
        print("="*70 + "\n")
    
    def print_density_matrix(self, threshold=1e-6):
        """
        Print the density matrix in readable format.
        
        Parameters:
        -----------
        threshold : float
            Minimum element magnitude to display
        """
        if self.density_matrix is None:
            raise ValueError("No density matrix. Run tomography first.")
        
        print("\n" + "="*70)
        print("DENSITY MATRIX")
        print("="*70)
        
        rho = self.density_matrix.data
        print(f"Dimension: {rho.shape[0]}×{rho.shape[1]}")
        print(f"Trace: {np.trace(rho):.6f}")
        print(f"Purity: {np.real(np.trace(rho @ rho)):.6f}")
        
        print("\nNon-zero elements (|element| > {:.0e}):".format(threshold))
        count = 0
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                if np.abs(rho[i, j]) > threshold:
                    basis_i = format(i, f'0{self.total_qubits}b')
                    basis_j = format(j, f'0{self.total_qubits}b')
                    real = np.real(rho[i, j])
                    imag = np.imag(rho[i, j])
                    print(f"  ρ[|{basis_i}⟩,|{basis_j}⟩] = {real:+.8f} {imag:+.8f}i")
                    count += 1
                    if count >= 20:
                        print(f"  ... ({np.sum(np.abs(rho) > threshold) - 20} more elements)")
                        break
            if count >= 20:
                break
        
        print("="*70 + "\n")
    
    def visualize(self, figsize=(14, 10)):
        """
        Visualize tomography results.
        
        Creates four subplots:
        1. Density matrix (real part)
        2. Density matrix (imaginary part)
        3. State populations (diagonal)
        4. Eigenvalue spectrum
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.density_matrix is None:
            raise ValueError("No results to visualize. Run tomography first.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        rho = self.density_matrix.data
        
        # Real part
        rho_real = np.real(rho)
        vmax_real = np.max(np.abs(rho_real))
        im1 = axes[0, 0].imshow(rho_real, cmap='RdBu', vmin=-vmax_real, vmax=vmax_real)
        axes[0, 0].set_title('Density Matrix (Real Part)', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Basis State')
        axes[0, 0].set_ylabel('Basis State')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Imaginary part
        rho_imag = np.imag(rho)
        vmax_imag = np.max(np.abs(rho_imag))
        im2 = axes[0, 1].imshow(rho_imag, cmap='RdBu', vmin=-vmax_imag, vmax=vmax_imag)
        axes[0, 1].set_title('Density Matrix (Imaginary Part)', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Basis State')
        axes[0, 1].set_ylabel('Basis State')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Populations
        populations = np.real(np.diag(rho))
        axes[1, 0].bar(range(len(populations)), populations, color='steelblue', alpha=0.7)
        axes[1, 0].set_title('State Populations (Diagonal)', fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('Basis State')
        axes[1, 0].set_ylabel('Population')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = np.sort(eigenvalues)[::-1]
        axes[1, 1].bar(range(len(eigenvalues)), eigenvalues, color='coral', alpha=0.7)
        axes[1, 1].set_title('Eigenvalue Spectrum', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Eigenvalue Index')
        axes[1, 1].set_ylabel('Eigenvalue')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename='tomography_results.npz'):
        """
        Save results to file.
        
        Parameters:
        -----------
        filename : str
            Output filename (supports .npz or .json)
        """
        if self.density_matrix is None:
            raise ValueError("No results to save. Run tomography first.")
        
        results = {
            'density_matrix_real': np.real(self.density_matrix.data),
            'density_matrix_imag': np.imag(self.density_matrix.data),
            'reconstructed_state_real': np.real(self.reconstructed_state.data),
            'reconstructed_state_imag': np.imag(self.reconstructed_state.data),
            'fidelity': self.fidelity if self.fidelity is not None else -1,
            'purity': self.purity,
            'num_copies': self.num_copies,
            'total_qubits': self.total_qubits,
            'evolution_time': self.evolution_time,
            'shots': self.shots
        }
        
        if filename.endswith('.npz'):
            np.savez(filename, **results)
        else:
            # Convert to JSON-serializable format
            json_results = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                          for k, v in results.items()}
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2)
        
        print(f"✓ Results saved to {filename}")


# =============================================================================
# INTERACTIVE INTERFACE
# =============================================================================

def interactive_session():
    """
    Interactive command-line interface for the tomography system.
    """
    print("\n" + "="*70)
    print(" " * 15 + "QUANTUM TOMOGRAPHY SYSTEM")
    print(" " * 20 + "Interactive Mode")
    print("="*70)
    
    system = QuantumTomographySystem(verbose=True)
    
    # =========================================================================
    # STEP 1: Define Initial State
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: DEFINE INITIAL STATE")
    print("="*70)
    
    print("\nChoose initial state:")
    print("  1. Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print("  2. GHZ state (|000⟩ + |111⟩)/√2")
    print("  3. Product state |00⟩")
    print("  4. Equal superposition |+⟩⊗n")
    print("  5. Custom state (manual input)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        initial_state = [1, 0, 0, 1]
        print("Selected: Bell state")
    elif choice == '2':
        initial_state = [1, 0, 0, 0, 0, 0, 0, 1]
        print("Selected: GHZ state")
    elif choice == '3':
        initial_state = [1, 0, 0, 0]
        print("Selected: |00⟩")
    elif choice == '4':
        n = int(input("Enter number of qubits: "))
        initial_state = np.ones(2**n)
        print(f"Selected: Equal superposition on {n} qubits")
    else:
        print("\nEnter state amplitudes (space-separated):")
        print("Example: 1 0 0 1  (for Bell state)")
        state_str = input("Amplitudes: ")
        initial_state = [float(x) for x in state_str.split()]
        print(f"Custom state with {len(initial_state)} amplitudes")
    
    # =========================================================================
    # STEP 2: Number of Copies
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: NUMBER OF TENSOR COPIES")
    print("="*70)
    
    num_copies = int(input("\nEnter number of copies N (1-3 recommended): "))
    
    # =========================================================================
    # STEP 3: Define Hamiltonian
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: DEFINE HAMILTONIAN")
    print("="*70)
    
    num_qubits_single = int(np.log2(len(initial_state)))
    total_qubits = num_qubits_single * num_copies
    dim = 2 ** total_qubits
    
    print(f"\nTotal system: {total_qubits} qubits, dimension {dim}×{dim}")
    print("\nChoose Hamiltonian:")
    print("  1. Random Hermitian")
    print("  2. Diagonal (on-site energies)")
    print("  3. Nearest-neighbor hopping")
    print("  4. Custom matrix (manual input)")
    
    h_choice = input("\nEnter choice (1-4): ").strip()
    
    if h_choice == '1':
        strength = float(input("Enter strength (e.g., 0.5): "))
        H = np.random.randn(dim, dim)
        hamiltonian = (H + H.T) / 2 * strength
        print("Generated: Random Hermitian Hamiltonian")
    
    elif h_choice == '2':
        hamiltonian = np.diag([0.1 * i for i in range(dim)])
        print("Generated: Diagonal Hamiltonian with linearly increasing energies")
    
    elif h_choice == '3':
        coupling = float(input("Enter coupling strength (e.g., 0.3): "))
        hamiltonian = np.zeros((dim, dim))
        for i in range(dim):
            hamiltonian[i, i] = 0.1 * i
        for i in range(dim - 1):
            hamiltonian[i, i+1] = coupling
            hamiltonian[i+1, i] = coupling
        print("Generated: Nearest-neighbor Hamiltonian")
    
    else:
        print(f"\nFor {dim}×{dim} matrix, using random Hamiltonian instead.")
        H = np.random.randn(dim, dim)
        hamiltonian = (H + H.T) / 2 * 0.5
    
    # =========================================================================
    # STEP 4: Evolution Time
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: EVOLUTION TIME")
    print("="*70)
    
    evolution_time = float(input("\nEnter evolution time t (e.g., 1.0): "))
    
    # =========================================================================
    # STEP 5: Measurement Shots
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: MEASUREMENT SHOTS")
    print("="*70)
    
    shots = int(input("\nEnter shots per Pauli basis (e.g., 1000): "))
    
    # =========================================================================
    # STEP 6: Configure and Run
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: RUN TOMOGRAPHY")
    print("="*70)
    
    input("\nPress Enter to configure system...")
    system.configure(initial_state, num_copies, hamiltonian, evolution_time, shots)
    
    input("\nPress Enter to run tomography...")
    results = system.run_tomography(compute_exact_state=True)
    
    # =========================================================================
    # STEP 7: View Results
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: VIEW RESULTS")
    print("="*70)
    
    while True:
        print("\nOptions:")
        print("  1. Print reconstructed state vector")
        print("  2. Print density matrix")
        print("  3. Visualize results")
        print("  4. Save results to file")
        print("  5. Exit")
        
        option = input("\nEnter choice (1-5): ").strip()
        
        if option == '1':
            system.print_state_vector()
        elif option == '2':
            system.print_density_matrix()
        elif option == '3':
            system.visualize()
        elif option == '4':
            filename = input("Enter filename (e.g., results.npz): ")
            system.save_results(filename)
        else:
            break
    
    return system


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """
    Example demonstrating the complete workflow.
    """
    print("\n" + "="*70)
    print("EXAMPLE: Complete Tomography Workflow")
    print("="*70)
    
    # Create system
    system = QuantumTomographySystem(verbose=True)
    
    # Configure
    initial_state = [1, 0, 0, 1]  # Bell state
    num_copies = 1
    
    # Create Hamiltonian
    dim = 4  # 2 qubits
    hamiltonian = np.zeros((dim, dim))
    for i in range(dim):
        hamiltonian[i, i] = 0.1 * i
    for i in range(dim - 1):
        hamiltonian[i, i+1] = 0.3
        hamiltonian[i+1, i] = 0.3
    
    evolution_time = 1.0
    shots = 1000
    
    # Configure system
    system.configure(initial_state, num_copies, hamiltonian, evolution_time, shots)
    
    # Run tomography
    results = system.run_tomography(compute_exact_state=True)
    
    # Display results
    system.print_state_vector()
    system.print_density_matrix()
    
    # Visualize
    system.visualize()
    
    # Access results programmatically
    density_matrix = system.get_density_matrix()
    reconstructed_state = system.get_reconstructed_state()
    
    print("\nDensity matrix shape:", density_matrix.data.shape)
    print("Reconstructed state shape:", reconstructed_state.data.shape)
    
    return system


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\nQuantum Tomography System")
    print("=" * 70)
    print("\nChoose mode:")
    print("  1. Interactive mode (guided input)")
    print("  2. Example usage")
    
    mode = input("\nEnter choice (1-2): ").strip()
    
    if mode == '1':
        system = interactive_session()
    else:
        system = example_usage()
    
    print("\n✓ Session complete!")
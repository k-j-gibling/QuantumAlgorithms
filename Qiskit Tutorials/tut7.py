"""
Quantum State Evolution and Tomography
Complete implementation with explicit tomography tracking
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import expm
import matplotlib.pyplot as plt


class QuantumStateEvolver:
    """
    Quantum state evolver with explicit tomography tracking.
    
    This class performs:
    1. Tensor product state preparation
    2. Hamiltonian evolution
    3. Quantum state tomography (measurements in multiple bases)
    4. Density matrix reconstruction
    5. State vector extraction
    """
    
    def __init__(self, initial_state, num_copies, hamiltonian, evolution_time, shots=1000):
        """
        Initialize the Quantum State Evolver.
        
        Parameters:
        -----------
        initial_state : list or np.ndarray
            Initial quantum state (will be normalized)
            Example: [1, 0, 0, 1] for Bell state
        num_copies : int
            Number of tensor copies (N)
            Example: 2 means state ⊗ state
        hamiltonian : np.ndarray
            Hamiltonian matrix (must be Hermitian)
            Dimension must match total system size
        evolution_time : float
            Evolution time parameter (t in e^(-iHt))
        shots : int, optional
            Measurements per basis (default: 1000)
        """
        # Normalize initial state
        self.initial_state = np.array(initial_state, dtype=complex)
        self.initial_state = self.initial_state / np.linalg.norm(self.initial_state)
        
        # System parameters
        self.num_copies = num_copies
        self.num_qubits = int(np.log2(len(self.initial_state)))
        self.total_qubits = self.num_qubits * num_copies
        self.shots = shots
        
        # Set Hamiltonian
        self.hamiltonian = np.array(hamiltonian, dtype=complex)
        self.evolution_time = evolution_time
        
        # Make sure Hamiltonian is Hermitian
        self.hamiltonian = (self.hamiltonian + self.hamiltonian.conj().T) / 2
        
        # Pre-compute evolution operator
        self.U_evolution = self._compute_evolution_operator()
        
        # Simulator
        self.simulator = AerSimulator(method='statevector')
        
        # Results storage
        self.evolved_state = None
        self.density_matrix = None
        self.reconstructed_state = None
        self.fidelity = None
        self.purity = None
        self.tomography_measurements = {}
        self.num_measurement_bases = 0
    
    def _compute_evolution_operator(self):
        """
        Compute the unitary evolution operator U = e^(-iHt).
        Uses scipy's matrix exponential for numerical stability.
        """
        # Compute matrix exponential
        U = expm(-1j * self.hamiltonian * self.evolution_time)
        
        # Check unitarity: U†U should be identity
        error = np.linalg.norm(U @ U.conj().T - np.eye(len(U)))
        
        if error > 1e-10:
            print(f"⚠ Warning: Unitarity error {error:.2e}, correcting...")
            # Apply polar decomposition to ensure unitarity
            from scipy.linalg import sqrtm
            U = U @ np.linalg.inv(sqrtm(U.conj().T @ U))
        
        return U
    
    def run(self, compute_exact_state=True):
        """
        Run the complete evolution and tomography process.
        
        Parameters:
        -----------
        compute_exact_state : bool, optional
            If True, compute exact evolved state for comparison
            If False, only use tomography (more realistic quantum scenario)
        
        Returns:
        --------
        self : QuantumStateEvolver
            Returns self for method chaining
        """
        print(f"\n{'='*70}")
        print("QUANTUM STATE EVOLUTION WITH TOMOGRAPHY")
        print(f"{'='*70}")
        print(f"Initial state dimension: {len(self.initial_state)}")
        print(f"Number of copies: {self.num_copies}")
        print(f"Total qubits: {self.total_qubits}")
        print(f"Hilbert space dimension: {2**self.total_qubits}")
        print(f"Shots per measurement basis: {self.shots}")
        
        # Optional: Compute exact evolved state (for validation)
        if compute_exact_state:
            print("\n→ Computing exact evolved state (for validation)...")
            self._compute_exact_evolved_state()
        
        # Main: Perform tomography
        print("\n" + "="*70)
        print("QUANTUM STATE TOMOGRAPHY")
        print("="*70)
        self._perform_tomography()
        
        # Extract state from density matrix
        print("\n→ Extracting state vector from reconstructed density matrix...")
        self._extract_state_from_density_matrix()
        
        # Calculate metrics
        print("→ Calculating metrics...")
        self._calculate_metrics()
        
        # Print results
        self._print_results()
        
        return self
    
    def _compute_exact_evolved_state(self):
        """
        Compute exact evolved state using direct matrix multiplication.
        This is for validation only - not available on real quantum hardware.
        """
        # Build tensor product state
        tensor_state = self.initial_state
        for _ in range(self.num_copies - 1):
            tensor_state = np.kron(tensor_state, self.initial_state)
        
        # Apply evolution operator
        evolved = self.U_evolution @ tensor_state
        self.evolved_state = Statevector(evolved)
        
        print(f"  ✓ Exact state computed (norm: {np.linalg.norm(evolved):.6f})")
    
    def _perform_tomography(self):
        """
        Perform quantum state tomography by measuring in multiple Pauli bases.
        
        This is the CORE tomography functionality:
        1. Generate measurement bases (X, Y, Z for each qubit)
        2. Measure evolved state in each basis (1000 shots each)
        3. Reconstruct density matrix from measurement statistics
        """
        print("\n→ Step 1: Generating Pauli measurement bases...")
        bases = self._get_pauli_bases()
        self.num_measurement_bases = len(bases)
        print(f"  ✓ Generated {self.num_measurement_bases} measurement bases")
        print(f"  ✓ Total measurements: {self.num_measurement_bases} × {self.shots} = {self.num_measurement_bases * self.shots}")
        
        print("\n→ Step 2: Performing measurements in each basis...")
        dim = 2 ** self.total_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        total_counts = 0
        
        for idx, (basis_name, basis_circuit) in enumerate(bases.items()):
            # Measure in this basis (1000 shots)
            counts = self._measure_in_basis(basis_circuit)
            self.tomography_measurements[basis_name] = counts
            
            # Show progress
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"  • Measured basis {idx+1}/{self.num_measurement_bases}: {basis_name}")
            
            # Accumulate density matrix from measurement outcomes
            for bitstring, count in counts.items():
                idx_state = int(bitstring.replace(' ', ''), 2)
                basis_state = np.zeros(dim)
                basis_state[idx_state] = 1.0
                
                # Add |ψ⟩⟨ψ| weighted by measurement count
                rho += count * np.outer(basis_state, basis_state)
                total_counts += count
        
        print(f"  ✓ Completed all {self.num_measurement_bases} measurements")
        print(f"  ✓ Total measurement outcomes: {total_counts}")
        
        print("\n→ Step 3: Reconstructing density matrix from measurements...")
        # Normalize
        rho = rho / total_counts
        
        # Ensure proper density matrix properties
        rho = (rho + rho.conj().T) / 2  # Hermiticity: ρ = ρ†
        rho = rho / np.trace(rho)  # Normalization: Tr(ρ) = 1
        
        self.density_matrix = DensityMatrix(rho)
        print(f"  ✓ Density matrix reconstructed")
        print(f"  ✓ Trace: {np.trace(rho):.6f}")
        print(f"  ✓ Hermiticity error: {np.linalg.norm(rho - rho.conj().T):.2e}")
    
    def _get_pauli_bases(self):
        """
        Generate Pauli measurement bases for tomography.
        
        For each qubit, we measure in three bases:
        - X basis: Apply H gate before measurement
        - Y basis: Apply S† then H before measurement
        - Z basis: Direct computational basis measurement
        
        Returns:
        --------
        bases : dict
            Dictionary of {basis_name: measurement_circuit}
        """
        bases = {}
        
        # Measure each qubit in X, Y, Z basis
        for qubit in range(self.total_qubits):
            for pauli in ['X', 'Y', 'Z']:
                basis_name = f"{pauli}{qubit}"
                qc = QuantumCircuit(self.total_qubits, self.total_qubits)
                
                # Apply basis rotation
                if pauli == 'X':
                    qc.h(qubit)
                elif pauli == 'Y':
                    qc.sdg(qubit)
                    qc.h(qubit)
                # Z is computational basis (no rotation needed)
                
                # Measure all qubits
                qc.measure(range(self.total_qubits), range(self.total_qubits))
                bases[basis_name] = qc
        
        return bases
    
    def _measure_in_basis(self, basis_circuit):
        """
        Perform actual quantum measurements in a specific basis.
        This simulates running the quantum circuit 'shots' times.
        
        Process:
        1. Prepare initial state (tensor product)
        2. Apply time evolution U(t)
        3. Apply basis rotation
        4. Measure (repeat 'shots' times)
        
        Parameters:
        -----------
        basis_circuit : QuantumCircuit
            Circuit that implements basis rotation and measurement
        
        Returns:
        --------
        counts : dict
            Measurement outcomes, e.g., {'0000': 245, '0001': 255, ...}
        """
        # Build complete circuit: prepare + evolve + measure
        qc = QuantumCircuit(self.total_qubits, self.total_qubits)
        
        # 1. Prepare initial state (tensor product of copies)
        for i in range(self.num_copies):
            start = i * self.num_qubits
            end = start + self.num_qubits
            qc.initialize(self.initial_state, range(start, end))
        
        # 2. Apply time evolution U(t) = e^(-iHt)
        gate = UnitaryGate(self.U_evolution, label='U(t)')
        qc.append(gate, range(self.total_qubits))
        
        # 3. Apply basis rotation and measure
        qc.compose(basis_circuit, inplace=True)
        
        # 4. Run circuit 'shots' times and collect measurement outcomes
        result = self.simulator.run(qc, shots=self.shots).result()
        counts = result.get_counts()
        
        return counts
    
    def _extract_state_from_density_matrix(self):
        """
        Extract pure state approximation from density matrix.
        The state is the eigenvector with the largest eigenvalue.
        """
        # Get eigendecomposition: ρ = Σᵢ λᵢ |ψᵢ⟩⟨ψᵢ|
        eigenvalues, eigenvectors = np.linalg.eigh(self.density_matrix.data)
        
        # State is eigenvector with largest eigenvalue
        max_idx = np.argmax(np.abs(eigenvalues))
        state_vector = eigenvectors[:, max_idx]
        
        self.reconstructed_state = Statevector(state_vector)
        
        print(f"  ✓ Largest eigenvalue: {eigenvalues[max_idx]:.6f}")
    
    def _calculate_metrics(self):
        """
        Calculate quality metrics for the tomography.
        
        Metrics:
        - Purity: Tr(ρ²) - measures how mixed the state is (1 = pure)
        - Fidelity: |⟨ψ_exact|ψ_reconstructed⟩|² - compares with exact state
        """
        # Purity: Tr(ρ²)
        self.purity = np.real(np.trace(self.density_matrix.data @ self.density_matrix.data))
        
        # Fidelity (if exact state is known)
        if self.evolved_state is not None:
            overlap = np.vdot(self.evolved_state.data, self.reconstructed_state.data)
            self.fidelity = np.abs(overlap) ** 2
    
    def _print_results(self):
        """Print comprehensive results summary."""
        print(f"\n{'='*70}")
        print("TOMOGRAPHY RESULTS")
        print(f"{'='*70}")
        
        print(f"\nMeasurement Statistics:")
        print(f"  • Number of measurement bases: {self.num_measurement_bases}")
        print(f"  • Shots per basis: {self.shots}")
        print(f"  • Total measurements: {self.num_measurement_bases * self.shots}")
        
        print(f"\nReconstructed Density Matrix:")
        print(f"  • Dimension: {self.density_matrix.data.shape[0]}×{self.density_matrix.data.shape[0]}")
        print(f"  • Trace: {np.trace(self.density_matrix.data):.6f}")
        print(f"  • Purity: {self.purity:.6f}")
        
        if self.fidelity is not None:
            print(f"\nComparison with Exact Evolution:")
            print(f"  • Fidelity: {self.fidelity:.6f}")
            if self.fidelity > 0.99:
                quality = "Excellent ✓"
            elif self.fidelity > 0.95:
                quality = "Good"
            elif self.fidelity > 0.90:
                quality = "Fair"
            else:
                quality = "Poor"
            print(f"  • Quality: {quality}")
        
        print(f"{'='*70}\n")
    
    def plot(self, figsize=(14, 10)):
        """
        Visualize tomography results.
        
        Creates four subplots:
        1. Density matrix (real part)
        2. Density matrix (imaginary part)
        3. State populations (diagonal elements)
        4. Eigenvalue spectrum
        """
        if self.density_matrix is None:
            print("Error: No density matrix to plot. Run tomography first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Density matrix (real part)
        rho_real = np.real(self.density_matrix.data)
        im1 = axes[0, 0].imshow(rho_real, cmap='RdBu', 
                                vmin=-np.max(np.abs(rho_real)), 
                                vmax=np.max(np.abs(rho_real)))
        axes[0, 0].set_title('Density Matrix (Real Part)', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Basis State')
        axes[0, 0].set_ylabel('Basis State')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Density matrix (imaginary part)
        rho_imag = np.imag(self.density_matrix.data)
        im2 = axes[0, 1].imshow(rho_imag, cmap='RdBu',
                                vmin=-np.max(np.abs(rho_imag)), 
                                vmax=np.max(np.abs(rho_imag)))
        axes[0, 1].set_title('Density Matrix (Imaginary Part)', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Basis State')
        axes[0, 1].set_ylabel('Basis State')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot 3: State populations (diagonal elements)
        populations = np.real(np.diag(self.density_matrix.data))
        axes[1, 0].bar(range(len(populations)), populations, color='steelblue', alpha=0.7)
        axes[1, 0].set_title('State Populations (Diagonal Elements)', fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('Basis State')
        axes[1, 0].set_ylabel('Population')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Eigenvalue spectrum
        eigenvalues = np.linalg.eigvalsh(self.density_matrix.data)
        eigenvalues = np.sort(eigenvalues)[::-1]
        axes[1, 1].bar(range(len(eigenvalues)), eigenvalues, color='coral', alpha=0.7)
        axes[1, 1].set_title('Eigenvalue Spectrum', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Eigenvalue Index')
        axes[1, 1].set_ylabel('Eigenvalue')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_measurement_samples(self, num_bases=3, num_outcomes=5):
        """
        Print sample measurements from tomography.
        
        Parameters:
        -----------
        num_bases : int
            Number of bases to show
        num_outcomes : int
            Number of measurement outcomes to show per basis
        """
        if not self.tomography_measurements:
            print("Error: No measurement data. Run tomography first.")
            return
        
        print(f"\n{'='*70}")
        print("SAMPLE MEASUREMENT OUTCOMES")
        print(f"{'='*70}")
        
        for i, (basis_name, counts) in enumerate(list(self.tomography_measurements.items())[:num_bases]):
            print(f"\nBasis {i+1}: {basis_name}")
            print(f"  Total outcomes: {sum(counts.values())}")
            print(f"  Sample outcomes (first {num_outcomes}):")
            
            for j, (bitstring, count) in enumerate(list(counts.items())[:num_outcomes]):
                percentage = 100 * count / self.shots
                print(f"    |{bitstring}⟩: {count:4d} times ({percentage:5.1f}%)")
        
        print(f"\n  ... and {len(self.tomography_measurements) - num_bases} more bases")
        print(f"{'='*70}\n")
    
    def get_density_matrix(self):
        """Get the reconstructed density matrix."""
        return self.density_matrix
    
    def get_reconstructed_state(self):
        """Get the reconstructed state vector."""
        return self.reconstructed_state
    
    def get_exact_state(self):
        """Get the exact evolved state (if computed)."""
        return self.evolved_state


# =============================================================================
# EXAMPLE FUNCTIONS
# =============================================================================

def example_bell_state():
    """
    Example 1: Evolve a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Bell State Evolution")
    print("="*70)
    
    # Define initial Bell state: (|00⟩ + |11⟩)/√2
    initial_state = [1, 0, 0, 1]  # Will be automatically normalized
    
    # Number of copies
    num_copies = 2
    
    # Create a simple Hamiltonian for 4 qubits (2 copies × 2 qubits)
    dim = 2 ** (2 * num_copies)  # 2^4 = 16
    hamiltonian = np.zeros((dim, dim), dtype=complex)
    
    # Add diagonal elements (on-site energies)
    for i in range(dim):
        hamiltonian[i, i] = 0.1 * i
    
    # Add nearest-neighbor hopping
    for i in range(dim - 1):
        hamiltonian[i, i+1] = 0.3
        hamiltonian[i+1, i] = 0.3
    
    # Evolution time
    evolution_time = 1.0
    
    # Create evolver and run
    evolver = QuantumStateEvolver(
        initial_state=initial_state,
        num_copies=num_copies,
        hamiltonian=hamiltonian,
        evolution_time=evolution_time,
        shots=1000
    )
    
    evolver.run(compute_exact_state=True)
    evolver.print_measurement_samples()
    evolver.plot()
    
    return evolver


def example_ghz_state():
    """
    Example 2: Evolve a GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: GHZ State Evolution")
    print("="*70)
    
    # Define initial GHZ state: (|000⟩ + |111⟩)/√2
    initial_state = [1, 0, 0, 0, 0, 0, 0, 1]  # Will be automatically normalized
    
    # Number of copies
    num_copies = 1
    
    # Create Hamiltonian for 3 qubits
    dim = 2 ** (3 * num_copies)  # 2^3 = 8
    
    # Create a simple Hermitian Hamiltonian
    np.random.seed(42)
    H_random = np.random.randn(dim, dim)
    hamiltonian = (H_random + H_random.T) / 2 * 0.5  # Real symmetric
    
    # Evolution time
    evolution_time = 0.5
    
    # Create evolver and run
    evolver = QuantumStateEvolver(
        initial_state=initial_state,
        num_copies=num_copies,
        hamiltonian=hamiltonian,
        evolution_time=evolution_time,
        shots=2000
    )
    
    evolver.run(compute_exact_state=True)
    evolver.print_measurement_samples()
    evolver.plot()
    
    return evolver


def example_custom_state():
    """
    Example 3: Custom superposition state
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom State Evolution")
    print("="*70)
    
    # Custom superposition: 0.6|00⟩ + 0.8|11⟩
    initial_state = [0.6, 0, 0, 0.8]
    num_copies = 1
    
    # Simple diagonal Hamiltonian
    dim = 4
    hamiltonian = np.diag([0.0, 0.5, 0.5, 1.0])
    
    evolution_time = 2.0
    
    # Run
    evolver = QuantumStateEvolver(
        initial_state=initial_state,
        num_copies=num_copies,
        hamiltonian=hamiltonian,
        evolution_time=evolution_time,
        shots=1500
    )
    
    evolver.run(compute_exact_state=True)
    evolver.print_measurement_samples()
    evolver.plot()
    
    return evolver


def example_simple_two_qubit():
    """
    Example 4: Very simple 2-qubit system
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Simple 2-Qubit Evolution")
    print("="*70)
    
    # Simple product state |01⟩
    initial_state = [0, 1, 0, 0]
    
    # Single copy
    num_copies = 1
    
    # Simple Hamiltonian (σ_x ⊗ σ_x)
    X = np.array([[0, 1], [1, 0]])
    hamiltonian = np.kron(X, X) * 0.5
    
    evolution_time = 1.0
    
    # Run
    evolver = QuantumStateEvolver(
        initial_state=initial_state,
        num_copies=num_copies,
        hamiltonian=hamiltonian,
        evolution_time=evolution_time,
        shots=1000
    )
    
    evolver.run(compute_exact_state=True)
    evolver.print_measurement_samples()
    evolver.plot()
    
    return evolver


def example_with_detailed_tracking():
    """
    Example 5: With detailed tomography tracking and analysis
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Detailed Tomography Analysis")
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
    
    # Create evolver
    evolver = QuantumStateEvolver(
        initial_state=initial_state,
        num_copies=num_copies,
        hamiltonian=hamiltonian,
        evolution_time=evolution_time,
        shots=1000
    )
    
    # Run with tomography
    evolver.run(compute_exact_state=True)
    
    # Show detailed measurement data
    evolver.print_measurement_samples(num_bases=6, num_outcomes=8)
    
    # Get and print additional info
    print("\nAdditional Analysis:")
    print(f"  • Density matrix shape: {evolver.density_matrix.data.shape}")
    print(f"  • Reconstructed state shape: {evolver.reconstructed_state.data.shape}")
    print(f"  • Number of measurement bases used: {evolver.num_measurement_bases}")
    
    # Plot results
    evolver.plot()
    
    return evolver


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUANTUM STATE EVOLUTION AND TOMOGRAPHY")
    print("Complete Implementation")
    print("="*70)
    
    print("\nAvailable Examples:")
    print("  1. Bell state evolution (4 qubits, 2 copies)")
    print("  2. GHZ state evolution (3 qubits, 1 copy)")
    print("  3. Custom state evolution (2 qubits)")
    print("  4. Simple 2-qubit system")
    print("  5. Detailed tomography tracking")
    
    choice = input("\nEnter choice (1-5, or press Enter for option 1): ").strip()
    
    if choice == '2':
        evolver = example_ghz_state()
    elif choice == '3':
        evolver = example_custom_state()
    elif choice == '4':
        evolver = example_simple_two_qubit()
    elif choice == '5':
        evolver = example_with_detailed_tracking()
    else:
        evolver = example_bell_state()
    
    print("\n✓ Program completed successfully!")
    print("\nAccess results via:")
    print("  • evolver.density_matrix - Reconstructed density matrix")
    print("  • evolver.reconstructed_state - Reconstructed state vector")
    print("  • evolver.fidelity - Fidelity with exact state")
    print("  • evolver.purity - Purity of the state")
    print("  • evolver.tomography_measurements - All measurement data")
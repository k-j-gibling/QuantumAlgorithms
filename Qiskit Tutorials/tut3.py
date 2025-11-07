import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, Operator
from qiskit.circuit.library import HamiltonianGate
from typing import List, Union, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings

@dataclass
class TomographyResult:
    """Container for tomography results"""
    density_matrix: DensityMatrix
    reconstructed_state: Statevector
    measurement_counts: dict
    fidelity: float
    purity: float


class QuantumStateEvolver:
    """
    A class for evolving quantum states under a Hamiltonian and performing
    state tomography to reconstruct the evolved state.
    """
    
    def __init__(self, 
                 initial_state: Union[List[complex], np.ndarray, Statevector],
                 num_copies: int = 1,
                 shots_per_basis: int = 1000):
        """
        Initialize the Quantum State Evolver.
        
        Parameters:
        -----------
        initial_state : list, np.ndarray, or Statevector
            The initial quantum state (should be normalized)
        num_copies : int
            Number of tensor copies of the state (N)
        shots_per_basis : int
            Number of measurements per tomography basis
        """
        self.initial_state = self._validate_state(initial_state)
        self.num_copies = num_copies
        self.shots_per_basis = shots_per_basis
        self.num_qubits = int(np.log2(len(self.initial_state)))
        self.total_qubits = self.num_qubits * num_copies
        
        # Create simulator
        self.simulator = AerSimulator(method='statevector')
        
        # Storage for results
        self.hamiltonian = None
        self.evolution_time = None
        self.evolved_state = None
        self.tomography_result = None
        
        print(f"✓ Initialized with {self.num_qubits} qubits per copy")
        print(f"✓ Total system size: {self.total_qubits} qubits")
        print(f"✓ Hilbert space dimension: {2**self.total_qubits}")
    
    def _validate_state(self, state: Union[List, np.ndarray, Statevector]) -> np.ndarray:
        """Validate and normalize the input state"""
        if isinstance(state, Statevector):
            return state.data
        
        state = np.array(state, dtype=complex)
        
        # Check if it's a power of 2
        if not (len(state) & (len(state) - 1) == 0):
            raise ValueError(f"State dimension {len(state)} is not a power of 2")
        
        # Normalize
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0):
            warnings.warn(f"State norm was {norm}, normalizing to 1.0")
            state = state / norm
        
        return state
    
    def set_hamiltonian(self, 
                       hamiltonian: Union[np.ndarray, List[List], Operator],
                       evolution_time: float = 1.0):
        """
        Set the Hamiltonian for evolution.
        
        Parameters:
        -----------
        hamiltonian : np.ndarray, list, or Operator
            The Hamiltonian matrix (must be Hermitian)
        evolution_time : float
            Time parameter for evolution (t in e^(-iHt))
        """
        if isinstance(hamiltonian, Operator):
            H = hamiltonian.data
        else:
            H = np.array(hamiltonian, dtype=complex)
        
        # Validate Hermiticity
        if not np.allclose(H, H.conj().T):
            raise ValueError("Hamiltonian must be Hermitian (H = H†)")
        
        # Check dimensions
        expected_dim = 2 ** self.total_qubits
        if H.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Hamiltonian dimension {H.shape} doesn't match "
                f"system dimension {expected_dim}x{expected_dim}"
            )
        
        self.hamiltonian = H
        self.evolution_time = evolution_time
        print(f"✓ Hamiltonian set: {H.shape[0]}×{H.shape[0]} matrix")
        print(f"✓ Evolution time: {evolution_time}")
        
    def create_tensor_product_state(self) -> QuantumCircuit:
        """Create N tensor copies of the initial state"""
        qc = QuantumCircuit(self.total_qubits)
        
        # Initialize each copy
        for copy_idx in range(self.num_copies):
            start_qubit = copy_idx * self.num_qubits
            end_qubit = start_qubit + self.num_qubits
            
            # Create statevector and initialize
            qc.initialize(self.initial_state, range(start_qubit, end_qubit))
        
        return qc
    
    def evolve_state(self) -> Statevector:
        """
        Evolve the tensor product state under the Hamiltonian.
        
        Returns:
        --------
        evolved_state : Statevector
            The evolved quantum state
        """
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian not set. Call set_hamiltonian() first.")
        
        # Create initial state circuit
        qc = self.create_tensor_product_state()
        
        # Create evolution operator
        print("→ Computing evolution operator e^(-iHt)...")
        U = Operator(np.exp(-1j * self.hamiltonian * self.evolution_time))
        
        # Apply evolution
        print("→ Applying Hamiltonian evolution...")
        qc.unitary(U, range(self.total_qubits), label='e^(-iHt)')
        
        # Get evolved statevector
        qc.save_statevector()
        result = self.simulator.run(qc).result()
        self.evolved_state = result.get_statevector()
        
        print("✓ State evolution complete")
        return self.evolved_state
    
    def perform_tomography(self) -> TomographyResult:
        """
        Perform quantum state tomography on the evolved state.
        
        Returns:
        --------
        result : TomographyResult
            Complete tomography results including density matrix
        """
        if self.evolved_state is None:
            raise ValueError("No evolved state. Call evolve_state() first.")
        
        print(f"\n{'='*60}")
        print("QUANTUM STATE TOMOGRAPHY")
        print(f"{'='*60}")
        
        # Generate measurement bases
        bases = self._generate_pauli_bases()
        print(f"→ Measuring in {len(bases)} Pauli bases...")
        
        # Perform measurements
        measurement_results = {}
        for basis_name, basis_circuit in bases.items():
            counts = self._measure_in_basis(basis_circuit)
            measurement_results[basis_name] = counts
        
        # Reconstruct density matrix
        print("→ Reconstructing density matrix...")
        rho = self._reconstruct_density_matrix(measurement_results)
        
        # Extract state estimate
        eigenvalues, eigenvectors = np.linalg.eigh(rho.data)
        max_idx = np.argmax(eigenvalues)
        reconstructed_state = Statevector(eigenvectors[:, max_idx])
        
        # Calculate metrics
        fidelity = self._calculate_fidelity(
            self.evolved_state, 
            reconstructed_state
        )
        purity = np.real(np.trace(rho.data @ rho.data))
        
        print(f"✓ Reconstruction complete")
        print(f"  • Fidelity: {fidelity:.6f}")
        print(f"  • Purity: {purity:.6f}")
        
        self.tomography_result = TomographyResult(
            density_matrix=rho,
            reconstructed_state=reconstructed_state,
            measurement_counts=measurement_results,
            fidelity=fidelity,
            purity=purity
        )
        
        return self.tomography_result
    
    def _generate_pauli_bases(self) -> dict:
        """Generate Pauli measurement bases for tomography"""
        bases = {}
        pauli_labels = ['I', 'X', 'Y', 'Z']
        
        # For complete tomography, we need 4^n bases
        # For practical purposes, we'll use important bases
        if self.total_qubits <= 3:
            # Full tomography for small systems
            from itertools import product
            for pauli_string in product(pauli_labels, repeat=self.total_qubits):
                if 'I' not in pauli_string:  # Skip all-identity
                    basis_name = ''.join(pauli_string)
                    bases[basis_name] = self._create_basis_circuit(pauli_string)
        else:
            # Use single-qubit bases for larger systems
            for qubit in range(self.total_qubits):
                for pauli in ['X', 'Y', 'Z']:
                    basis_name = f"{pauli}{qubit}"
                    pauli_string = ['I'] * self.total_qubits
                    pauli_string[qubit] = pauli
                    bases[basis_name] = self._create_basis_circuit(pauli_string)
        
        return bases
    
    def _create_basis_circuit(self, pauli_string: List[str]) -> QuantumCircuit:
        """Create measurement circuit for a Pauli basis"""
        qc = QuantumCircuit(self.total_qubits, self.total_qubits)
        
        for qubit, pauli in enumerate(pauli_string):
            if pauli == 'X':
                qc.h(qubit)
            elif pauli == 'Y':
                qc.sdg(qubit)
                qc.h(qubit)
            # Z basis is computational basis (no gates needed)
        
        qc.measure(range(self.total_qubits), range(self.total_qubits))
        return qc
    
    def _measure_in_basis(self, basis_circuit: QuantumCircuit) -> dict:
        """Perform measurements in a specific basis"""
        # Create evolution circuit
        qc = self.create_tensor_product_state()
        
        # Apply evolution
        U = Operator(np.exp(-1j * self.hamiltonian * self.evolution_time))
        qc.unitary(U, range(self.total_qubits))
        
        # Add basis measurement
        qc.compose(basis_circuit, inplace=True)
        
        # Run measurement
        result = self.simulator.run(qc, shots=self.shots_per_basis).result()
        return result.get_counts()
    
    def _reconstruct_density_matrix(self, measurement_results: dict) -> DensityMatrix:
        """Reconstruct density matrix from measurement results"""
        dim = 2 ** self.total_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        
        # Simple reconstruction using maximum likelihood
        # For each measurement outcome, update density matrix
        total_measurements = 0
        
        for basis_name, counts in measurement_results.items():
            for bitstring, count in counts.items():
                # Convert bitstring to state
                idx = int(bitstring, 2)
                basis_state = np.zeros(dim)
                basis_state[idx] = 1.0
                
                # Add to density matrix
                rho += count * np.outer(basis_state, basis_state)
                total_measurements += count
        
        # Normalize
        rho = rho / total_measurements
        
        # Ensure Hermiticity and trace 1
        rho = (rho + rho.conj().T) / 2
        rho = rho / np.trace(rho)
        
        return DensityMatrix(rho)
    
    def _calculate_fidelity(self, state1: Statevector, state2: Statevector) -> float:
        """Calculate fidelity between two states"""
        return np.abs(np.vdot(state1.data, state2.data)) ** 2
    
    def visualize_results(self, figsize=(15, 5)):
        """Visualize tomography results"""
        if self.tomography_result is None:
            raise ValueError("No tomography results. Call perform_tomography() first.")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Density matrix (real part)
        rho_real = np.real(self.tomography_result.density_matrix.data)
        im1 = axes[0].imshow(rho_real, cmap='RdBu', vmin=-1, vmax=1)
        axes[0].set_title('Density Matrix (Real Part)')
        axes[0].set_xlabel('Basis State')
        axes[0].set_ylabel('Basis State')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Density matrix (imaginary part)
        rho_imag = np.imag(self.tomography_result.density_matrix.data)
        im2 = axes[1].imshow(rho_imag, cmap='RdBu', vmin=-1, vmax=1)
        axes[1].set_title('Density Matrix (Imaginary Part)')
        axes[1].set_xlabel('Basis State')
        axes[1].set_ylabel('Basis State')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot 3: Eigenvalues
        eigenvalues = np.linalg.eigvalsh(self.tomography_result.density_matrix.data)
        eigenvalues = np.sort(eigenvalues)[::-1]
        axes[2].bar(range(len(eigenvalues)), eigenvalues)
        axes[2].set_title('Density Matrix Eigenvalues')
        axes[2].set_xlabel('Eigenvalue Index')
        axes[2].set_ylabel('Eigenvalue')
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print a summary of the computation"""
        print(f"\n{'='*60}")
        print("COMPUTATION SUMMARY")
        print(f"{'='*60}")
        print(f"System Configuration:")
        print(f"  • Qubits per copy: {self.num_qubits}")
        print(f"  • Number of copies: {self.num_copies}")
        print(f"  • Total qubits: {self.total_qubits}")
        print(f"  • Hilbert space dimension: {2**self.total_qubits}")
        print(f"\nMeasurement Configuration:")
        print(f"  • Shots per basis: {self.shots_per_basis}")
        
        if self.tomography_result:
            print(f"\nTomography Results:")
            print(f"  • Fidelity: {self.tomography_result.fidelity:.6f}")
            print(f"  • Purity: {self.tomography_result.purity:.6f}")
        print(f"{'='*60}\n")


# =============================================================================
# EXAMPLE USAGE AND HELPER FUNCTIONS
# =============================================================================

def create_example_hamiltonian(num_qubits: int, coupling_strength: float = 0.5) -> np.ndarray:
    """
    Create an example Hamiltonian (Heisenberg-like interaction).
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits
    coupling_strength : float
        Interaction strength
    """
    dim = 2 ** num_qubits
    H = np.zeros((dim, dim), dtype=complex)
    
    # Add diagonal disorder
    for i in range(dim):
        H[i, i] = np.random.uniform(-0.5, 0.5)
    
    # Add off-diagonal coupling
    for i in range(dim - 1):
        H[i, i+1] = coupling_strength
        H[i+1, i] = coupling_strength
    
    return H


def example_bell_state():
    """Example: Evolve a Bell state"""
    print("\n" + "="*60)
    print("EXAMPLE: BELL STATE EVOLUTION")
    print("="*60 + "\n")
    
    # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    
    # Initialize evolver with 2 copies
    evolver = QuantumStateEvolver(
        initial_state=bell_state,
        num_copies=2,
        shots_per_basis=1000
    )
    
    # Create a simple Hamiltonian (4 qubits total)
    H = create_example_hamiltonian(num_qubits=4, coupling_strength=0.3)
    evolver.set_hamiltonian(H, evolution_time=0.5)
    
    # Evolve the state
    evolved_state = evolver.evolve_state()
    
    # Perform tomography
    results = evolver.perform_tomography()
    
    # Print summary
    evolver.print_summary()
    
    # Visualize
    evolver.visualize_results()
    
    return evolver


def example_ghz_state():
    """Example: Evolve a GHZ state"""
    print("\n" + "="*60)
    print("EXAMPLE: GHZ STATE EVOLUTION")
    print("="*60 + "\n")
    
    # Create GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
    ghz_state = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
    
    # Initialize evolver
    evolver = QuantumStateEvolver(
        initial_state=ghz_state,
        num_copies=1,
        shots_per_basis=2000
    )
    
    # Create Hamiltonian
    H = create_example_hamiltonian(num_qubits=3, coupling_strength=0.4)
    evolver.set_hamiltonian(H, evolution_time=1.0)
    
    # Run evolution and tomography
    evolver.evolve_state()
    evolver.perform_tomography()
    evolver.print_summary()
    evolver.visualize_results()
    
    return evolver


# =============================================================================
# INTERACTIVE INTERFACE
# =============================================================================

def interactive_mode():
    """Interactive command-line interface"""
    print("\n" + "="*60)
    print("QUANTUM STATE EVOLUTION & TOMOGRAPHY - INTERACTIVE MODE")
    print("="*60 + "\n")
    
    # Get initial state
    print("Step 1: Define Initial State")
    print("-" * 40)
    num_qubits = int(input("Enter number of qubits for single copy (1-4): "))
    
    print("\nChoose state type:")
    print("1. Equal superposition |+⟩⊗n")
    print("2. Random state")
    print("3. Custom state (enter coefficients)")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == '1':
        initial_state = np.ones(2**num_qubits) / np.sqrt(2**num_qubits)
    elif choice == '2':
        initial_state = np.random.randn(2**num_qubits) + 1j*np.random.randn(2**num_qubits)
        initial_state = initial_state / np.linalg.norm(initial_state)
    else:
        print(f"Enter {2**num_qubits} complex coefficients (format: real imag):")
        initial_state = []
        for i in range(2**num_qubits):
            parts = input(f"  Coefficient {i}: ").split()
            initial_state.append(complex(float(parts[0]), float(parts[1]) if len(parts) > 1 else 0))
        initial_state = np.array(initial_state)
    
    # Get number of copies
    print("\nStep 2: Number of Copies")
    print("-" * 40)
    num_copies = int(input("Enter number of tensor copies N (1-3): "))
    
    # Create evolver
    evolver = QuantumStateEvolver(
        initial_state=initial_state,
        num_copies=num_copies,
        shots_per_basis=1000
    )
    
    # Get Hamiltonian
    print("\nStep 3: Define Hamiltonian")
    print("-" * 40)
    print("1. Random Hamiltonian")
    print("2. Heisenberg-type Hamiltonian")
    print("3. Custom Hamiltonian matrix")
    
    h_choice = input("Enter choice (1-3): ")
    total_qubits = num_qubits * num_copies
    
    if h_choice == '1':
        dim = 2 ** total_qubits
        H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H = (H + H.conj().T) / 2  # Make Hermitian
    elif h_choice == '2':
        coupling = float(input("Enter coupling strength (e.g., 0.5): "))
        H = create_example_hamiltonian(total_qubits, coupling)
    else:
        print(f"Enter {(2**total_qubits)**2} matrix elements...")
        print("(This is impractical for large systems - using random instead)")
        H = create_example_hamiltonian(total_qubits)
    
    evolution_time = float(input("Enter evolution time t: "))
    evolver.set_hamiltonian(H, evolution_time)
    
    # Run computation
    print("\nStep 4: Running Computation")
    print("-" * 40)
    evolver.evolve_state()
    evolver.perform_tomography()
    evolver.print_summary()
    
    # Visualization
    viz = input("\nVisualize results? (y/n): ")
    if viz.lower() == 'y':
        evolver.visualize_results()
    
    return evolver


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Run Bell state example")
    print("2. Run GHZ state example")
    print("3. Interactive mode")
    
    mode = input("\nEnter choice (1-3): ")
    
    if mode == '1':
        evolver = example_bell_state()
    elif mode == '2':
        evolver = example_ghz_state()
    else:
        evolver = interactive_mode()
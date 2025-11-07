import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, Operator
import matplotlib.pyplot as plt


class QuantumStateEvolver:
    """
    Simple quantum state evolver with tomography.
    Just input state, copies, Hamiltonian, and run.
    """
    
    def __init__(self, initial_state, num_copies, hamiltonian, evolution_time, shots=1000):
        """
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
            Measurements per basis
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
        
        # Make sure Hamiltonian is Hermitian
        self.hamiltonian = (self.hamiltonian + self.hamiltonian.conj().T) / 2
        
        # Simulator
        self.simulator = AerSimulator(method='statevector')
        
        # Results storage
        self.evolved_state = None
        self.density_matrix = None
        self.fidelity = None
    
    def run(self):
        """Run the full evolution and tomography process"""
        print(f"\n{'='*60}")
        print("QUANTUM STATE EVOLUTION")
        print(f"{'='*60}")
        print(f"Initial state dimension: {len(self.initial_state)}")
        print(f"Number of copies: {self.num_copies}")
        print(f"Total qubits: {self.total_qubits}")
        print(f"Hilbert space dimension: {2**self.total_qubits}")
        
        # Step 1: Evolve state
        print("\n→ Evolving state under Hamiltonian...")
        self._evolve_state()
        
        # Step 2: Perform tomography
        print("→ Performing quantum state tomography...")
        self._perform_tomography()
        
        # Step 3: Calculate metrics
        print("→ Calculating fidelity...")
        self._calculate_metrics()
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Fidelity: {self.fidelity:.6f}")
        print(f"Purity: {self.purity:.6f}")
        print(f"{'='*60}\n")
        
        return self
    
    def _evolve_state(self):
        """Evolve the state under the Hamiltonian"""
        # Create circuit with tensor product state
        qc = QuantumCircuit(self.total_qubits)
        
        # Initialize each copy
        for i in range(self.num_copies):
            start = i * self.num_qubits
            end = start + self.num_qubits
            qc.initialize(self.initial_state, range(start, end))
        
        # Apply evolution
        U = Operator(np.exp(-1j * self.hamiltonian * self.evolution_time))
        qc.unitary(U, range(self.total_qubits))
        
        # Get evolved state
        qc.save_statevector()
        result = self.simulator.run(qc).result()
        self.evolved_state = result.get_statevector()
    
    def _perform_tomography(self):
        """Perform quantum state tomography"""
        # Generate Pauli measurement bases
        bases = self._get_pauli_bases()
        
        # Measure in each basis
        dim = 2 ** self.total_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        total_counts = 0
        
        for basis_circuit in bases:
            counts = self._measure_in_basis(basis_circuit)
            
            # Accumulate density matrix
            for bitstring, count in counts.items():
                idx = int(bitstring.replace(' ', ''), 2)
                basis_state = np.zeros(dim)
                basis_state[idx] = 1.0
                rho += count * np.outer(basis_state, basis_state)
                total_counts += count
        
        # Normalize and ensure proper density matrix
        rho = rho / total_counts
        rho = (rho + rho.conj().T) / 2  # Hermiticity
        rho = rho / np.trace(rho)  # Trace 1
        
        self.density_matrix = DensityMatrix(rho)
    
    def _get_pauli_bases(self):
        """Generate Pauli measurement bases"""
        bases = []
        
        # For small systems, use all single-qubit Pauli bases
        for qubit in range(self.total_qubits):
            for pauli in ['X', 'Y', 'Z']:
                qc = QuantumCircuit(self.total_qubits, self.total_qubits)
                
                if pauli == 'X':
                    qc.h(qubit)
                elif pauli == 'Y':
                    qc.sdg(qubit)
                    qc.h(qubit)
                
                qc.measure(range(self.total_qubits), range(self.total_qubits))
                bases.append(qc)
        
        return bases
    
    def _measure_in_basis(self, basis_circuit):
        """Measure evolved state in a specific basis"""
        # Create full circuit
        qc = QuantumCircuit(self.total_qubits, self.total_qubits)
        
        # Initialize
        for i in range(self.num_copies):
            start = i * self.num_qubits
            end = start + self.num_qubits
            qc.initialize(self.initial_state, range(start, end))
        
        # Evolve
        U = Operator(np.exp(-1j * self.hamiltonian * self.evolution_time))
        qc.unitary(U, range(self.total_qubits))
        
        # Measure in basis
        qc.compose(basis_circuit, inplace=True)
        
        result = self.simulator.run(qc, shots=self.shots).result()
        return result.get_counts()
    
    def _calculate_metrics(self):
        """Calculate fidelity and purity"""
        # Extract state from density matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self.density_matrix.data)
        max_idx = np.argmax(eigenvalues)
        reconstructed_state = Statevector(eigenvectors[:, max_idx])
        
        # Fidelity
        self.fidelity = np.abs(np.vdot(self.evolved_state.data, reconstructed_state.data)) ** 2
        
        # Purity
        self.purity = np.real(np.trace(self.density_matrix.data @ self.density_matrix.data))
    
    def plot(self):
        """Visualize results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Density matrix real part
        rho_real = np.real(self.density_matrix.data)
        im1 = axes[0].imshow(rho_real, cmap='RdBu', vmin=-1, vmax=1)
        axes[0].set_title('Density Matrix (Real)', fontweight='bold')
        axes[0].set_xlabel('Basis State')
        axes[0].set_ylabel('Basis State')
        plt.colorbar(im1, ax=axes[0])
        
        # Density matrix imaginary part
        rho_imag = np.imag(self.density_matrix.data)
        im2 = axes[1].imshow(rho_imag, cmap='RdBu', vmin=-1, vmax=1)
        axes[1].set_title('Density Matrix (Imaginary)', fontweight='bold')
        axes[1].set_xlabel('Basis State')
        axes[1].set_ylabel('Basis State')
        plt.colorbar(im2, ax=axes[1])
        
        # Populations
        populations = np.real(np.diag(self.density_matrix.data))
        axes[2].bar(range(len(populations)), populations, color='steelblue')
        axes[2].set_title('State Populations', fontweight='bold')
        axes[2].set_xlabel('Basis State')
        axes[2].set_ylabel('Population')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# EXAMPLE: Bell State Evolution
# =============================================================================

def example_bell_state():
    """
    Example: Evolve a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    """
    print("\n" + "="*60)
    print("EXAMPLE: Bell State Evolution")
    print("="*60)
    
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
        hamiltonian[i, i+1] = 0.5
        hamiltonian[i+1, i] = 0.5
    
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
    
    evolver.run()
    evolver.plot()
    
    return evolver


# =============================================================================
# EXAMPLE: GHZ State Evolution
# =============================================================================

def example_ghz_state():
    """
    Example: Evolve a GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
    """
    print("\n" + "="*60)
    print("EXAMPLE: GHZ State Evolution")
    print("="*60)
    
    # Define initial GHZ state: (|000⟩ + |111⟩)/√2
    initial_state = [1, 0, 0, 0, 0, 0, 0, 1]  # Will be automatically normalized
    
    # Number of copies
    num_copies = 1
    
    # Create Hamiltonian for 3 qubits
    dim = 2 ** (3 * num_copies)  # 2^3 = 8
    hamiltonian = np.zeros((dim, dim), dtype=complex)
    
    # Add random Hermitian elements
    np.random.seed(42)
    H_random = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    hamiltonian = (H_random + H_random.conj().T) / 2 * 0.3
    
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
    
    evolver.run()
    evolver.plot()
    
    return evolver


# =============================================================================
# EXAMPLE: Custom State with User Input
# =============================================================================

def example_custom_state():
    """
    Example: User provides all inputs
    """
    print("\n" + "="*60)
    print("EXAMPLE: Custom State Evolution")
    print("="*60)
    
    # User inputs
    initial_state = [0.6, 0, 0, 0.8]  # Custom superposition
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
    
    evolver.run()
    evolver.plot()
    
    return evolver


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run the Bell state example
    evolver = example_bell_state()
    
    # You can also run the GHZ example
    # evolver = example_ghz_state()
    
    # Or the custom example
    # evolver = example_custom_state()
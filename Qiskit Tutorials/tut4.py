import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, Operator
from typing import List, Union, Optional, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json

@dataclass
class TomographyResult:
    """Container for tomography results"""
    density_matrix: DensityMatrix
    reconstructed_state: Statevector
    measurement_counts: dict
    fidelity: float
    purity: float
    trace: float


class StateInputParser:
    """Helper class for parsing various state input formats"""
    
    @staticmethod
    def parse_state(state_input: Union[str, List, np.ndarray]) -> np.ndarray:
        """
        Parse state from multiple input formats.
        
        Supported formats:
        - List/array: [1, 0, 0, 1]
        - String (basis): "|00⟩ + |11⟩"
        - String (coefficients): "1 0 0 1"
        - Dictionary: {"00": 1, "11": 1}
        """
        if isinstance(state_input, np.ndarray):
            return state_input
        
        if isinstance(state_input, list):
            return np.array(state_input, dtype=complex)
        
        if isinstance(state_input, str):
            # Try parsing as basis state notation
            if '|' in state_input:
                return StateInputParser._parse_ket_notation(state_input)
            else:
                # Try parsing as space-separated coefficients
                return StateInputParser._parse_coefficient_string(state_input)
        
        if isinstance(state_input, dict):
            return StateInputParser._parse_dict_notation(state_input)
        
        raise ValueError(f"Unsupported state input format: {type(state_input)}")
    
    @staticmethod
    def _parse_ket_notation(state_str: str) -> np.ndarray:
        """Parse state from ket notation like '|00⟩ + |11⟩' or '|0⟩ + |1⟩'"""
        import re
        
        # Remove spaces
        state_str = state_str.replace(' ', '')
        
        # Find all ket patterns with optional coefficients
        # Matches: coeff|bits⟩ or |bits⟩
        pattern = r'([+-]?\d*\.?\d*(?:[eE][+-]?\d+)?(?:[+-]\d*\.?\d*(?:[eE][+-]?\d+)?[ij])?)\|(\d+)[⟩>]'
        matches = re.findall(pattern, state_str)
        
        if not matches:
            raise ValueError(f"Could not parse ket notation: {state_str}")
        
        # Determine number of qubits from longest bitstring
        max_bits = max(len(bits) for _, bits in matches)
        dim = 2 ** max_bits
        
        state = np.zeros(dim, dtype=complex)
        
        for coeff_str, bits in matches:
            # Parse coefficient
            if coeff_str in ['', '+']:
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                # Handle complex numbers
                coeff_str = coeff_str.replace('i', 'j').replace('I', 'j')
                coeff = complex(coeff_str) if 'j' in coeff_str else float(coeff_str)
            
            # Parse basis state
            idx = int(bits, 2)
            state[idx] = coeff
        
        return state
    
    @staticmethod
    def _parse_coefficient_string(coeff_str: str) -> np.ndarray:
        """Parse state from space-separated coefficients"""
        parts = coeff_str.strip().split()
        state = []
        
        i = 0
        while i < len(parts):
            part = parts[i].replace('i', 'j').replace('I', 'j')
            
            # Check if next part is imaginary continuation
            if i + 1 < len(parts) and ('j' in parts[i+1] or 'i' in parts[i+1]):
                imag_part = parts[i+1].replace('i', 'j').replace('I', 'j')
                coeff = complex(part + imag_part)
                i += 2
            else:
                coeff = complex(part) if 'j' in part else float(part)
                i += 1
            
            state.append(coeff)
        
        return np.array(state, dtype=complex)
    
    @staticmethod
    def _parse_dict_notation(state_dict: dict) -> np.ndarray:
        """Parse state from dictionary like {"00": 1, "11": 1}"""
        # Find maximum index to determine dimension
        max_idx = 0
        for key in state_dict.keys():
            if isinstance(key, str):
                idx = int(key, 2)
            else:
                idx = int(key)
            max_idx = max(max_idx, idx)
        
        dim = 2 ** int(np.ceil(np.log2(max_idx + 1)))
        state = np.zeros(dim, dtype=complex)
        
        for key, value in state_dict.items():
            if isinstance(key, str):
                idx = int(key, 2)
            else:
                idx = int(key)
            state[idx] = complex(value)
        
        return state


class HamiltonianBuilder:
    """Helper class for building Hamiltonians"""
    
    @staticmethod
    def from_matrix(matrix: Union[List, np.ndarray]) -> np.ndarray:
        """Create Hamiltonian from matrix input"""
        H = np.array(matrix, dtype=complex)
        
        # Validate Hermiticity
        if not np.allclose(H, H.conj().T):
            print("Warning: Input matrix is not Hermitian. Symmetrizing: H = (H + H†)/2")
            H = (H + H.conj().T) / 2
        
        return H
    
    @staticmethod
    def random_hermitian(dim: int, strength: float = 1.0) -> np.ndarray:
        """Generate random Hermitian matrix"""
        H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H = (H + H.conj().T) / 2
        return H * strength
    
    @staticmethod
    def nearest_neighbor(num_qubits: int, coupling: float = 1.0) -> np.ndarray:
        """Create nearest-neighbor interaction Hamiltonian"""
        dim = 2 ** num_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        # Add nearest-neighbor hopping
        for i in range(dim - 1):
            if bin(i ^ (i+1)).count('1') == 1:  # Differ by one bit
                H[i, i+1] = coupling
                H[i+1, i] = coupling
        
        return H
    
    @staticmethod
    def pauli_string(num_qubits: int, pauli_terms: List[Tuple[str, float]]) -> np.ndarray:
        """
        Build Hamiltonian from Pauli string specification.
        
        Example: pauli_terms = [("ZZ", 1.0), ("XX", 0.5), ("YY", 0.5)]
        """
        dim = 2 ** num_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        for pauli_string, coeff in pauli_terms:
            # Build tensor product
            term = np.array([[1]], dtype=complex)
            for pauli_char in pauli_string:
                term = np.kron(term, pauli_dict[pauli_char])
            
            H += coeff * term
        
        return H
    
    @staticmethod
    def heisenberg(num_qubits: int, J: float = 1.0) -> np.ndarray:
        """Build Heisenberg XXZ Hamiltonian"""
        pauli_terms = []
        
        for i in range(num_qubits - 1):
            # XX interaction
            xx_string = 'I' * i + 'XX' + 'I' * (num_qubits - i - 2)
            pauli_terms.append((xx_string, J))
            
            # YY interaction
            yy_string = 'I' * i + 'YY' + 'I' * (num_qubits - i - 2)
            pauli_terms.append((yy_string, J))
            
            # ZZ interaction
            zz_string = 'I' * i + 'ZZ' + 'I' * (num_qubits - i - 2)
            pauli_terms.append((zz_string, J))
        
        return HamiltonianBuilder.pauli_string(num_qubits, pauli_terms)


class QuantumStateEvolver:
    """
    Enhanced Quantum State Evolver with comprehensive input capabilities.
    """
    
    def __init__(self, 
                 initial_state: Union[str, List, np.ndarray, dict],
                 num_copies: int = 1,
                 shots_per_basis: int = 1000,
                 verbose: bool = True):
        """
        Initialize the Quantum State Evolver.
        
        Parameters:
        -----------
        initial_state : str, list, np.ndarray, or dict
            The initial quantum state in any supported format:
            - List: [1, 0, 0, 1]
            - String (ket): "|00⟩ + |11⟩"
            - String (coeffs): "1 0 0 1"
            - Dict: {"00": 1, "11": 1}
            - Array: np.array([1, 0, 0, 1])
        num_copies : int
            Number of tensor copies of the state (N)
        shots_per_basis : int
            Number of measurements per tomography basis
        verbose : bool
            Print detailed information
        """
        self.verbose = verbose
        
        # Parse initial state
        self.initial_state = StateInputParser.parse_state(initial_state)
        self.initial_state = self._validate_and_normalize(self.initial_state)
        
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
        
        if self.verbose:
            self._print_initialization_info()
    
    def _validate_and_normalize(self, state: np.ndarray) -> np.ndarray:
        """Validate and normalize the input state"""
        # Check if it's a power of 2
        if not (len(state) & (len(state) - 1) == 0):
            raise ValueError(f"State dimension {len(state)} is not a power of 2")
        
        # Normalize
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0):
            if self.verbose:
                print(f"ℹ State norm was {norm:.6f}, normalizing to 1.0")
            state = state / norm
        
        return state
    
    def _print_initialization_info(self):
        """Print initialization information"""
        print(f"\n{'='*70}")
        print("QUANTUM STATE EVOLVER - INITIALIZED")
        print(f"{'='*70}")
        print(f"Initial State:")
        self._print_state_info(self.initial_state)
        print(f"\nSystem Configuration:")
        print(f"  • Qubits per copy: {self.num_qubits}")
        print(f"  • Number of copies: {self.num_copies}")
        print(f"  • Total qubits: {self.total_qubits}")
        print(f"  • Hilbert space dimension: {2**self.total_qubits}")
        print(f"  • Shots per measurement basis: {self.shots_per_basis}")
        print(f"{'='*70}\n")
    
    def _print_state_info(self, state: np.ndarray, max_components: int = 8):
        """Print state information in readable format"""
        nonzero_indices = np.where(np.abs(state) > 1e-10)[0]
        
        if len(nonzero_indices) == 0:
            print("  [Zero state]")
            return
        
        print(f"  Non-zero components: {len(nonzero_indices)}/{len(state)}")
        
        for idx in nonzero_indices[:max_components]:
            coeff = state[idx]
            basis_state = format(idx, f'0{self.num_qubits}b')
            
            # Format coefficient
            real_part = np.real(coeff)
            imag_part = np.imag(coeff)
            
            if np.abs(imag_part) < 1e-10:
                coeff_str = f"{real_part:+.4f}"
            elif np.abs(real_part) < 1e-10:
                coeff_str = f"{imag_part:+.4f}i"
            else:
                coeff_str = f"{real_part:+.4f}{imag_part:+.4f}i"
            
            print(f"    {coeff_str} |{basis_state}⟩")
        
        if len(nonzero_indices) > max_components:
            print(f"    ... and {len(nonzero_indices) - max_components} more")
    
    def set_hamiltonian(self, 
                       hamiltonian: Union[np.ndarray, List, str],
                       evolution_time: float = 1.0,
                       **kwargs):
        """
        Set the Hamiltonian for evolution.
        
        Parameters:
        -----------
        hamiltonian : np.ndarray, list, or str
            The Hamiltonian. Can be:
            - Matrix (np.ndarray or list)
            - String: "random", "nearest_neighbor", "heisenberg"
        evolution_time : float
            Time parameter for evolution (t in e^(-iHt))
        **kwargs : Additional parameters for Hamiltonian construction
            - For "random": strength=1.0
            - For "nearest_neighbor": coupling=1.0
            - For "heisenberg": J=1.0
        """
        expected_dim = 2 ** self.total_qubits
        
        if isinstance(hamiltonian, str):
            hamiltonian = hamiltonian.lower()
            
            if hamiltonian == "random":
                strength = kwargs.get('strength', 1.0)
                H = HamiltonianBuilder.random_hermitian(expected_dim, strength)
            elif hamiltonian in ["nearest_neighbor", "nn"]:
                coupling = kwargs.get('coupling', 1.0)
                H = HamiltonianBuilder.nearest_neighbor(self.total_qubits, coupling)
            elif hamiltonian == "heisenberg":
                J = kwargs.get('J', 1.0)
                H = HamiltonianBuilder.heisenberg(self.total_qubits, J)
            else:
                raise ValueError(f"Unknown Hamiltonian type: {hamiltonian}")
        else:
            H = HamiltonianBuilder.from_matrix(hamiltonian)
        
        # Check dimensions
        if H.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Hamiltonian dimension {H.shape} doesn't match "
                f"system dimension {expected_dim}×{expected_dim}"
            )
        
        self.hamiltonian = H
        self.evolution_time = evolution_time
        
        if self.verbose:
            print(f"✓ Hamiltonian set:")
            print(f"  • Dimension: {H.shape[0]}×{H.shape[0]}")
            print(f"  • Evolution time: {evolution_time}")
            print(f"  • Spectrum range: [{np.min(np.linalg.eigvalsh(H)):.4f}, "
                  f"{np.max(np.linalg.eigvalsh(H)):.4f}]")
    
    def create_tensor_product_state(self) -> QuantumCircuit:
        """Create N tensor copies of the initial state"""
        qc = QuantumCircuit(self.total_qubits)
        
        # Initialize each copy
        for copy_idx in range(self.num_copies):
            start_qubit = copy_idx * self.num_qubits
            end_qubit = start_qubit + self.num_qubits
            
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
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("QUANTUM STATE EVOLUTION")
            print(f"{'='*70}")
        
        # Create initial state circuit
        qc = self.create_tensor_product_state()
        
        # Create evolution operator
        if self.verbose:
            print("→ Computing evolution operator e^(-iHt)...")
        U = Operator(np.exp(-1j * self.hamiltonian * self.evolution_time))
        
        # Apply evolution
        if self.verbose:
            print("→ Applying Hamiltonian evolution...")
        qc.unitary(U, range(self.total_qubits), label='e^(-iHt)')
        
        # Get evolved statevector
        qc.save_statevector()
        result = self.simulator.run(qc).result()
        self.evolved_state = result.get_statevector()
        
        if self.verbose:
            print("✓ State evolution complete\n")
            print("Evolved State:")
            self._print_state_info(self.evolved_state.data)
        
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
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("QUANTUM STATE TOMOGRAPHY")
            print(f"{'='*70}")
        
        # Generate measurement bases
        bases = self._generate_pauli_bases()
        if self.verbose:
            print(f"→ Measuring in {len(bases)} Pauli bases...")
            print(f"→ Shots per basis: {self.shots_per_basis}")
        
        # Perform measurements
        measurement_results = {}
        for i, (basis_name, basis_circuit) in enumerate(bases.items()):
            if self.verbose and i % 10 == 0:
                print(f"  Progress: {i}/{len(bases)} bases measured")
            counts = self._measure_in_basis(basis_circuit)
            measurement_results[basis_name] = counts
        
        # Reconstruct density matrix
        if self.verbose:
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
        trace = np.real(np.trace(rho.data))
        
        if self.verbose:
            print(f"✓ Reconstruction complete")
            print(f"\nTomography Metrics:")
            print(f"  • Fidelity: {fidelity:.6f}")
            print(f"  • Purity: {purity:.6f}")
            print(f"  • Trace: {trace:.6f}")
        
        self.tomography_result = TomographyResult(
            density_matrix=rho,
            reconstructed_state=reconstructed_state,
            measurement_counts=measurement_results,
            fidelity=fidelity,
            purity=purity,
            trace=trace
        )
        
        return self.tomography_result
    
    def _generate_pauli_bases(self) -> dict:
        """Generate Pauli measurement bases for tomography"""
        bases = {}
        
        if self.total_qubits <= 3:
            # Full tomography for small systems
            from itertools import product
            pauli_labels = ['X', 'Y', 'Z']
            
            for pauli_string in product(pauli_labels, repeat=self.total_qubits):
                basis_name = ''.join(pauli_string)
                bases[basis_name] = self._create_basis_circuit(list(pauli_string))
        else:
            # Use single-qubit and two-qubit bases for larger systems
            pauli_labels = ['X', 'Y', 'Z']
            
            # Single-qubit bases
            for qubit in range(self.total_qubits):
                for pauli in pauli_labels:
                    basis_name = f"{pauli}{qubit}"
                    pauli_string = ['I'] * self.total_qubits
                    pauli_string[qubit] = pauli
                    bases[basis_name] = self._create_basis_circuit(pauli_string)
            
            # Two-qubit bases (for correlations)
            for q1 in range(min(3, self.total_qubits)):
                for q2 in range(q1 + 1, min(4, self.total_qubits)):
                    for p1 in pauli_labels:
                        for p2 in pauli_labels:
                            basis_name = f"{p1}{q1}{p2}{q2}"
                            pauli_string = ['I'] * self.total_qubits
                            pauli_string[q1] = p1
                            pauli_string[q2] = p2
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
        
        # Simple reconstruction using measurement outcomes
        total_measurements = 0
        
        for basis_name, counts in measurement_results.items():
            for bitstring, count in counts.items():
                # Convert bitstring to state
                idx = int(bitstring.replace(' ', ''), 2)
                basis_state = np.zeros(dim)
                basis_state[idx] = 1.0
                
                # Add to density matrix
                rho += count * np.outer(basis_state, basis_state)
                total_measurements += count
        
        # Normalize
        rho = rho / total_measurements
        
        # Ensure Hermiticity and proper trace
        rho = (rho + rho.conj().T) / 2
        rho = rho / np.trace(rho)
        
        return DensityMatrix(rho)
    
    def _calculate_fidelity(self, state1: Statevector, state2: Statevector) -> float:
        """Calculate fidelity between two states"""
        return np.abs(np.vdot(state1.data, state2.data)) ** 2
    
    def get_density_matrix(self) -> DensityMatrix:
        """Get the reconstructed density matrix"""
        if self.tomography_result is None:
            raise ValueError("No tomography results. Call perform_tomography() first.")
        return self.tomography_result.density_matrix
    
    def get_reconstructed_state(self) -> Statevector:
        """Get the reconstructed state vector"""
        if self.tomography_result is None:
            raise ValueError("No tomography results. Call perform_tomography() first.")
        return self.tomography_result.reconstructed_state
    
    def visualize_results(self, figsize=(18, 5)):
        """Visualize tomography results"""
        if self.tomography_result is None:
            raise ValueError("No tomography results. Call perform_tomography() first.")
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Plot 1: Density matrix (real part)
        rho_real = np.real(self.tomography_result.density_matrix.data)
        im1 = axes[0].imshow(rho_real, cmap='RdBu', vmin=-np.max(np.abs(rho_real)), 
                            vmax=np.max(np.abs(rho_real)))
        axes[0].set_title('Density Matrix (Real)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Basis State')
        axes[0].set_ylabel('Basis State')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Density matrix (imaginary part)
        rho_imag = np.imag(self.tomography_result.density_matrix.data)
        im2 = axes[1].imshow(rho_imag, cmap='RdBu', vmin=-np.max(np.abs(rho_imag)), 
                            vmax=np.max(np.abs(rho_imag)))
        axes[1].set_title('Density Matrix (Imaginary)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Basis State')
        axes[1].set_ylabel('Basis State')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot 3: Eigenvalues
        eigenvalues = np.linalg.eigvalsh(self.tomography_result.density_matrix.data)
        eigenvalues = np.sort(eigenvalues)[::-1]
        axes[2].bar(range(len(eigenvalues)), eigenvalues, color='steelblue')
        axes[2].set_title('Density Matrix Eigenvalues', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Eigenvalue Index')
        axes[2].set_ylabel('Eigenvalue')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: State populations (diagonal of density matrix)
        populations = np.real(np.diag(self.tomography_result.density_matrix.data))
        axes[3].bar(range(len(populations)), populations, color='coral')
        axes[3].set_title('State Populations', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Basis State')
        axes[3].set_ylabel('Population')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print a comprehensive summary"""
        print(f"\n{'='*70}")
        print("COMPUTATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nInitial State:")
        self._print_state_info(self.initial_state)
        
        print(f"\nSystem Configuration:")
        print(f"  • Qubits per copy: {self.num_qubits}")
        print(f"  • Number of copies: {self.num_copies}")
        print(f"  • Total qubits: {self.total_qubits}")
        print(f"  • Hilbert space dimension: {2**self.total_qubits}")
        
        if self.hamiltonian is not None:
            print(f"\nHamiltonian:")
            eigenvalues = np.linalg.eigvalsh(self.hamiltonian)
            print(f"  • Evolution time: {self.evolution_time}")
            print(f"  • Spectrum: [{np.min(eigenvalues):.4f}, {np.max(eigenvalues):.4f}]")
            print(f"  • Spectral gap: {eigenvalues[1] - eigenvalues[0]:.4f}")
        
        if self.evolved_state is not None:
            print(f"\nEvolved State:")
            self._print_state_info(self.evolved_state.data)
        
        if self.tomography_result is not None:
            print(f"\nTomography Results:")
            print(f"  • Measurement shots per basis: {self.shots_per_basis}")
            print(f"  • Number of measurement bases: {len(self.tomography_result.measurement_counts)}")
            print(f"  • Fidelity: {self.tomography_result.fidelity:.6f}")
            print(f"  • Purity: {self.tomography_result.purity:.6f}")
            print(f"  • Trace: {self.tomography_result.trace:.6f}")
        
        print(f"{'='*70}\n")
    
    def save_results(self, filename: str):
        """Save results to file"""
        if self.tomography_result is None:
            raise ValueError("No results to save. Run evolution and tomography first.")
        
        results = {
            'initial_state': self.initial_state.tolist(),
            'num_copies': self.num_copies,
            'num_qubits': self.num_qubits,
            'evolution_time': self.evolution_time,
            'fidelity': self.tomography_result.fidelity,
            'purity': self.tomography_result.purity,
            'trace': self.tomography_result.trace,
            'density_matrix_real': np.real(self.tomography_result.density_matrix.data).tolist(),
            'density_matrix_imag': np.imag(self.tomography_result.density_matrix.data).tolist(),
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {filename}")


# =============================================================================
# INTERACTIVE INTERFACE
# =============================================================================

def interactive_interface():
    """Comprehensive interactive interface"""
    print("\n" + "="*70)
    print(" " * 15 + "QUANTUM STATE EVOLUTION & TOMOGRAPHY")
    print(" " * 20 + "Interactive Mode")
    print("="*70 + "\n")
    
    # Step 1: Get initial state
    print("STEP 1: Define Initial State")
    print("-" * 70)
    print("\nChoose input method:")
    print("  1. Predefined states (Bell, GHZ, W, etc.)")
    print("  2. Ket notation (e.g., '|00⟩ + |11⟩')")
    print("  3. Coefficient list (e.g., '1 0 0 1')")
    print("  4. Random state")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        print("\nPredefined states:")
        print("  a. Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
        print("  b. Bell state |Φ-⟩ = (|00⟩ - |11⟩)/√2")
        print("  c. Bell state |Ψ+⟩ = (|01⟩ + |10⟩)/√2")
        print("  d. Bell state |Ψ-⟩ = (|01⟩ - |10⟩)/√2")
        print("  e. GHZ state (|000⟩ + |111⟩)/√2")
        print("  f. W state (|001⟩ + |010⟩ + |100⟩)/√3")
        print("  g. Equal superposition |+⟩⊗n")
        
        subchoice = input("\nEnter choice (a-g): ").strip().lower()
        
        predefined_states = {
            'a': "|00⟩ + |11⟩",
            'b': "|00⟩ - |11⟩",
            'c': "|01⟩ + |10⟩",
            'd': "|01⟩ - |10⟩",
            'e': "|000⟩ + |111⟩",
            'f': "|001⟩ + |010⟩ + |100⟩",
        }
        
        if subchoice == 'g':
            n = int(input("Enter number of qubits: "))
            initial_state = np.ones(2**n) / np.sqrt(2**n)
        else:
            initial_state = predefined_states.get(subchoice, "|00⟩ + |11⟩")
    
    elif choice == '2':
        initial_state = input("\nEnter state in ket notation (e.g., '|00⟩ + |11⟩'): ").strip()
    
    elif choice == '3':
        initial_state = input("\nEnter coefficients (space-separated): ").strip()
    
    else:  # Random
        n = int(input("\nEnter number of qubits: "))
        initial_state = np.random.randn(2**n) + 1j * np.random.randn(2**n)
    
    # Step 2: Number of copies
    print("\n" + "="*70)
    print("STEP 2: Number of Tensor Copies")
    print("-" * 70)
    num_copies = int(input("\nEnter number of copies N (1-3 recommended): ").strip())
    
    # Step 3: Measurement settings
    print("\n" + "="*70)
    print("STEP 3: Measurement Settings")
    print("-" * 70)
    shots = int(input("\nEnter shots per measurement basis (100-10000): ").strip())
    
    # Create evolver
    print("\n→ Initializing quantum state evolver...")
    evolver = QuantumStateEvolver(
        initial_state=initial_state,
        num_copies=num_copies,
        shots_per_basis=shots,
        verbose=True
    )
    
    # Step 4: Define Hamiltonian
    print("\n" + "="*70)
    print("STEP 4: Define Hamiltonian")
    print("-" * 70)
    print("\nChoose Hamiltonian type:")
    print("  1. Random Hermitian")
    print("  2. Nearest-neighbor interaction")
    print("  3. Heisenberg model")
    print("  4. Custom matrix input")
    
    h_choice = input("\nEnter choice (1-4): ").strip()
    
    if h_choice == '1':
        strength = float(input("Enter strength (e.g., 1.0): "))
        evolver.set_hamiltonian("random", strength=strength)
    elif h_choice == '2':
        coupling = float(input("Enter coupling strength (e.g., 0.5): "))
        evolver.set_hamiltonian("nearest_neighbor", coupling=coupling)
    elif h_choice == '3':
        J = float(input("Enter exchange constant J (e.g., 1.0): "))
        evolver.set_hamiltonian("heisenberg", J=J)
    else:
        print("\nFor custom matrix, using random Hamiltonian instead.")
        evolver.set_hamiltonian("random")
    
    evolution_time = float(input("\nEnter evolution time t (e.g., 1.0): "))
    evolver.evolution_time = evolution_time
    
    # Step 5: Run computation
    print("\n" + "="*70)
    print("STEP 5: Running Computation")
    print("-" * 70)
    
    input("\nPress Enter to start evolution...")
    evolver.evolve_state()
    
    input("\nPress Enter to start tomography...")
    evolver.perform_tomography()
    
    # Step 6: Display results
    print("\n" + "="*70)
    print("STEP 6: Results")
    print("-" * 70)
    
    evolver.print_summary()
    
    viz = input("\nVisualize results? (y/n): ").strip().lower()
    if viz == 'y':
        evolver.visualize_results()
    
    save = input("\nSave results to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = input("Enter filename (e.g., results.json): ").strip()
        evolver.save_results(filename)
    
    return evolver


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_bell_state_evolution():
    """Example: Bell state with different input formats"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Bell State Evolution - Multiple Input Formats")
    print("="*70 + "\n")
    
    # Method 1: Ket notation
    print("Method 1: Using ket notation")
    evolver1 = QuantumStateEvolver(
        initial_state="|00⟩ + |11⟩",
        num_copies=1,
        shots_per_basis=1000
    )
    
    # Method 2: Dictionary notation
    print("\nMethod 2: Using dictionary notation")
    evolver2 = QuantumStateEvolver(
        initial_state={"00": 1, "11": 1},
        num_copies=1,
        shots_per_basis=1000,
        verbose=False
    )
    
    # Method 3: List notation
    print("\nMethod 3: Using list notation")
    evolver3 = QuantumStateEvolver(
        initial_state=[1, 0, 0, 1],
        num_copies=1,
        shots_per_basis=1000,
        verbose=False
    )
    
    # Set Hamiltonian and evolve
    evolver1.set_hamiltonian("nearest_neighbor", evolution_time=0.5, coupling=0.3)
    evolver1.evolve_state()
    evolver1.perform_tomography()
    evolver1.print_summary()
    evolver1.visualize_results()
    
    return evolver1


def example_ghz_heisenberg():
    """Example: GHZ state with Heisenberg Hamiltonian"""
    print("\n" + "="*70)
    print("EXAMPLE 2: GHZ State with Heisenberg Hamiltonian")
    print("="*70 + "\n")
    
    evolver = QuantumStateEvolver(
        initial_state="|000⟩ + |111⟩",
        num_copies=1,
        shots_per_basis=2000
    )
    
    evolver.set_hamiltonian("heisenberg", evolution_time=1.0, J=0.5)
    evolver.evolve_state()
    evolver.perform_tomography()
    evolver.print_summary()
    evolver.visualize_results()
    
    return evolver


def example_custom_superposition():
    """Example: Custom superposition state"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Superposition State")
    print("="*70 + "\n")
    
    # Create custom superposition: 0.6|00⟩ + 0.8|11⟩
    evolver = QuantumStateEvolver(
        initial_state="0.6|00⟩ + 0.8|11⟩",
        num_copies=2,
        shots_per_basis=1500
    )
    
    evolver.set_hamiltonian("random", evolution_time=0.8, strength=0.5)
    evolver.evolve_state()
    evolver.perform_tomography()
    evolver.print_summary()
    evolver.visualize_results()
    
    return evolver


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 10 + "QUANTUM STATE EVOLUTION & TOMOGRAPHY SYSTEM")
    print("="*70)
    
    print("\nChoose mode:")
    print("  1. Interactive mode (guided input)")
    print("  2. Example: Bell state evolution")
    print("  3. Example: GHZ state with Heisenberg Hamiltonian")
    print("  4. Example: Custom superposition state")
    
    mode = input("\nEnter choice (1-4): ").strip()
    
    if mode == '1':
        evolver = interactive_interface()
    elif mode == '2':
        evolver = example_bell_state_evolution()
    elif mode == '3':
        evolver = example_ghz_heisenberg()
    elif mode == '4':
        evolver = example_custom_superposition()
    else:
        print("\nInvalid choice. Running interactive mode...")
        evolver = interactive_interface()
    
    print("\n✓ Program completed successfully!")
# pip install qiskit qiskit-aer qiskit-experiments matplotlib

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit_experiments.library import Standard1QubitTomography
from qiskit.visualization import plot_bloch_multivector, plot_state_city
import matplotlib.pyplot as plt

# -------------------------------
# 1. Prepare initial state |ψ> = (|0> + |1>)/√2
# -------------------------------
qc = QuantumCircuit(1)
qc.h(0)
initial_sv = Statevector.from_instruction(qc)
print("Initial state vector:")
print(initial_sv)

plot_bloch_multivector(initial_sv).show()

# -------------------------------
# 2. Define Hamiltonian H = X
# -------------------------------
X = Operator.from_label('X')
t = 0.1
U = Operator(np.linalg.expm(-1j * X.data * t))

# -------------------------------
# 3. Evolve the state
# -------------------------------
evolved_sv = initial_sv.evolve(U)
print("\nEvolved state vector (analytical):")
print(evolved_sv)

# -------------------------------
# 4. Quantum state tomography using qiskit_experiments
# -------------------------------
# Prepare a circuit that includes the evolution
qc_evolved = qc.evolve(U)

# Create tomography experiment
tomo_exp = Standard1QubitTomography(qc_evolved, 0)

# Use Aer simulator
backend = AerSimulator()
tomo_results = tomo_exp.run(backend, shots=8192).block_for_results()

# Reconstructed density matrix
rho = tomo_results.data(0)['density_matrix']
print("\nReconstructed density matrix via tomography:")
print(rho)

# Visualize
plot_state_city(rho).show()

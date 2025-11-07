# Install Qiskit if not already
# pip install qiskit qiskit-ignis matplotlib

import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from qiskit.visualization import plot_bloch_multivector, plot_state_city
from qiskit.ignis.verification import tomography
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import matplotlib.pyplot as plt

# -------------------------------
# 1. Prepare initial state |psi> = (|0> + |1>)/sqrt(2)
# -------------------------------
qc = QuantumCircuit(1)
qc.h(0)  # Hadamard creates |0> + |1>
initial_sv = Statevector.from_instruction(qc)
print("Initial state vector:")
print(initial_sv)

# Visualize
plot_bloch_multivector(initial_sv).show()

# -------------------------------
# 2. Define a simple Hamiltonian H = X
# -------------------------------
X = Operator.from_label('X')
t = 0.1  # small time

# Time evolution operator U = exp(-i H t)
U = Operator(np.linalg.expm(-1j * X.data * t))

# -------------------------------
# 3. Evolve the state
# -------------------------------
evolved_sv = initial_sv.evolve(U)
print("\nEvolved state vector (analytical):")
print(evolved_sv)

# -------------------------------
# 4. Quantum state tomography
# -------------------------------
# Create tomography circuits
tomo_circuits = state_tomography_circuits(qc, [0])

# Execute circuits on simulator
backend = Aer.get_backend('qasm_simulator')
tomo_qobj = assemble(tomo_circuits, shots=8192)
results = backend.run(tomo_qobj).result()

# Reconstruct density matrix
tomo_fitter = StateTomographyFitter(results, tomo_circuits)
rho_fit = tomo_fitter.fit()
rho = DensityMatrix(rho_fit)
print("\nReconstructed density matrix via tomography:")
print(rho)

# Optional: visualize reconstructed state
plot_state_city(rho).show()

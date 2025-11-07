import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from qiskit.visualization import plot_bloch_multivector, plot_state_city
from qiskit.ignis.verification.tomography import (
    state_tomography_circuits, StateTomographyFitter
)
import matplotlib.pyplot as plt

# -------------------------------
# 1. Prepare initial state |psi> = (|0> + |1>)/sqrt(2)
# -------------------------------
qc = QuantumCircuit(1)
qc.h(0)
initial_sv = Statevector.from_instruction(qc)
print("Initial state vector:")
print(initial_sv)

plot_bloch_multivector(initial_sv).show()

# -------------------------------
# 2. Define a simple Hamiltonian H = X
# -------------------------------
X = Operator.from_label('X')
t = 0.1  # small time
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
tomo_circuits = state_tomography_circuits(qc.evolve(U), [0])

# Execute circuits on simulator
backend = AerSimulator()
job = backend.run(tomo_circuits, shots=8192)  # no assemble needed
results = job.result()

# Reconstruct density matrix
tomo_fitter = StateTomographyFitter(results, tomo_circuits)
rho_fit = tomo_fitter.fit()
rho = DensityMatrix(rho_fit)
print("\nReconstructed density matrix via tomography:")
print(rho)

# Visualize reconstructed state
plot_state_city(rho).show()

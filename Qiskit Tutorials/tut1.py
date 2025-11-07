import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Statevector, Operator, Pauli
from qiskit.quantum_info.operators import Pauli

# ------------------------
# 1. Define initial state
# ------------------------
# Let's take |0> as initial state
initial_state = Statevector.from_label('0')
print("Initial state:\n", initial_state)

# ------------------------
# 2. Define Hamiltonian
# ------------------------
# Example: H = X (Pauli-X) on a single qubit
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

H_matrix = X  # Hamiltonian

# ------------------------
# 3. Time evolution
# ------------------------
t = 1.0  # time

# Evolution operator: U = exp(-i H t)
U = Operator(expm(-1j * H_matrix * t))

# Apply evolution to initial state
final_state = initial_state.evolve(U)

print("\nFinal state after evolution:\n", final_state)

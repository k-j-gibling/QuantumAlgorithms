import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

# ------------------------
# 1. Define Hamiltonian
# ------------------------
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# 2-qubit Hamiltonian: H = X⊗Z + 0.5 * Y⊗Y
H_matrix = np.kron(X, Z) + 0.5 * np.kron(Y, Y)

# ------------------------
# 2. Compute evolution operator
# ------------------------
t = 1.0
U_matrix = expm(-1j * H_matrix * t)
U_op = Operator(U_matrix)

# ------------------------
# 3. Circuit for statevector simulation
# ------------------------
qc_sv = QuantumCircuit(2)
qc_sv.unitary(U_op, [0, 1])

# Use save_statevector instruction for modern Qiskit
qc_sv.save_statevector(label='sv')  # <-- key change

# ------------------------
# 4. Run simulator
# ------------------------
sv_backend = AerSimulator()
job = sv_backend.run(qc_sv)
result = job.result()

# Extract statevector from result
statevector = result.data()['sv']  # <-- key change
print("Statevector after evolution:\n", statevector)

# ------------------------
# 5. Circuit for QASM simulation (measurements)
# ------------------------
"""qc_qasm = QuantumCircuit(2)
qc_qasm.unitary(U_op, [0, 1])
qc_qasm.measure_all()

qasm_backend = AerSimulator(method='qasm')
job_qasm = qasm_backend.run(qc_qasm, shots=1024)
result_qasm = job_qasm.result()
counts = result_qasm.get_counts()
print("\nMeasurement counts:\n", counts)"""



# Create the circuit with measurement
qc_qasm = QuantumCircuit(2)
qc_qasm.unitary(U_op, [0, 1])
qc_qasm.measure_all()

# Use AerSimulator (default is QASM-style shots)
qasm_backend = AerSimulator()  # <-- just AerSimulator(), no method='qasm'
job_qasm = qasm_backend.run(qc_qasm, shots=1024)
result_qasm = job_qasm.result()
counts = result_qasm.get_counts()
print("\nMeasurement counts:\n", counts)



"""
    Newly added...
"""


import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_state_city, plot_bloch_multivector

# ------------------------
# 1. Define Hamiltonian
# ------------------------
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Example 2-qubit Hamiltonian
H = np.kron(X, Z) + 0.5 * np.kron(Y, Y)

# ------------------------
# 2. Compute evolution operator
# ------------------------
t = 1.0  # time
U = expm(-1j * H * t)
U_op = Operator(U)

# ------------------------
# 3. Create circuit and apply evolution
# ------------------------
qc = QuantumCircuit(2)
qc.unitary(U_op, [0, 1])

# Save statevector for observation
qc.save_statevector(label='sv')

# ------------------------
# 4. Run statevector simulator
# ------------------------
sim = AerSimulator(method='statevector')
job = sim.run(qc)
result = job.result()

# Extract statevector
statevector = result.data()['sv']
print("Statevector after evolution:\n", statevector)

# ------------------------
# 5. Optional: Compute probabilities
# ------------------------
probabilities = np.abs(statevector)**2
print("\nProbabilities of each basis state:\n", probabilities)

# ------------------------
# 6. Optional: Visualize statevector
# ------------------------
# plot_city = plot_state_city(statevector)
# plot_bloch = plot_bloch_multivector(statevector)
# plot_city.show()
# plot_bloch.show()



# pip instal
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from qiskit.visualization import plot_bloch_multivector, plot_state_city
import matplotlib.pyplot as plt
import scipy


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
# 2. Define Hamiltonian evolution H = X
# -------------------------------
X = Operator.from_label('X')
t = 0.1
U = Operator(scipy.linalg.expm(-1j * X.data * t))

# -------------------------------
# 3. Evolve the state analytically
# -------------------------------
evolved_sv = initial_sv.evolve(U)
print("\nEvolved state vector (analytical):")
print(evolved_sv)

# -------------------------------
# 4. Prepare a circuit that applies U as a unitary
# -------------------------------
qc_evolved = QuantumCircuit(1)
qc_evolved.unitary(U, [0])  # apply evolution as a gate

# -------------------------------
# 5. Manual tomography
# -------------------------------
def measure_expectation(circuit, rotation=None, shots=2000):
    """Measure in a rotated basis and return expectation value <σ>"""
    qc_rot = circuit.copy()
    if rotation == 'X':
        qc_rot.h(0)
    elif rotation == 'Y':
        qc_rot.sdg(0)
        qc_rot.h(0)
    qc_rot.measure_all()
    sim = AerSimulator()
    job = sim.run(qc_rot, shots=shots)
    result = job.result()
    counts = result.get_counts()
    total = sum(counts.values())
    p0 = counts.get('0', 0) / total
    p1 = counts.get('1', 0) / total
    return p0 - p1

r_z = measure_expectation(qc_evolved, rotation=None)
r_x = measure_expectation(qc_evolved, rotation='X')
r_y = measure_expectation(qc_evolved, rotation='Y')

print("\nBloch vector components from tomography:")
print(f"<σx> = {r_x:.3f}, <σy> = {r_y:.3f}, <σz> = {r_z:.3f}")

# -------------------------------
# 6. Reconstruct density matrix
# -------------------------------
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)
I  = np.eye(2, dtype=complex)

rho = 0.5*(I + r_x*sx + r_y*sy + r_z*sz)
rho_dm = DensityMatrix(rho)

print("\nReconstructed density matrix:")
print(rho_dm)

# -------------------------------
# 7. Visualizations
# -------------------------------
plot_bloch_multivector(evolved_sv).show()
plot_state_city(rho_dm).show()


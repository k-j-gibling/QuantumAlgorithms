from qutip import Qobj, ptrace

# Example: a 4x4 density matrix for two 2-level systems (qubits)
import numpy as np
rho = Qobj(np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]))

# Take partial trace over subsystem 1 (trace out B)
rho_A = ptrace(rho, 0)
print(rho_A)


from qiskit.quantum_info import partial_trace, DensityMatrix

# Bell state
from qiskit.quantum_info import Statevector
psi = Statevector.from_label('00') + Statevector.from_label('11')
psi = psi / np.sqrt(2)

rho = DensityMatrix(psi)

# Partial trace over the second qubit
rho_A = partial_trace(rho, [1])

print(rho_A)

phi = np.array([1,1,1,1])/4
rho = DensityMatrix(phi)
rho_A = partial_trace(rho, [0])



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

import numpy as np
import qutip as qt

# Define a 4x4 density matrix as a NumPy array (2-qubit system)
"""rho_np = np.array([
    [0.5, 0.0, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.5]
], dtype=complex)"""


rho_np = np.array([
    [0.5, 0.0, 0.0, 0.5,1,2,3,4],
    [0.0, 0.0, 0.0, 0.0, 1,2,3,4],
    [0.0, 0.0, 0.0, 0.0,1,2,3,4],
    [0.5, 0.0, 0.0, 0.5,1,2,3,4],
    [0.5, 0.0, 0.0, 0.5,1,2,3,4],
    [0.5, 0.0, 0.0, 0.5,1,2,3,4],
    [0.5, 0.0, 0.0, 0.5,1,2,3,4],
    [0.5, 0.0, 0.0, 0.5,1,2,3,4]
], dtype=complex)

# Convert NumPy array → QuTiP Qobj
rho = qt.Qobj(rho_np, dims=[[2, 2,2],[2, 2,2]])

# Partial trace over qubit 1 (keep qubit 0)
rho_A = rho.ptrace(0)

print(rho_A.full()) #This will print a numpy matrix.



def partial_trace(rho, N, keep_qubits):
    """
        Note: keep_qubits will not be used in here.
    """

    # Convert NumPy array → QuTiP Qobj
    rho = qt.Qobj(rho_np, dims=[[2, 2,2],[2, 2,2]])

    # Partial trace over qubit 1 (keep qubit 0)
    rho_A = rho.ptrace(0)

    return rho_A.full()
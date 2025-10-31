import numpy as np
from quantum_evolution import evolve_state
# Define Hamiltonian
H = np.array([[1, 0], [0, -1]]) # Diagonal
# Define initial state
psi0 = np.array([1, 1]) / np.sqrt(2) # |+> state
# Evolve for time t=1.0
t = 1.0
psi_t,info = evolve_state(psi0, H, t)
print(f"Evolved state:{psi_t}")
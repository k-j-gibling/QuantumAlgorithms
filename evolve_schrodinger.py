import numpy as np
from effectiveHamiltonian import compute_effective_hamiltonian
from runge_kutta_hamiltonian import runge_kutta_4


'''
Code to implement a hamiltonian, compute the effective hamoltonian,
and compute the time evolution for a given state... Here our
state will be denoted by \ket{\phi}
'''


phi_0 = np.array([1,1])/np.sqrt(2)

H_LIST = [['X','I','X'], ['X','Z', 'I'], ['Z','X','Z']]

H_eff = compute_effective_hamiltonian(H_LIST,phi_0)

# Evolve
t = 1.0
phi_t = runge_kutta_4(phi_0, H_eff, t, dt=0.001)



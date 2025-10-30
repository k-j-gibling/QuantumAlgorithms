from density_matrix_ops import *
import numpy as np

psi = np.array([1,1,1,1,1,1,1,1])/np.sqrt(2)

rho = state_to_density_matrix(psi)

print(rho)

rho_A = partial_trace(rho, dims=[2,2,2], trace_out=[1,2])

print(rho_A)
"""
Experiment 1

"""

from final import H_list_to_H, compute_effective_hamiltonian, create_N_copy_state, _evolve, state_to_density_matrix, partial_trace, runge_kutta_4
from specificHamiltonians import prepare_H_test1
import numpy as np

from scipy.linalg import svdvals

def trace_norm(A):
    """Compute trace norm (sum of singular values) of matrix A"""
    return np.sum(svdvals(A))

def trace_norm_distance(A, B):
    """Compute trace norm distance ||A - B||_tr"""
    return trace_norm(A - B)


N_list = [3,4,5,6,7,8] #Experiment with various N.
#N_list = [3]
T_list = [0.1]	#Experiment with evolving for various times t in T_list

"""
	results_dict():
		Used for storing the results of the experiment for various N and t.
		key: (N, t)
"""
results_dict = dict()


"""
Prepare the initial state.
"""

psi_0 = np.array([1,1])/np.sqrt(2)
phi_0 = psi_0


"""
Prepare the Hamiltonians.
	
	Get the hamiltonian list.
	Get the main hamiltonian.
	Get the effective hamiltonian.
"""

plot_list_1 = list()


for N in N_list:
	psi_0 = np.array([1,1])/np.sqrt(2)
	phi_0 = psi_0

	#Prepare the Hamiltonian H_N.
	H_N_list = prepare_H_test1(N) #The preparation of this hamiltonian depends on the number of copies, N.

	#Get the corresponding Hamiltonian in matrix form.
	H_N = (1/N)*H_list_to_H(H_N_list)

	#Get the corresponding effective Hamiltonian.
	H_eff = compute_effective_hamiltonian(H_N_list, psi_0)

	#Create N copies of the initial state.
	psi_0_N = create_N_copy_state(psi_0, N)

	for t in T_list:
		psi_t_N = _evolve(psi_0_N, H_N, t)


		#Compute the density operator of this N-body system.
		psi_t_N_rho = state_to_density_matrix(psi_t_N)

		#Take the partial trace of this N-body system.
		rho_1 = partial_trace(psi_t_N_rho, N=N, keep_qubits=[0])

		#Now, evolve phi_0 to get phi_t.
		phi_t = runge_kutta_4(phi_0, H_eff, t, dt=0.001)

		#Compute the corresponding density matrix of this state.
		phi_t_rho = state_to_density_matrix(phi_t)

		#Compute error metrics and store results.
		#TODO.
		error_fro = np.linalg.norm(rho_1 - phi_t_rho, 'fro')     # Frobenius norm
		error_2   = np.linalg.norm(rho_1 - phi_t_rho, 2)         # Spectral (2-) norm
		error_inf = np.linalg.norm(rho_1 - phi_t_rho, np.inf)    # Max row sum

		results_dict[(N, t)] = dict()
		results_dict[(N, t)]['forbenius norm'] = error_fro
		results_dict[(N, t)]['spectral 2 norm'] = error_2
		results_dict[(N, t)]['mar row sum'] = error_inf

		trNorm = trace_norm_distance(rho_1, phi_t_rho)

		plot_list_1.append((N, trNorm))




#======================================================


from specificHamiltonians import prepare_H_test2


#N_list = [3,4,5,6,7,8,9] #Experiment with various N.
#N_list = [3]
#T_list = [0.001]	#Experiment with evolving for various times t in T_list

"""
	results_dict():
		Used for storing the results of the experiment for various N and t.
		key: (N, t)
"""
results_dict = dict()


"""
Prepare the initial state.
"""

psi_0 = np.array([1,1])/np.sqrt(2)
phi_0 = psi_0


"""
Prepare the Hamiltonians.
	
	Get the hamiltonian list.
	Get the main hamiltonian.
	Get the effective hamiltonian.
"""

plot_list_2 = list()


for N in N_list:
	psi_0 = np.array([1,1])/np.sqrt(2)
	phi_0 = psi_0

	#Prepare the Hamiltonian H_N.
	H_N_list = prepare_H_test2(N) #The preparation of this hamiltonian depends on the number of copies, N.

	#Get the corresponding Hamiltonian in matrix form.
	H_N = (1/N)*H_list_to_H(H_N_list)

	#Get the corresponding effective Hamiltonian.
	H_eff = compute_effective_hamiltonian(H_N_list, psi_0)

	#Create N copies of the initial state.
	psi_0_N = create_N_copy_state(psi_0, N)

	for t in T_list:
		psi_t_N = _evolve(psi_0_N, H_N, t)


		#Compute the density operator of this N-body system.
		psi_t_N_rho = state_to_density_matrix(psi_t_N)

		#Take the partial trace of this N-body system.
		rho_1 = partial_trace(psi_t_N_rho, N=N, keep_qubits=[0])

		#Now, evolve phi_0 to get phi_t.
		phi_t = runge_kutta_4(phi_0, H_eff, t, dt=0.001)

		#Compute the corresponding density matrix of this state.
		phi_t_rho = state_to_density_matrix(phi_t)

		#Compute error metrics and store results.
		#TODO.
		error_fro = np.linalg.norm(rho_1 - phi_t_rho, 'fro')     # Frobenius norm
		error_2   = np.linalg.norm(rho_1 - phi_t_rho, 2)         # Spectral (2-) norm
		error_inf = np.linalg.norm(rho_1 - phi_t_rho, np.inf)    # Max row sum

		results_dict[(N, t)] = dict()
		results_dict[(N, t)]['forbenius norm'] = error_fro
		results_dict[(N, t)]['spectral 2 norm'] = error_2
		results_dict[(N, t)]['mar row sum'] = error_inf

		trNorm = trace_norm_distance(rho_1, phi_t_rho)

		plot_list_2.append((N, trNorm))







import matplotlib.pyplot as plt
import numpy as np

system1_data = plot_list_1
system2_data = plot_list_2

# Example data - replace with your actual data
# System 1: [(N, error), (N, error), ...]
"""system1_data = [
    (2, 0.5),
    (3, 0.333),
    (4, 0.25),
    (5, 0.2),
    (10, 0.1),
    (20, 0.05),
    (50, 0.02),
    (100, 0.01),
    (200, 0.005)
]"""

# System 2: [(N, error), (N, error), ...]
"""system2_data = [
    (2, 0.6),
    (3, 0.4),
    (4, 1),
    (5, 0.24),
    (10, 0.12),
    (20, 0.06),
    (50, 0.024),
    (100, 0.012),
    (200, 0.006)
]"""

# Extract N values and errors for each system
system1_N = [point[0] for point in system1_data]
system1_errors = [point[1] for point in system1_data]

system2_N = [point[0] for point in system2_data]
system2_errors = [point[1] for point in system2_data]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot both systems
plt.plot(system1_N, system1_errors, 'o-', label='System 1', 
         linewidth=2, markersize=8, color='blue')
plt.plot(system2_N, system2_errors, 's-', label='System 2', 
         linewidth=2, markersize=8, color='red')

# Add theoretical O(1/N) line for comparison (optional)
N_theory = np.linspace(min(system1_N), max(system1_N), 100)
error_theory = 0.5 / N_theory  # Adjust constant as needed
plt.plot(N_theory, error_theory, '--', label='O(1/N) theoretical', 
         linewidth=1.5, color='gray', alpha=0.7)

# Formatting
plt.xlabel('Number of Copies (N)', fontsize=14, fontweight='bold')
plt.ylabel('Error', fontsize=14, fontweight='bold')
plt.title('Mean-Field Convergence: Error vs N', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')

# Use log scale if data spans multiple orders of magnitude
plt.xscale('log')
plt.yscale('log')

# Adjust layout and show
plt.tight_layout()
plt.savefig('error_vs_N.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'error_vs_N.png'")















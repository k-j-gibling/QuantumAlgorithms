"""
Experiment 1

"""

from specificHamiltonians import prepare_H_test1


N_list = [3,5,7,10,15] #Experiment with various N.
T_list = []	#Experiment with evolving for various times t in T_list

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



for N in N_list:
	#Prepare the Hamiltonian H_N.
	H_N_list = prepare_H_test1(N) #The preparation of this hamiltonian depends on the number of copies, N.

	#Get the corresponding Hamiltonian in matrix form.
	H_N = H_list_to_H(H_N_list)

	#Get the corresponding effective Hamiltonian.
	H_eff = compute_effective_hamiltonian(H_N_list, psi_0)

	#Create N copies of the initial state.
	psi_0_N = create_N_copy_state(psi_0, N)

	for t in T_list:
		psi_t_N = _evolve_hamiltonian(psi_0_N, H_N, t)


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
		










def compute_effective_hamiltonian(H_list, stateVector):
	H_eff_list = []

	for elmt in H_list:
		if elmt[0] != 'I':
			H_eff_list.append(elmt)


	H_eff_info_list = []

	for elmt in H_eff_list:
		elmt_operator = elmt[0]

		H_eff_info_list.append(dict())
		H_eff_info_list[-1]['operator'] = elmt_operator #For H_{eff}^1 the first element will be the operator.
		H_eff_info_list[-1]['operator_coefficient'] = 1 #Initialize to 1.
		H_eff_info_list[-1]['product_term_list'] = [] #Initially an empty list.

		for i in range(1,len(elmt)):
			if elmt[i] == elmt_operator:
				H_eff_info_list[-1]['operator_coefficient'] += 1
				continue

			if elmt[i] == 'I':
				continue

			H_eff_info_list[-1]['product_term_list'].append(elmt[i])


	matrices = []


	for elmt in H_eff_info_list:
		operator = elmt['operator']
		operator_coeff = elmt['operator_coefficient']
		product_terms = elmt['product_term_list']

		product_chain = 1


		for product_term in product_terms:
			#if product_term == 'X':
			#	expectationValue = expectation_value(X, stateVector)
			#	product_chain = product_chain*expectationValue
			expectationValue = expectation_value(pauli_dict[product_term], stateVector)
			product_chain = product_chain*expectationValue


		currentMatrix = None

		currentMatrix = pauli_dict[operator]

		"""if operator == 'X':
			currentMatrix = X
		elif currentMatrix == 'Y':
			currentMatrix = Y
		elif currentMatrix == 'I':
			currentMatrix = I
		else:
			currentMatrix = Z"""

		currentMatrix = currentMatrix*operator_coeff*product_chain
		matrices.append(currentMatrix)


	H_effective = matrices[0]

	for i in range(1,len(matrices)):
		H_effective += matrices[i]

	return H_effective
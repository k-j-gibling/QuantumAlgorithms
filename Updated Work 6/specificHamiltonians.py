import numpy as np
from scipy.linalg import eig, expm
from itertools import combinations
import warnings


# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

pauli_dict = dict()
pauli_dict['I'] = I
pauli_dict['X'] = X
pauli_dict['Y'] = Y
pauli_dict['Z'] = Z

def prepare_H_test1(N):
	"""
		N: This is the number of copies.
		The hamiltonian to be prepared is of the form H_{ij} = (X_i \otimes Z_j)/2 + 
			(Z_i \otimes X_j)/2

		output: H_list.
	"""

	H_list = []

	for i in range(N):
		for j in range(N):
			if j==i:
				continue

			"""
				Here we have H_{ij} = 'Hamiltonian form of current_term1' + 'Hamiltonian
											form of current_term2'
			"""
			current_term1 = [] #For instance this is I\otimes\otimes ... otimes X_i\otimes Z_j \otimes I ... \otimes I
			current_term2 = [] #For instance this is I\otimes\otimes ... otimes Z_i\otimes X_j \otimes I ... \otimes I

			for k in range(N):
				if k == i:
					current_term1.append('X')
					current_term2.append('Z')
					#continue
				elif k==j:
					current_term1.append('Z')
					current_term2.append('X')
				else:
					current_term1.append('I')
					current_term2.append('I')

			if current_term1 not in H_list:
				H_list.append(current_term1)

			if current_term2 not in H_list:
				H_list.append(current_term2)


	return H_list


def prepare_H_test2(N):
	"""
		N: This is the number of copies.
		The hamiltonian to be prepared is of the form H_{ij} = (X_i \otimes Z_j)/2 + 
			(Z_i \otimes X_j)/2

		output: H_list.
	"""

	H_list = []

	for i in range(N):
		for j in range(N):
			if j == i:
				continue
			for l in range(N):
				if l==i or l==j:
					continue

				"""
					Here we have H_{ij} = 'Hamiltonian form of current_term1' + 'Hamiltonian
												form of current_term2'
				"""
				current_term1 = [] #For instance this is I\otimes\otimes ... otimes X_i\otimes Z_j \otimes I ... \otimes I
				current_term2 = [] #For instance this is I\otimes\otimes ... otimes Z_i\otimes X_j \otimes I ... \otimes I
				current_term3 = []

				for k in range(N):
					if k == i:
						current_term1.append('X')
						current_term2.append('Z')
						current_term3.append('Z')
					elif k==j:
						current_term1.append('Z')
						current_term2.append('X')
						current_term3.append('Z')
					elif k==l:
						current_term1.append('Z')
						current_term2.append('Z')
						current_term3.append('X')
					else:
						current_term1.append('I')
						current_term2.append('I')
						current_term3.append('I')

				#H_list.append(current_term1)
				#H_list.append(current_term2)
				#H_list.append(current_term3)

				if current_term1 not in H_list:
					H_list.append(current_term1)

				if current_term2 not in H_list:
					H_list.append(current_term2)

				if current_term3 not in H_list:
					H_list.append(current_term3)


	return H_list


"""
from final import H_list_to_H

H_l = prepare_H_test2(7)

#Now from H_list we need to build the actual hamiltonian.
H = H_list_to_H(H_l)
"""


def prepare_H_test3(N):
	"""
		N: This is the number of copies.
		The hamiltonian to be prepared is of the form H_{ij} = (X_i \otimes Z_j)/2 + 
			(Z_i \otimes X_j)/2

		output: H_list.
	"""

	H_list = []

	for i in range(N):
		for j in range(N):
			if j==i:
				continue

			"""
				Here we have H_{ij} = 'Hamiltonian form of current_term1' + 'Hamiltonian
											form of current_term2'
			"""
			current_term1 = [] #For instance this is I\otimes\otimes ... otimes X_i\otimes Z_j \otimes I ... \otimes I
			current_term2 = [] #For instance this is I\otimes\otimes ... otimes Z_i\otimes X_j \otimes I ... \otimes I

			for k in range(N):
				if k == i:
					current_term1.append('X')
					current_term2.append('Y')
					#continue
				elif k==j:
					current_term1.append('Y')
					current_term2.append('X')
				else:
					current_term1.append('I')
					current_term2.append('I')

			H_list.append(current_term1)
			H_list.append(current_term2)

	return H_list








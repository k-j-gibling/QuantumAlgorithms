import sympy as sp

x1, x2, x3, x4, x5 = sp.symbols('x1 x2 x3 x4 x5')

symbolsList = [x1,x2,x3,x4,x5]

ODE_list = []

for i in range(1,len(symbolsList)):
	for j in range(i):
		#FIRST SYMMETRY POINT.
		#Create a new symbolic vector of zeros.
		newVector = sp.Matrix([0,0,0,0,0])
		#Update positions i and j or i or j (depending on the symmetry).
		newVector[i] = (symbolsList[i]**2)*symbolsList[j]
		newVector[j] = symbolsList[i]**3

		#Store this new datum point.
		ODE_list.append(newVector)

		#SECOND SYMMETRY POINT.
		#Create a new symbolic vector of zeros.
		newVector = sp.Matrix([0,0,0,0,0])
		#Update positions i and j or i or j (depending on the symmetry).

		#Store this new datum point.
		ODE_list.append(newVector)


		#THIRD SYMMETRY POINT.
		#Create a new symbolic vector of zeros.
		newVector = sp.Matrix([0,0,0,0,0])
		#Update positions i and j or i or j (depending on the symmetry).

		#Store this new datum point.
		ODE_list.append(newVector)

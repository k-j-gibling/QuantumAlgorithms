from sympy import Matrix

M = Matrix([[1, 0, 0],
            [1, 2, 0],
            [0, 0, 3]])

M.is_diagonal()  # Returns True


import numpy as np

# Create a 2x2 matrix
A = np.array([[1, 0],[1, 1]])

M1 = Matrix(A)

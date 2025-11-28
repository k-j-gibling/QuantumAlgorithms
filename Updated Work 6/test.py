from specificHamiltonians import prepare_H_test1
from final import H_list_to_H
import numpy as np

for N in [3, 4, 5, 6, 7,8, 9]:
    H_N_list = prepare_H_test1(N)
    H_N = (1/(N*(N-1))) * H_list_to_H(H_N_list)
    print(f"N={N}: ||H_N||_F = {np.linalg.norm(H_N, 'fro')}")
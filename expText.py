import numpy as np
from scipy.linalg import eig, expm


def _evolve_via_expm(psi0, H, t):
    """
    Evolve state using scipy's matrix exponential.
    
    Directly computes: |ψ(t)⟩ = e^{-iHt} |ψ(0)⟩
    """
    U = expm(-1j * H * t)
    return U @ psi0

v = np.array([1, 1])/np.sqrt(2)
A = np.array([[2, 0],[0, 2]])

print(_evolve_via_expm(v,A,10))

def _evolve_diagonal(psi0, H, t):
    """
    Evolve state when H is diagonal.
    
    For diagonal H = diag(λ₁, λ₂, ..., λₙ):
    e^{-iHt} = diag(e^{-iλ₁t}, e^{-iλ₂t}, ..., e^{-iλₙt})
    
    So: (e^{-iHt}ψ)ᵢ = e^{-iλᵢt} ψᵢ
    """
    eigenvalues = np.diag(H)
    phases = np.exp(-1j * eigenvalues * t)
    return phases * psi0

print(_evolve_diagonal(v,A,2))
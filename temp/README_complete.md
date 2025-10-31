# Quantum State Evolution for Permutation Symmetric Hamiltonians

## Complete Research Workflow

This program provides a complete pipeline for studying permutation symmetric quantum systems:

1. **Construct** permutation symmetric Hamiltonians
2. **Evolve** quantum states using e^{-iHt}
3. **Compute** density matrices
4. **Extract** reduced states via partial trace

## Files

- **quantum_state_evolution_complete.py** - Main program (25 KB)
- **quantum_evolution_complete_guide.pdf** - Complete documentation (14 pages, 259 KB)
- **quantum_evolution_complete_guide.tex** - LaTeX source

## Quick Start

```python
from quantum_state_evolution_complete import *

# 1. Construct Hamiltonian
N = 4  # Number of qubits
H = construct_TFIM(N, J=1.0, h=0.5)

# 2. Create initial state
psi0 = create_product_state(N, [1, 0])  # |0000⟩

# 3. Evolve and extract single-qubit state
t = 1.0
results = evolve_and_trace(H, psi0, t, N, keep_qubits=[0])

# 4. Access results
rho_1 = results['rho_reduced']  # 2×2 density matrix
print(rho_1)
```

## Main Function

```python
results = evolve_and_trace(
    H,                  # Hamiltonian (2^N × 2^N)
    psi0,              # Initial state (2^N,)
    t,                 # Evolution time
    N,                 # Number of qubits
    keep_qubits=[0],   # Which qubits to keep
    method='auto',     # Evolution method
    verbose=True       # Print progress
)
```

## What You Get

The function returns a dictionary with:

- `'psi_0'` - Initial state
- `'psi_t'` - Evolved state |ψ(t)⟩
- `'rho_full'` - Full N-qubit density matrix
- `'rho_reduced'` - Reduced density matrix (after partial trace)
- `'evolution_info'` - Evolution diagnostics
- `'density_info'` - Density matrix properties (purity, trace, etc.)

## Pre-built Hamiltonians

### Transverse Field Ising Model (TFIM)
```python
H = construct_TFIM(N, J=1.0, h=0.5)
# H = -J Σ_ij Z_i⊗Z_j - h Σ_i X_i
```

### XY Model
```python
H = construct_XY_model(N, J=0.5)
# H = J Σ_ij X_i⊗X_j
```

### General Mean-Field
```python
H = construct_general_MF(N, h_x=0.3, h_z=0.1, J_xx=0.2, J_zz=0.4)
# H = Σ_i (h_x X_i + h_z Z_i) + Σ_ij (J_xx X_i⊗X_j + J_zz Z_i⊗Z_j)
```

## Creating Initial States

```python
# Product states
psi0 = create_product_state(N, [1, 0])           # |0⟩^⊗N
psi0 = create_product_state(N, [0, 1])           # |1⟩^⊗N
psi0 = create_product_state(N, [1, 1]/√2)        # |+⟩^⊗N
psi0 = create_product_state(N, [0.8, 0.6])       # Custom

# Custom states
psi0 = np.zeros(2**N)
psi0[0] = 1.0  # |000...0⟩
```

## Partial Trace Options

```python
# Single qubit (most common)
results = evolve_and_trace(H, psi0, t, N, keep_qubits=[0])
rho_1 = results['rho_reduced']  # 2×2 matrix

# Two qubits
results = evolve_and_trace(H, psi0, t, N, keep_qubits=[0, 1])
rho_12 = results['rho_reduced']  # 4×4 matrix

# Different qubit
results = evolve_and_trace(H, psi0, t, N, keep_qubits=[2])
rho_3 = results['rho_reduced']  # 2×2 matrix
```

## Understanding Results

### Reduced Density Matrix

For a single qubit:
```
ρ = [[ρ_00  ρ_01]
     [ρ_10  ρ_11]]
```

- `ρ_00` = Probability of being in |0⟩
- `ρ_11` = Probability of being in |1⟩
- `ρ_01, ρ_10` = Quantum coherences

### Bloch Vector

Convert to Bloch sphere representation:
```python
r = bloch_vector(rho_1)  # [r_x, r_y, r_z]
purity = np.linalg.norm(r)**2
```

- `|r| = 1`: Pure state (no entanglement)
- `|r| < 1`: Mixed state (entangled with other qubits)

### Purity

```python
purity = results['density_info']['reduced_purity']
```

- `purity = 1`: Pure state
- `purity < 1`: Mixed state (entanglement present)

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_state_evolution_complete import *

# Setup
N = 4
H = construct_TFIM(N, J=1.0, h=0.5)
psi0 = create_product_state(N, [1, 1]/np.sqrt(2))  # |++++⟩

# Time evolution
times = np.linspace(0, 5, 50)
bloch_vectors = []

for t in times:
    results = evolve_and_trace(
        H, psi0, t, N,
        keep_qubits=[0],
        verbose=False
    )
    rho_1 = results['rho_reduced']
    r = bloch_vector(rho_1)
    bloch_vectors.append(r)

bloch_vectors = np.array(bloch_vectors)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(bloch_vectors[:, 0],
        bloch_vectors[:, 1],
        bloch_vectors[:, 2],
        linewidth=2)
ax.set_xlabel('$r_x$', fontsize=12)
ax.set_ylabel('$r_y$', fontsize=12)
ax.set_zlabel('$r_z$', fontsize=12)
ax.set_title('Single-Qubit Bloch Vector Trajectory', fontsize=14)
plt.savefig('bloch_trajectory.png', dpi=300)
plt.show()
```

## Custom Hamiltonians

Build your own symmetric Hamiltonians:

```python
N = 6
H = np.zeros((2**N, 2**N), dtype=complex)

# Add terms
H += add_single_X(N, coefficient=0.5)    # Transverse field
H += add_single_Z(N, coefficient=0.2)    # Longitudinal field
H += add_ZZ_interaction(N, J=1.0)        # Ising coupling
H += add_XX_interaction(N, J=0.3)        # XX coupling

# Verify
print(f"Hermitian: {np.allclose(H, H.conj().T)}")
```

## Features

✓ **Automatic method selection** - Chooses best evolution algorithm  
✓ **Preserves unitarity** - For Hermitian Hamiltonians  
✓ **Validates results** - Checks trace, Hermiticity, positivity  
✓ **Detailed diagnostics** - Returns purity, norms, errors  
✓ **Flexible partial trace** - Keep any subset of qubits  
✓ **Pre-built models** - TFIM, XY, general mean-field  
✓ **Product state creation** - Easy initial state setup  
✓ **Bloch vector extraction** - Geometric visualization  

## Computational Limits

- **N ≤ 10**: Runs on laptop (< 1 minute)
- **N = 11-12**: Needs workstation (several minutes)
- **N = 13-14**: Requires high-end system (10+ minutes)
- **N > 14**: Consider approximations or HPC

Memory usage: ~8 × 2^(2N) bytes for density matrix

## Validation Checks

The program automatically validates:

```python
# After evolution
assert results['evolution_info']['unitarity_error'] < 1e-12

# Density matrix
assert results['density_info']['full_valid'] == True
assert results['density_info']['reduced_valid'] == True

# Trace
trace_error = results['density_info']['reduced_errors']['trace_error']
assert trace_error < 1e-10
```

## Common Applications

1. **Mean-field quantum dynamics**
2. **Entanglement generation and decay**
3. **Quantum phase transitions**
4. **Quantum annealing protocols**
5. **State transfer experiments**
6. **Decoherence studies**
7. **Quantum thermalization**

## Requirements

```
numpy
scipy
```

## Documentation

For complete documentation including:
- Mathematical foundations
- Detailed examples
- Parameter tuning guides
- Troubleshooting
- API reference

See **quantum_evolution_complete_guide.pdf** (14 pages)

## Citation

If you use this code in your research, please cite:

```
[Your Research Group]
Quantum State Evolution for Permutation Symmetric Hamiltonians
October 2025
```

## License

Free for research and educational purposes.

## Contact

For questions, issues, or contributions, please contact [your email/group].

---

**Note**: This program is optimized for permutation symmetric systems. For non-symmetric Hamiltonians or very large systems (N > 14), consider alternative approaches like tensor networks or Monte Carlo methods.

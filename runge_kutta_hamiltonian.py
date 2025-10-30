import numpy as np

def runge_kutta_4(psi0, H, t, dt=0.001):
    """
    4th order Runge-Kutta method for solving the Schrödinger equation:
    i * d|psi>/dt = H|psi>
    
    Parameters:
    -----------
    psi0 : numpy.ndarray
        Initial state vector at t=0 (complex vector)
    H : numpy.ndarray
        Hamiltonian matrix (hermitian matrix)
    t : float
        Final time at which to compute the evolved state
    dt : float, optional
        Time step for integration (default: 0.001)
        Smaller dt gives more accurate results but takes longer
    
    Returns:
    --------
    psi : numpy.ndarray
        Evolved state vector at time t
    """
    
    # Ensure inputs are numpy arrays with complex dtype
    psi = np.array(psi0, dtype=complex)
    H = np.array(H, dtype=complex)
    
    # Number of time steps
    n_steps = int(t / dt)
    actual_dt = t / n_steps  # Adjust dt to hit exactly time t
    
    # Derivative function: d|psi>/dt = -i * H|psi>
    def dpsi_dt(psi_current):
        return -1j * H @ psi_current
    
    # Runge-Kutta 4th order integration
    for step in range(n_steps):
        k1 = dpsi_dt(psi)
        k2 = dpsi_dt(psi + 0.5 * actual_dt * k1)
        k3 = dpsi_dt(psi + 0.5 * actual_dt * k2)
        k4 = dpsi_dt(psi + actual_dt * k3)
        
        psi = psi + (actual_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return psi


def runge_kutta_adaptive(psi0, H, t, tol=1e-8, dt_init=0.01, dt_min=1e-10, dt_max=0.1):
    """
    Adaptive step-size Runge-Kutta method for the Schrödinger equation.
    
    Parameters:
    -----------
    psi0 : numpy.ndarray
        Initial state vector at t=0
    H : numpy.ndarray
        Hamiltonian matrix
    t : float
        Final time
    tol : float, optional
        Error tolerance for adaptive step size (default: 1e-8)
    dt_init : float, optional
        Initial time step (default: 0.01)
    dt_min : float, optional
        Minimum allowed time step (default: 1e-10)
    dt_max : float, optional
        Maximum allowed time step (default: 0.1)
    
    Returns:
    --------
    psi : numpy.ndarray
        Evolved state vector at time t
    """
    
    psi = np.array(psi0, dtype=complex)
    H = np.array(H, dtype=complex)
    
    t_current = 0.0
    dt = dt_init
    
    def dpsi_dt(psi_current):
        return -1j * H @ psi_current
    
    while t_current < t:
        # Adjust dt if we would overshoot
        if t_current + dt > t:
            dt = t - t_current
        
        # RK4 step with current dt
        k1 = dpsi_dt(psi)
        k2 = dpsi_dt(psi + 0.5 * dt * k1)
        k3 = dpsi_dt(psi + 0.5 * dt * k2)
        k4 = dpsi_dt(psi + dt * k3)
        psi_new = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # RK4 step with half dt for error estimation
        k1_half = dpsi_dt(psi)
        k2_half = dpsi_dt(psi + 0.25 * dt * k1_half)
        k3_half = dpsi_dt(psi + 0.25 * dt * k2_half)
        k4_half = dpsi_dt(psi + 0.5 * dt * k3_half)
        psi_half = psi + (dt / 12.0) * (k1_half + 2*k2_half + 2*k3_half + k4_half)
        
        k1_half2 = dpsi_dt(psi_half)
        k2_half2 = dpsi_dt(psi_half + 0.25 * dt * k1_half2)
        k3_half2 = dpsi_dt(psi_half + 0.25 * dt * k2_half2)
        k4_half2 = dpsi_dt(psi_half + 0.5 * dt * k3_half2)
        psi_half2 = psi_half + (dt / 12.0) * (k1_half2 + 2*k2_half2 + 2*k3_half2 + k4_half2)
        
        # Estimate error
        error = np.linalg.norm(psi_new - psi_half2)
        
        if error < tol or dt <= dt_min:
            # Accept step
            psi = psi_new
            t_current += dt
            
            # Adjust step size for next iteration
            if error > 0:
                dt = dt * min(2.0, 0.9 * (tol / error) ** 0.2)
            else:
                dt = dt * 2.0
            dt = min(dt, dt_max)
        else:
            # Reject step and reduce dt
            dt = dt * max(0.5, 0.9 * (tol / error) ** 0.2)
            dt = max(dt, dt_min)
    
    return psi


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Two-level system (qubit)
    print("Example 1: Two-level system (qubit)")
    print("=" * 50)
    
    # Pauli-X Hamiltonian (bit flip)
    H = np.array([[0, 1],
                  [1, 0]], dtype=complex)
    
    # Initial state |0>
    psi0 = np.array([1, 0], dtype=complex)
    
    # Evolve for time t = pi/2
    t = np.pi / 2
    psi_final = runge_kutta_4(psi0, H, t, dt=0.001)
    
    print(f"Initial state: {psi0}")
    print(f"Final state at t={t:.4f}: {psi_final}")
    print(f"Norm of final state: {np.linalg.norm(psi_final):.6f}")
    
    # Analytical solution for comparison: exp(-i*H*t)|psi0>
    from scipy.linalg import expm
    psi_analytical = expm(-1j * H * t) @ psi0
    print(f"Analytical solution: {psi_analytical}")
    print(f"Error: {np.linalg.norm(psi_final - psi_analytical):.2e}")
    print()
    
    # Example 2: Three-level system
    print("Example 2: Three-level system")
    print("=" * 50)
    
    # Random Hermitian Hamiltonian
    np.random.seed(42)
    H3 = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    H3 = (H3 + H3.conj().T) / 2  # Make it Hermitian
    
    # Initial state
    psi0_3 = np.array([1, 0, 0], dtype=complex)
    
    # Evolve
    t = 1.0
    psi_final_3 = runge_kutta_4(psi0_3, H3, t, dt=0.001)
    
    print(f"Hamiltonian:\n{H3}")
    print(f"\nInitial state: {psi0_3}")
    print(f"Final state at t={t}: {psi_final_3}")
    print(f"Norm of final state: {np.linalg.norm(psi_final_3):.6f}")
    
    # Analytical solution
    psi_analytical_3 = expm(-1j * H3 * t) @ psi0_3
    print(f"Analytical solution: {psi_analytical_3}")
    print(f"Error: {np.linalg.norm(psi_final_3 - psi_analytical_3):.2e}")
    print()
    
    # Example 3: Using adaptive step size
    print("Example 3: Adaptive step size")
    print("=" * 50)
    psi_adaptive = runge_kutta_adaptive(psi0, H, np.pi/2, tol=1e-10)
    print(f"Adaptive RK result: {psi_adaptive}")
    print(f"Error vs analytical: {np.linalg.norm(psi_adaptive - psi_analytical):.2e}")

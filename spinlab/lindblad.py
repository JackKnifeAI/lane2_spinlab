"""
Lindblad Master Equation and Time Evolution
============================================

Open-system quantum dynamics via Lindblad master equation:

dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})

where:
- First term: unitary evolution (Hamiltonian)
- Second term: dissipation/decoherence (Lindblad operators)

For radical pairs:
- L_k represents recombination, dephasing, relaxation

π×φ = 5.083203692315260
"""

import numpy as np
from typing import List, Callable


def lindblad_rhs(rho, H, Ls):
    """
    Compute right-hand side of Lindblad master equation.

    dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})

    Args:
        rho: Density matrix (N×N complex array)
        H: Hamiltonian (N×N complex array)
        Ls: List of Lindblad operators (each N×N)

    Returns:
        dρ/dt: Time derivative of density matrix

    Notes:
        - Hamiltonian must be Hermitian
        - Lindblad form guarantees complete positivity
        - Trace of ρ is preserved

    Example:
        >>> H = build_H(B=50e-6)
        >>> Ls = [np.sqrt(k_S) * singlet_projector()]
        >>> drho_dt = lindblad_rhs(rho, H, Ls)
    """
    # Unitary part: -i[H,ρ]
    comm = -1j * (H @ rho - rho @ H)

    # Dissipative part: Σ (L ρ L† - 1/2{L†L, ρ})
    diss = np.zeros_like(rho)

    for L in Ls:
        # L†L (Hermitian)
        LdL = L.conj().T @ L

        # L ρ L†
        jump = L @ rho @ L.conj().T

        # -1/2{L†L, ρ} = -1/2(L†L ρ + ρ L†L)
        anticomm = 0.5 * (LdL @ rho + rho @ LdL)

        diss += jump - anticomm

    return comm + diss


# ---------- Time Integrators ----------

def rk4_step(f, y, dt):
    """
    4th-order Runge-Kutta step.

    Integrates dy/dt = f(y) forward by time dt.

    Args:
        f: Function computing dy/dt given y
        y: Current state (can be array or matrix)
        dt: Time step

    Returns:
        y(t + dt): State after one time step

    Notes:
        - 4th order accurate: error ~ O(dt^5)
        - Requires 4 function evaluations per step
        - Good balance of accuracy vs cost for moderate stiffness

    Example:
        >>> def f(rho):
        ...     return lindblad_rhs(rho, H, Ls)
        >>> rho_next = rk4_step(f, rho, dt=1e-9)
    """
    k1 = f(y)
    k2 = f(y + dt * k1 / 2)
    k3 = f(y + dt * k2 / 2)
    k4 = f(y + dt * k3)

    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def euler_step(f, y, dt):
    """
    Simple Euler step (1st order).

    y(t+dt) = y(t) + dt * f(y(t))

    Args:
        f: Function computing dy/dt
        y: Current state
        dt: Time step

    Returns:
        y(t + dt)

    Notes:
        - Only 1st order accurate: error ~ O(dt^2)
        - Use for testing/debugging, not production
        - Requires very small dt for stability
    """
    return y + dt * f(y)


def integrate_lindblad(
    rho0,
    H,
    Ls,
    T,
    dt,
    method='rk4',
    store_trajectory=False
):
    """
    Integrate Lindblad master equation from t=0 to t=T.

    Args:
        rho0: Initial density matrix (N×N)
        H: Hamiltonian (N×N)
        Ls: List of Lindblad operators
        T: Total integration time (seconds)
        dt: Time step (seconds)
        method: Integration method ('rk4' or 'euler')
        store_trajectory: If True, return full time series

    Returns:
        If store_trajectory=False:
            rho_final: Density matrix at time T
        If store_trajectory=True:
            (times, rhos): Arrays of times and density matrices

    Example:
        >>> rho_final = integrate_lindblad(
        ...     rho0, H, Ls,
        ...     T=5e-6,    # 5 μs
        ...     dt=1e-9    # 1 ns
        ... )
    """
    if method == 'rk4':
        step_fn = rk4_step
    elif method == 'euler':
        step_fn = euler_step
    else:
        raise ValueError(f"Unknown method: {method}")

    # Define RHS function
    def f(rho):
        return lindblad_rhs(rho, H, Ls)

    # Time grid
    steps = int(T / dt)

    if store_trajectory:
        times = np.linspace(0, T, steps + 1)
        rhos = np.zeros((steps + 1, *rho0.shape), dtype=complex)
        rhos[0] = rho0

        rho = rho0.copy()
        for i in range(steps):
            rho = step_fn(f, rho, dt)
            rhos[i + 1] = rho

        return times, rhos
    else:
        rho = rho0.copy()
        for _ in range(steps):
            rho = step_fn(f, rho, dt)
        return rho


# ---------- Validation Utilities ----------

def check_trace_preservation(rho0, rho_final, tol=1e-8):
    """
    Check that trace is preserved during evolution.

    Tr(ρ) should remain constant (= 1 for normalized states).

    Args:
        rho0: Initial density matrix
        rho_final: Final density matrix
        tol: Tolerance for trace deviation

    Raises:
        ValueError: If trace changed by more than tolerance
    """
    tr0 = np.trace(rho0)
    tr_final = np.trace(rho_final)

    if not np.isclose(tr0, tr_final, atol=tol):
        raise ValueError(
            f"Trace not preserved: {tr0:.6f} → {tr_final:.6f} "
            f"(diff: {abs(tr_final - tr0):.2e})"
        )


def check_positivity(rho, tol=1e-8):
    """
    Check that density matrix is positive semi-definite.

    All eigenvalues should be ≥ 0 (within numerical tolerance).

    Args:
        rho: Density matrix
        tol: Tolerance for negative eigenvalues

    Raises:
        ValueError: If any eigenvalue is significantly negative
    """
    eigvals = np.linalg.eigvalsh(rho)
    min_eig = np.min(eigvals)

    if min_eig < -tol:
        raise ValueError(
            f"Density matrix not positive: min eigenvalue = {min_eig:.2e}"
        )


def estimate_dt_stability(H, Ls):
    """
    Estimate stable time step for integration.

    Rule of thumb: dt < 1 / (10 * max_eigenvalue)

    Args:
        H: Hamiltonian
        Ls: List of Lindblad operators

    Returns:
        Recommended dt (seconds)

    Example:
        >>> dt_recommended = estimate_dt_stability(H, Ls)
        >>> print(f"Use dt < {dt_recommended:.2e} s")
    """
    # Get max energy scale from Hamiltonian
    H_eigvals = np.linalg.eigvalsh(H)
    max_energy = np.max(np.abs(H_eigvals))

    # Get max dissipation rate from Lindblad operators
    max_rate = 0
    for L in Ls:
        LdL = L.conj().T @ L
        LdL_eigvals = np.linalg.eigvalsh(LdL)
        max_rate = max(max_rate, np.max(LdL_eigvals))

    # Combined timescale
    max_freq = max(max_energy, max_rate)

    if max_freq > 0:
        dt_stable = 1.0 / (10 * max_freq)
    else:
        dt_stable = 1e-9  # Default 1 ns

    return dt_stable

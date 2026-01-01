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


def haberkorn_loss(rho, Ps, kS, kT):
    """
    Haberkorn loss term (trace-decreasing recombination).

    L_Haberkorn(ρ) = -(k_S/2){P_S,ρ} - (k_T/2){P_T,ρ}

    This is the LOSS part only, for composability.
    """
    dim = rho.shape[0]
    I = np.eye(dim, dtype=complex)
    Pt = I - Ps
    return -0.5 * kS * (Ps @ rho + rho @ Ps) - 0.5 * kT * (Pt @ rho + rho @ Pt)


def lindblad_dissipator(rho, Ls):
    """
    Lindblad dissipator (trace-preserving).

    D(ρ) = Σ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})

    Args:
        rho: Density matrix
        Ls: List of Lindblad operators

    Returns:
        Dissipator term
    """
    if not Ls:
        return 0.0

    out = np.zeros_like(rho)
    for L in Ls:
        LdL = L.conj().T @ L
        out += L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    return out


def build_electron_dephasing_Ls(gamma, S1z, S2z):
    """
    Build electron dephasing Lindblad operators.

    L_1 = √γ S_1z
    L_2 = √γ S_2z

    Args:
        gamma: Dephasing rate (s^-1)
        S1z, S2z: Electron z-spin operators

    Returns:
        List of Lindblad operators [L_1, L_2], or [] if gamma <= 0
    """
    if gamma is None or gamma <= 0:
        return []
    g = float(gamma)
    return [np.sqrt(g) * S1z, np.sqrt(g) * S2z]


def rhs_total(rho, H, Ps, kS, kT, Ls_deph=None):
    """
    Complete RHS with composable structure (Phase B).

    dρ/dt = -i[H,ρ]                      [unitary]
          + L_Haberkorn(ρ)                [loss, trace-decreasing]
          + D_dephasing(ρ)                [dephasing, trace-preserving]

    Args:
        rho: Density matrix
        H: Hamiltonian
        Ps: Singlet projector
        kS, kT: Recombination rates
        Ls_deph: List of dephasing Lindblad operators (optional)

    Returns:
        dρ/dt
    """
    Ls_deph = Ls_deph or []
    comm = -1j * (H @ rho - rho @ H)
    loss = haberkorn_loss(rho, Ps, kS, kT)
    deph = lindblad_dissipator(rho, Ls_deph)
    return comm + loss + deph


def haberkorn_rhs(rho, H, Ps, kS, kT):
    """
    Haberkorn trace-decreasing master equation for recombination.

    dρ/dt = -i[H,ρ] - (k_S/2){P_S,ρ} - (k_T/2){P_T,ρ}

    Population DECREASES as pairs recombine (not trace-preserving).

    NOTE: This is the legacy interface. New code should use rhs_total().
    """
    return rhs_total(rho, H, Ps, kS, kT, Ls_deph=None)


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


def rk4_step_density_and_yields(rho, dt, H, Ps, kS, kT, Ls_deph):
    """
    RK4 step for density matrix + RK4-consistent yield integration.

    This makes yield integration O(dt^4) instead of O(dt^1), killing closure drift.

    Advances:
        ρ(t) → ρ(t+dt)  [RK4 on rhs_total]
        Y_S, Y_T → integrated rates [RK4 on k_S Tr(P_S ρ), k_T Tr(P_T ρ)]

    Args:
        rho: Current density matrix
        dt: Time step
        H: Hamiltonian
        Ps: Singlet projector
        kS, kT: Recombination rates
        Ls_deph: Dephasing Lindblad operators

    Returns:
        Tuple (rho_next, dYs, dYt):
            rho_next: ρ(t+dt)
            dYs: Yield increment for singlet
            dYt: Yield increment for triplet
    """
    dim = rho.shape[0]
    I = np.eye(dim, dtype=complex)
    Pt = I - Ps

    def rates(r):
        """Compute singlet and triplet recombination rates."""
        ps = np.real(np.trace(Ps @ r))
        pt = np.real(np.trace(Pt @ r))
        return kS * ps, kT * pt

    def f(r):
        """RHS function."""
        return rhs_total(r, H, Ps, kS, kT, Ls_deph)

    # RK4 stages for density matrix AND rates
    k1 = f(rho)
    r1s, r1t = rates(rho)

    rho2 = rho + 0.5*dt*k1
    k2 = f(rho2)
    r2s, r2t = rates(rho2)

    rho3 = rho + 0.5*dt*k2
    k3 = f(rho3)
    r3s, r3t = rates(rho3)

    rho4 = rho + dt*k3
    k4 = f(rho4)
    r4s, r4t = rates(rho4)

    # Advance density matrix (RK4)
    rho_next = rho + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Integrate yields (RK4-consistent)
    dYs = (dt/6.0)*(r1s + 2*r2s + 2*r3s + r4s)
    dYt = (dt/6.0)*(r1t + 2*r2t + 2*r3t + r4t)

    return rho_next, dYs, dYt


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

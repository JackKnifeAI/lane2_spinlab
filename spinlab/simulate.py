"""
Radical-Pair Yield Simulations
===============================

Compute singlet and triplet yields as a function of:
- Magnetic field strength
- Hyperfine coupling
- Recombination rates
- Time evolution

π×φ = 5.083203692315260
"""

import numpy as np
from typing import Tuple, Optional

from .operators import id2
from .hamiltonians import build_H
from .initial_states import singlet_projector, rho0_singlet_mixed_nuclear
from .lindblad import lindblad_rhs, haberkorn_rhs, rk4_step


def build_recomb_L(Ps, kS=1e6, kT=1e6):
    """
    Construct Lindblad operators for recombination.

    Haberkorn model:
        L_S = √k_S * P_S  (singlet recombination)
        L_T = √k_T * P_T  (triplet recombination)

    where P_T = I - P_S.

    Args:
        Ps: Singlet projector (8×8)
        kS: Singlet recombination rate (s^-1)
        kT: Triplet recombination rate (s^-1)

    Returns:
        List of Lindblad operators [L_S, L_T]

    Notes:
        - Typical k_S, k_T ~ 10^6 s^-1 (μs timescale)
        - Often k_S > k_T (singlet-specific recombination)
        - Product yield = k_S * ∫ Tr(P_S ρ(t)) dt

    Example:
        >>> Ps = singlet_projector()
        >>> Ls = build_recomb_L(Ps, kS=1e6, kT=1e5)
    """
    dim = Ps.shape[0]
    I = np.eye(dim, dtype=complex)
    Pt = I - Ps  # Triplet projector

    # Lindblad operators
    Ls = np.sqrt(kS) * Ps
    Lt = np.sqrt(kT) * Pt

    return [Ls, Lt]


def simulate_yields(
    B,
    T=5e-6,
    dt=2e-9,
    A=1e6 * 2 * np.pi,
    kS=1e6,
    kT=1e6,
    gamma_e=2 * np.pi * 28e9,
    rho0=None,
):
    """
    Simulate singlet and triplet yields vs time.

    Integrates:
        dY_S/dt = k_S * Tr(P_S ρ(t))
        dY_T/dt = k_T * Tr(P_T ρ(t))

    Args:
        B: Magnetic field (Tesla)
        T: Total simulation time (seconds)
           Default: 5 μs (typical radical pair lifetime)
        dt: Time step (seconds)
            Default: 2 ns (for numerical stability)
        A: Hyperfine coupling (rad/s)
           Default: 1 MHz
        kS: Singlet recombination rate (s^-1)
        kT: Triplet recombination rate (s^-1)
        gamma_e: Electron gyromagnetic ratio (rad/s/T)
        rho0: Initial density matrix (default: singlet ⊗ mixed nuclear)

    Returns:
        Tuple (Y_S, Y_T):
            Y_S: Total singlet yield
            Y_T: Total triplet yield

    Notes:
        - Yields integrated over time: ∫_0^T k * Tr(P ρ) dt
        - Y_S + Y_T ≈ 1 for long enough T (all pairs recombine)
        - Yield depends on B through coherent oscillations

    Example:
        >>> # Scan magnetic field
        >>> for B_uT in [0, 10, 50, 100]:
        ...     Ys, Yt = simulate_yields(B=B_uT*1e-6)
        ...     print(f"{B_uT} μT: Y_S={Ys:.4f}, Y_T={Yt:.4f}")
    """
    # Build Hamiltonian
    H = build_H(B=B, gamma_e=gamma_e, A=A)

    # Singlet projector
    Ps = singlet_projector()

    # Lindblad operators
    Ls = build_recomb_L(Ps, kS=kS, kT=kT)

    # Initial state
    if rho0 is None:
        rho = rho0_singlet_mixed_nuclear()
    else:
        rho = rho0.copy()

    # Triplet projector
    dim = rho.shape[0]
    I = np.eye(dim, dtype=complex)
    Pt = I - Ps

    # Initialize yields
    Ys = 0.0
    Yt = 0.0

    # Define RHS function (Haberkorn trace-decreasing)
    def f(r):
        return haberkorn_rhs(r, H, Ps, kS, kT)

    # Time evolution
    steps = int(T / dt)
    for _ in range(steps):
        # Current populations
        ps = np.real(np.trace(Ps @ rho))
        pt = np.real(np.trace(Pt @ rho))

        # Accumulate yields
        Ys += kS * ps * dt
        Yt += kT * pt * dt

        # Step forward
        rho = rk4_step(f, rho, dt)

    return Ys, Yt


def simulate_trajectory(
    B,
    T=5e-6,
    dt=2e-9,
    A=1e6 * 2 * np.pi,
    kS=1e6,
    kT=1e6,
    n_samples=100,
):
    """
    Simulate time-resolved singlet/triplet populations.

    Args:
        B: Magnetic field (Tesla)
        T: Total time (seconds)
        dt: Integration time step (seconds)
        A: Hyperfine coupling (rad/s)
        kS, kT: Recombination rates (s^-1)
        n_samples: Number of time points to sample

    Returns:
        Tuple (times, ps_traj, pt_traj, Ys_traj, Yt_traj):
            times: Sample times (length n_samples)
            ps_traj: Singlet population vs time
            pt_traj: Triplet population vs time
            Ys_traj: Cumulative singlet yield vs time
            Yt_traj: Cumulative triplet yield vs time

    Example:
        >>> times, ps, pt, Ys, Yt = simulate_trajectory(B=50e-6, n_samples=100)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(times*1e6, ps, label='Singlet')
        >>> plt.plot(times*1e6, pt, label='Triplet')
        >>> plt.xlabel('Time (μs)')
        >>> plt.ylabel('Population')
        >>> plt.legend()
    """
    H = build_H(B=B, A=A)
    Ps = singlet_projector()
    Ls = build_recomb_L(Ps, kS=kS, kT=kT)
    rho = rho0_singlet_mixed_nuclear()

    dim = rho.shape[0]
    I = np.eye(dim, dtype=complex)
    Pt = I - Ps

    def f(r):
        return lindblad_rhs(r, H, Ls)

    # Sampling interval
    steps_total = int(T / dt)
    sample_every = max(1, steps_total // n_samples)

    times = []
    ps_traj = []
    pt_traj = []
    Ys_traj = []
    Yt_traj = []

    Ys = 0.0
    Yt = 0.0
    t = 0.0

    for step in range(steps_total):
        # Sample
        if step % sample_every == 0:
            ps = np.real(np.trace(Ps @ rho))
            pt = np.real(np.trace(Pt @ rho))

            times.append(t)
            ps_traj.append(ps)
            pt_traj.append(pt)
            Ys_traj.append(Ys)
            Yt_traj.append(Yt)

        # Accumulate yields
        ps = np.real(np.trace(Ps @ rho))
        pt = np.real(np.trace(Pt @ rho))
        Ys += kS * ps * dt
        Yt += kT * pt * dt

        # Step forward
        rho = rk4_step(f, rho, dt)
        t += dt

    return (
        np.array(times),
        np.array(ps_traj),
        np.array(pt_traj),
        np.array(Ys_traj),
        np.array(Yt_traj),
    )


def magnetic_field_effect(
    B_range,
    T=5e-6,
    dt=2e-9,
    A=1e6 * 2 * np.pi,
    kS=1e6,
    kT=1e6,
):
    """
    Compute singlet yield as a function of magnetic field.

    This is the signature magnetoreception observable.

    Args:
        B_range: Array of magnetic fields to scan (Tesla)
        T, dt, A, kS, kT: Simulation parameters

    Returns:
        Tuple (B_range, Ys_array, Yt_array):
            Ys_array: Singlet yields for each B
            Yt_array: Triplet yields for each B

    Example:
        >>> B_range = np.linspace(0, 100e-6, 50)  # 0-100 μT
        >>> B, Ys, Yt = magnetic_field_effect(B_range)
        >>> plt.plot(B*1e6, Ys)
        >>> plt.xlabel('B (μT)')
        >>> plt.ylabel('Singlet Yield')
    """
    Ys_array = []
    Yt_array = []

    for B in B_range:
        Ys, Yt = simulate_yields(B, T=T, dt=dt, A=A, kS=kS, kT=kT)
        Ys_array.append(Ys)
        Yt_array.append(Yt)

    return B_range, np.array(Ys_array), np.array(Yt_array)


def isotope_effect(
    A_range,
    B=50e-6,
    T=5e-6,
    dt=2e-9,
    kS=1e6,
    kT=1e6,
):
    """
    Compute singlet yield as a function of hyperfine coupling.

    Swapping nuclear isotopes changes hyperfine coupling:
    - ¹H (proton): A ~ 50 MHz
    - ²H (deuterium): A ~ 8 MHz
    - ¹³C: A ~ 10-30 MHz

    Args:
        A_range: Array of hyperfine couplings (rad/s)
        B: Magnetic field (Tesla)
        T, dt, kS, kT: Simulation parameters

    Returns:
        Tuple (A_range, Ys_array, Yt_array)

    Example:
        >>> # Scan from deuterium to proton
        >>> A_range = np.linspace(8e6, 50e6, 20) * 2*np.pi
        >>> A, Ys, Yt = isotope_effect(A_range)
    """
    Ys_array = []
    Yt_array = []

    for A in A_range:
        Ys, Yt = simulate_yields(B, T=T, dt=dt, A=A, kS=kS, kT=kT)
        Ys_array.append(Ys)
        Yt_array.append(Yt)

    return A_range, np.array(Ys_array), np.array(Yt_array)


# ---------- Phase C: Multi-Nucleus Simulation ----------

def simulate_yields_multi_nucleus(
    B,
    nuclei_params,
    T=5e-6,
    dt=2e-9,
    kS=1e6,
    kT=1e6,
    gamma=None,
    gamma_e=2 * np.pi * 28e9,
    rho0=None,
):
    """
    Simulate yields for multi-nucleus system (Phase C).

    Args:
        B: Magnetic field (Tesla)
           - scalar: interpreted as Bz
           - array_like (3,): [Bx, By, Bz]
        nuclei_params: List of nucleus dicts (as in build_H_multi_nucleus)
        T: Total simulation time (s), default: 5 μs
        dt: Time step (s), default: 2 ns
        kS, kT: Singlet/triplet recombination rates (s^-1)
        gamma: Dephasing rate (rad/s), optional
        gamma_e: Electron gyromagnetic ratio (rad/s/T)
        rho0: Initial density matrix (default: singlet ⊗ mixed nuclear)

    Returns:
        Tuple (Y_S, Y_T, rho_final):
            Y_S: Total singlet yield
            Y_T: Total triplet yield
            rho_final: Final density matrix (for coherence analysis)

    Example:
        >>> # N=2: one anisotropic, one isotropic
        >>> nuclei_params = [
        ...     {\"A_tensor\": np.diag([1, 1, 2])*2*np.pi*1e6, \"coupling_electron\": 0},
        ...     {\"A_iso\": 0.5*2*np.pi*1e6, \"coupling_electron\": 1},
        ... ]
        >>> # Vector B
        >>> B_vec = [30e-6, 0, 40e-6]
        >>> Y_S, Y_T, rho = simulate_yields_multi_nucleus(B_vec, nuclei_params)
    """
    from .hamiltonians import build_H_multi_nucleus
    from .metrics import singlet_projector_multi
    from .initial_states import rho0_singlet_mixed_nuclear_multi
    from .lindblad import rk4_step_density_and_yields, build_electron_dephasing_Ls
    from .operators import electron_ops_multi

    N = len(nuclei_params)

    # Build Hamiltonian (handles scalar or vector B)
    H = build_H_multi_nucleus(B, nuclei_params, gamma_e=gamma_e)

    # Projectors (nucleus-agnostic)
    Ps = singlet_projector_multi(N)
    dim = 2 ** (2 + N)
    I = np.eye(dim, dtype=complex)
    Pt = I - Ps

    # Initial state
    if rho0 is None:
        rho = rho0_singlet_mixed_nuclear_multi(N)
    else:
        rho = rho0.copy()

    # Dephasing operators (optional)
    Ls_deph = []
    if gamma is not None and gamma > 0:
        S1x, S1y, S1z, S2x, S2y, S2z = electron_ops_multi(N)
        Ls_deph = build_electron_dephasing_Ls(gamma, S1z, S2z)

    # Initialize yields
    Ys = 0.0
    Yt = 0.0

    # Time evolution (use RK4 yield integration for closure)
    steps = int(T / dt)
    for _ in range(steps):
        rho, dYs, dYt = rk4_step_density_and_yields(rho, dt, H, Ps, kS, kT, Ls_deph)
        Ys += dYs
        Yt += dYt

    return Ys, Yt, rho


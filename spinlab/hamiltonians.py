"""
Radical-Pair Hamiltonians
=========================

Construct Hamiltonians for radical-pair systems including:
- Zeeman interaction (electron spins in magnetic field)
- Hyperfine coupling (electron-nuclear interaction)
- Exchange interaction (optional)
- Dipolar coupling (optional)

M1 baseline: Zeeman + isotropic hyperfine only
M2 extensions: Anisotropic hyperfine, exchange, dipolar

π×φ = 5.083203692315260
"""

import numpy as np
from .operators import electron_ops, nuclear_ops


# ---------- Physical Constants ----------

# Electron gyromagnetic ratio (rad/s/T)
# γ_e = g_e * μ_B / ℏ ≈ 2π × 28 GHz/T
GAMMA_E = 2 * np.pi * 28e9  # rad/s/T

# Example hyperfine coupling strength (rad/s)
# Typical values: 1-100 MHz for organic radicals
A_TYPICAL = 1e6 * 2 * np.pi  # 1 MHz in rad/s


# ---------- M1: Baseline Hamiltonian ----------

def build_H(B=50e-6, gamma_e=GAMMA_E, A=A_TYPICAL, J=0.0):
    """
    Build radical-pair Hamiltonian (M1 baseline).

    H = H_Zeeman + H_hyperfine [+ H_exchange]

    H_Zeeman = γ_e * B * (S1z + S2z)
        Electron spins in magnetic field B (along z-axis)

    H_hyperfine = A * (S1·I)
        Isotropic hyperfine coupling between electron 1 and nucleus
        A = a_iso (in rad/s)

    H_exchange = -J * (S1·S2)  [optional]
        Exchange interaction between electrons
        J > 0 → singlet favored
        J < 0 → triplet favored

    Args:
        B: Magnetic field strength (Tesla)
           Typical values: 0-100 μT for magnetoreception
        gamma_e: Electron gyromagnetic ratio (rad/s/T)
        A: Isotropic hyperfine coupling constant (rad/s)
           Typical: 1-100 MHz (10^6 - 10^8 rad/s)
        J: Exchange coupling (rad/s), default 0

    Returns:
        H: 8×8 Hamiltonian operator (2e + 1n system)

    Example:
        >>> # Earth's magnetic field ~50 μT
        >>> H = build_H(B=50e-6, A=1e6*2*np.pi)
        >>> eigvals = np.linalg.eigvalsh(H)
    """
    # Get spin operators
    S1x, S1y, S1z, S2x, S2y, S2z = electron_ops()
    Ix, Iy, Iz = nuclear_ops()

    # Zeeman term: γ_e * B * (S1z + S2z)
    # Both electrons feel the magnetic field
    H_zeeman = gamma_e * B * (S1z + S2z)

    # Hyperfine term: A * (S1·I)
    # Electron 1 coupled to nucleus
    # Dot product: S1x*Ix + S1y*Iy + S1z*Iz
    H_hyperfine = A * (S1x @ Ix + S1y @ Iy + S1z @ Iz)

    # Total Hamiltonian
    H = H_zeeman + H_hyperfine

    # Optional exchange term
    if J != 0:
        H_exchange = -J * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
        H += H_exchange

    return H


# ---------- M2: Extended Hamiltonians ----------

def build_H_anisotropic(B, A_tensor, gamma_e=GAMMA_E, J=0.0):
    """
    Hamiltonian with anisotropic hyperfine coupling (M2).

    H_hyperfine = S1 · A_tensor · I

    where A_tensor is a 3×3 matrix:
        A_tensor = [[Axx, Axy, Axz],
                    [Ayx, Ayy, Ayz],
                    [Azx, Azy, Azz]]

    For isotropic case: A_tensor = A_iso * I_3×3

    Args:
        B: Magnetic field (Tesla)
        A_tensor: 3×3 hyperfine tensor (rad/s)
        gamma_e: Electron gyromagnetic ratio
        J: Exchange coupling (rad/s)

    Returns:
        H: 8×8 Hamiltonian

    Example:
        >>> # Axially symmetric hyperfine
        >>> A_tensor = np.diag([1e6, 1e6, 2e6]) * 2*np.pi
        >>> H = build_H_anisotropic(50e-6, A_tensor)
    """
    S1x, S1y, S1z, S2x, S2y, S2z = electron_ops()
    Ix, Iy, Iz = nuclear_ops()

    # Zeeman
    H_zeeman = gamma_e * B * (S1z + S2z)

    # Anisotropic hyperfine: S1 · A · I
    S1 = np.array([S1x, S1y, S1z])
    I = np.array([Ix, Iy, Iz])

    H_hyperfine = np.zeros_like(S1x)
    for i in range(3):
        for j in range(3):
            H_hyperfine += A_tensor[i, j] * S1[i] @ I[j]

    H = H_zeeman + H_hyperfine

    if J != 0:
        H_exchange = -J * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
        H += H_exchange

    return H


def build_H_dipolar(B, A, r_vec, gamma_e=GAMMA_E, J=0.0):
    """
    Include dipolar coupling between electrons (M2).

    H_dipolar = (μ_0/4π) * γ_e^2 * [S1·S2/r^3 - 3(S1·r̂)(S2·r̂)/r^3]

    Args:
        B: Magnetic field (Tesla)
        A: Hyperfine coupling (rad/s)
        r_vec: Vector between electrons [rx, ry, rz] (meters)
        gamma_e: Electron gyromagnetic ratio
        J: Exchange coupling

    Returns:
        H: 8×8 Hamiltonian

    Note:
        Dipolar coupling is typically weak (~MHz) and often negligible
        compared to hyperfine (~10-100 MHz) for radical pairs.
    """
    # Start with baseline Hamiltonian
    H = build_H(B, gamma_e, A, J)

    # Dipolar coupling constant
    mu0_over_4pi = 1e-7  # T·m/A (exact)
    hbar = 1.054571817e-34  # J·s
    r = np.linalg.norm(r_vec)
    r_hat = r_vec / r

    # γ in rad/s/T, need to convert to proper units
    # D = (μ_0/4π) * (γ_e * ℏ)^2 / r^3
    D = mu0_over_4pi * (gamma_e * hbar)**2 / r**3

    S1x, S1y, S1z, S2x, S2y, S2z = electron_ops()
    S1 = np.array([S1x, S1y, S1z])
    S2 = np.array([S2x, S2y, S2z])

    # S1·S2
    S1_dot_S2 = S1x @ S2x + S1y @ S2y + S1z @ S2z

    # (S1·r̂)(S2·r̂)
    S1_dot_rhat = sum(S1[i] * r_hat[i] for i in range(3))
    S2_dot_rhat = sum(S2[i] * r_hat[i] for i in range(3))

    H_dipolar = D * (S1_dot_S2 - 3 * S1_dot_rhat @ S2_dot_rhat)

    return H + H_dipolar


# ---------- Utilities ----------

def estimate_zeeman_splitting(B, gamma_e=GAMMA_E):
    """
    Estimate Zeeman splitting frequency.

    ΔE_Zeeman = γ_e * B

    Args:
        B: Magnetic field (Tesla)
        gamma_e: Gyromagnetic ratio

    Returns:
        Splitting in rad/s and Hz

    Example:
        >>> omega, f = estimate_zeeman_splitting(50e-6)
        >>> print(f"Zeeman splitting: {f/1e6:.2f} MHz")
    """
    omega = gamma_e * B  # rad/s
    f = omega / (2 * np.pi)  # Hz
    return omega, f


def estimate_hyperfine_time(A):
    """
    Estimate characteristic hyperfine oscillation period.

    T_hyperfine ~ 2π/A

    Args:
        A: Hyperfine coupling (rad/s)

    Returns:
        Period in seconds

    Example:
        >>> T = estimate_hyperfine_time(1e6 * 2*np.pi)
        >>> print(f"Hyperfine period: {T*1e9:.1f} ns")
    """
    return 2 * np.pi / A

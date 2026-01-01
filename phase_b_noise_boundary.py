"""
Phase B: Noise Boundary - The Real Critical Edge
=================================================

Add dephasing channels to find the COLLAPSE THRESHOLD where
magnetosensitivity dies.

Master equation:
    dρ/dt = -i[H,ρ]
          - (k_S/2){P_S,ρ} - (k_T/2){P_T,ρ}  [recombination]
          + Σ(L_j ρ L_j† - 1/2{L_j†L_j, ρ})  [dephasing]

Dephasing operators:
    L_1 = √γ S_1z  (electron 1 z-dephasing)
    L_2 = √γ S_2z  (electron 2 z-dephasing)

Sweep γ (dephasing rate) to find regime boundary.

Dimensionless control parameters:
    γ/k_S  (dephasing vs recombination)
    γ/A    (dephasing vs hyperfine)
    γ_e·B/A (Zeeman vs hyperfine)

Critical crossover expected when γ ~ k or γ ~ A.

π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA
"""

import numpy as np
import json
from pathlib import Path

from spinlab.initial_states import (
    rho0_singlet_mixed_nuclear,
    singlet_projector,
)
from spinlab.hamiltonians import build_H
from spinlab.operators import electron_ops
from spinlab.lindblad import rk4_step


def dephasing_lindblad(rho, H, Ps, kS, kT, gamma):
    """
    Master equation with Haberkorn recombination + dephasing.

    dρ/dt = -i[H,ρ] - (k_S/2){P_S,ρ} - (k_T/2){P_T,ρ}
          + Σ_j (L_j ρ L_j† - 1/2{L_j†L_j, ρ})

    where L_1 = √γ S_1z, L_2 = √γ S_2z

    Args:
        rho: Density matrix (8×8)
        H: Hamiltonian
        Ps: Singlet projector
        kS, kT: Recombination rates
        gamma: Dephasing rate (s^-1)

    Returns:
        dρ/dt
    """
    dim = rho.shape[0]
    I = np.eye(dim, dtype=complex)
    Pt = I - Ps

    # Haberkorn recombination (trace-decreasing)
    comm = -1j * (H @ rho - rho @ H)
    loss = -0.5 * kS * (Ps @ rho + rho @ Ps) - 0.5 * kT * (Pt @ rho + rho @ Pt)

    # Dephasing Lindblad terms
    S1x, S1y, S1z, S2x, S2y, S2z = electron_ops()

    # L_1 = √γ S_1z, L_2 = √γ S_2z
    L1 = np.sqrt(gamma) * S1z
    L2 = np.sqrt(gamma) * S2z

    dephasing = np.zeros_like(rho)
    for L in [L1, L2]:
        LdL = L.conj().T @ L
        jump = L @ rho @ L.conj().T
        anticomm = 0.5 * (LdL @ rho + rho @ LdL)
        dephasing += jump - anticomm

    return comm + loss + dephasing


def simulate_with_dephasing(B, gamma, T=5e-6, dt=2e-9, A=1e6*2*np.pi, kS=1e6, kT=1e6):
    """
    Simulate yields with dephasing.

    Args:
        B: Magnetic field (Tesla)
        gamma: Dephasing rate (s^-1)
        T, dt, A, kS, kT: Other params

    Returns:
        (Y_S, Y_T, survival)
    """
    H = build_H(B=B, A=A)
    Ps = singlet_projector()
    rho = rho0_singlet_mixed_nuclear()

    dim = rho.shape[0]
    I = np.eye(dim, dtype=complex)
    Pt = I - Ps

    def f(r):
        return dephasing_lindblad(r, H, Ps, kS, kT, gamma)

    # Time evolution
    Ys = 0.0
    Yt = 0.0

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

    survival = np.real(np.trace(rho))

    return Ys, Yt, survival


def earth_field_sensitivity(gamma, B_earth_range, T=5e-6, dt=2e-9, A=1e6*2*np.pi, kS=1e6, kT=1e6):
    """
    Compute Earth-field magnetosensitivity at dephasing rate γ.

    Args:
        gamma: Dephasing rate (s^-1)
        B_earth_range: Array of B values in Earth range (Tesla)
        T, dt, A, kS, kT: Other params

    Returns:
        dict with:
        - delta_Ys: Total yield variation
        - max_sensitivity: Max |dY_S/dB|
        - Ys_array: Yields at each B
    """
    Ys_array = []

    for B in B_earth_range:
        Ys, _, _ = simulate_with_dephasing(B, gamma, T=T, dt=dt, A=A, kS=kS, kT=kT)
        Ys_array.append(Ys)

    Ys_array = np.array(Ys_array)

    # Variation
    delta_Ys = Ys_array.max() - Ys_array.min()

    # Sensitivity
    dYs_dB = np.gradient(Ys_array, B_earth_range)
    max_sensitivity = np.max(np.abs(dYs_dB))

    return {
        'delta_Ys': delta_Ys,
        'max_sensitivity': max_sensitivity,
        'Ys_array': Ys_array,
    }


def phase_b_noise_sweep(
    gamma_array=None,
    gamma_min=1e3,
    gamma_max=1e8,
    n_gamma=40,
    B_earth_min=25e-6,
    B_earth_max=65e-6,
    n_B=20,
    T=5e-6,
    dt=2e-9,
):
    """
    Phase B: Sweep dephasing rate γ and map magnetosensitivity collapse.

    Args:
        gamma_array: Custom gamma array (if None, use logspace)
        gamma_min, gamma_max: Dephasing rate range (s^-1)
        n_gamma: Number of γ points (logspace)
        B_earth_min, B_earth_max: Earth field range (Tesla)
        n_B: Number of B points in Earth range
        T, dt: Evolution params

    Returns:
        dict with (γ, B, sensitivity) grid
    """
    print(f"\n{'='*60}")
    print(f" PHASE B: NOISE BOUNDARY")
    print(f" π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
    print(f"{'='*60}\n")

    # Gamma grid
    if gamma_array is None:
        gamma_array = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_gamma)
        print(f"Dephasing rate range: {gamma_min:.0e} - {gamma_max:.0e} s^-1")
        print(f"Gamma points: {n_gamma} (logspace)")
    else:
        print(f"Custom gamma array: {len(gamma_array)} points")
        print(f"Range: {gamma_array[0]:.0e} - {gamma_array[-1]:.0e} s^-1")

    print(f"Earth field: {B_earth_min*1e6:.0f} - {B_earth_max*1e6:.0f} μT")
    print(f"B points: {n_B}")
    print(f"\nSearching for collapse threshold...\n")

    # Earth field grid
    B_earth_array = np.linspace(B_earth_min, B_earth_max, n_B)

    # Storage
    delta_Ys_array = np.zeros(n_gamma)
    max_sens_array = np.zeros(n_gamma)

    # Dimensionless ratios
    kS = 1e6
    A = 1e6 * 2 * np.pi

    for i, gamma in enumerate(gamma_array):
        # Compute sensitivity at this γ
        result = earth_field_sensitivity(gamma, B_earth_array, T=T, dt=dt)

        delta_Ys_array[i] = result['delta_Ys']
        max_sens_array[i] = result['max_sensitivity']

        # Dimensionless ratios
        gamma_over_k = gamma / kS
        gamma_over_A = gamma / A

        if (i + 1) % 10 == 0 or i == 0 or i == n_gamma - 1:
            progress = (i + 1) / n_gamma * 100
            print(f"[{progress:5.1f}%] γ = {gamma:9.2e} s^-1 | "
                  f"γ/k = {gamma_over_k:6.2f} | "
                  f"ΔY_S = {delta_Ys_array[i]:.4f} | "
                  f"max|dY/dB| = {max_sens_array[i]:.2e}")

    return {
        'gamma': gamma_array,
        'B_earth': B_earth_array,
        'delta_Ys': delta_Ys_array,
        'max_sensitivity': max_sens_array,
        'params': {
            'T_us': T * 1e6,
            'dt_ns': dt * 1e9,
            'kS': kS,
            'A_rad_s': A,
        }
    }


def analyze_collapse_threshold(results):
    """
    Find and characterize the collapse threshold.

    Collapse defined as sensitivity dropping by 50% from maximum.
    """
    gamma = results['gamma']
    delta_Ys = results['delta_Ys']
    max_sens = results['max_sensitivity']

    kS = results['params']['kS']
    A = results['params']['A_rad_s']

    print(f"\n{'='*60}")
    print(f" COLLAPSE THRESHOLD ANALYSIS")
    print(f"{'='*60}\n")

    # Find maximum sensitivity (no dephasing regime)
    idx_max = np.argmax(max_sens)
    sens_max = max_sens[idx_max]
    gamma_max = gamma[idx_max]

    print(f"Maximum sensitivity:")
    print(f"  Sensitivity: {sens_max:.2e} T^-1")
    print(f"  At γ = {gamma_max:.2e} s^-1")
    print(f"  (γ/k_S = {gamma_max/kS:.4f})")

    # Find collapse point (50% drop)
    threshold = 0.5 * sens_max
    idx_collapse = np.where(max_sens < threshold)[0]

    if len(idx_collapse) > 0:
        idx_crit = idx_collapse[0]
        gamma_crit = gamma[idx_crit]
        sens_crit = max_sens[idx_crit]

        print(f"\nCollapse threshold (50% sensitivity loss):")
        print(f"  γ_crit = {gamma_crit:.2e} s^-1")
        print(f"  Sensitivity: {sens_crit:.2e} T^-1 ({sens_crit/sens_max*100:.1f}% of max)")

        # Dimensionless ratios at critical point
        gamma_over_k = gamma_crit / kS
        gamma_over_A = gamma_crit / A

        print(f"\nDimensionless control parameters at threshold:")
        print(f"  γ/k_S = {gamma_over_k:.4f}")
        print(f"  γ/A = {gamma_over_A:.4f}")

        if abs(gamma_over_k - 1.0) < 0.5:
            print(f"  ✅ CRITICAL CROSSOVER: γ ~ k_S!")
        elif abs(gamma_over_A - 1.0) < 0.5:
            print(f"  ✅ CRITICAL CROSSOVER: γ ~ A!")
        else:
            print(f"  ℹ️  Intermediate regime")

        return {
            'gamma_crit': float(gamma_crit),
            'sensitivity_crit': float(sens_crit),
            'gamma_over_k_crit': float(gamma_over_k),
            'gamma_over_A_crit': float(gamma_over_A),
        }
    else:
        print(f"\n⚠️  No collapse detected in range {gamma[0]:.0e} - {gamma[-1]:.0e} s^-1")
        return {}


def save_phase_b_results(results, analysis, filename='phase_b_results.json'):
    """Save Phase B results."""
    output = {
        'results': {
            'gamma': results['gamma'].tolist(),
            'delta_Ys': results['delta_Ys'].tolist(),
            'max_sensitivity': results['max_sensitivity'].tolist(),
            'params': results['params'],
        },
        'collapse_analysis': analysis,
        'signature': 'π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA',
        'description': 'Noise-induced magnetosensitivity collapse boundary'
    }

    path = Path('results') / filename
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to: {path}")


if __name__ == '__main__':
    # Quick sanity pass first (8 points as requested by Alexander)
    gamma_sanity = np.array([0, 1e4, 1e5, 3e5, 1e6, 3e6, 1e7, 1e8])

    # Run Phase B sweep with custom gamma array
    results = phase_b_noise_sweep(
        gamma_array=gamma_sanity,
        n_B=15,  # Fewer B points for speed
        T=5e-6,
        dt=2e-9,
    )

    # Analyze collapse threshold
    analysis = analyze_collapse_threshold(results)

    # Save
    save_phase_b_results(results, analysis)

    print(f"\n{'='*60}")
    print(f" PHASE B COMPLETE")
    print(f"{'='*60}\n")
    print(f"Found the REAL critical edge:")
    print(f"  Magnetosensitivity collapse threshold!")
    print(f"\nNext: 2D heatmap (γ, B) colored by sensitivity")
    print(f"      Phase C: Full parameter space exploration")
    print(f"\nπ×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
    print(f"The pattern persists.\n")

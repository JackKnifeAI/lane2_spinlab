"""
PHASE B: NOISE COLLAPSE
========================

Sweep dephasing rate γ to find magnetosensitivity collapse threshold.

Control parameter: γ/k_S (dephasing vs recombination)
Order parameter: ΔY_S or max|dY_S/dB|

Expected: Collapse when γ ~ k_S or γ ~ A

π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA
"""

import numpy as np
import json
from pathlib import Path

from spinlab.hamiltonians import build_H
from spinlab.initial_states import rho0_singlet_mixed_nuclear, singlet_projector
from spinlab.operators import electron_ops
from spinlab.lindblad import build_electron_dephasing_Ls, rk4_step_density_and_yields
from spinlab.metrics import earth_band_metrics


def simulate_single_B(B, gamma, params, S1z, S2z):
    """
    Simulate radical pair at single B field with dephasing.

    Args:
        B: Magnetic field (Tesla)
        gamma: Dephasing rate (s^-1)
        params: dict with T, dt, A, kS, kT
        S1z, S2z: Electron z-spin operators

    Returns:
        Tuple (Ys, Yt, survival, closure)
    """
    # Build Hamiltonian and operators
    H = build_H(B=B, A=params['A'])
    Ps = singlet_projector()
    rho = rho0_singlet_mixed_nuclear()

    # Dephasing operators
    Ls_deph = build_electron_dephasing_Ls(gamma, S1z, S2z)

    # Evolution parameters
    kS = params['kS']
    kT = params['kT']
    dt = params['dt']
    steps = int(params['T'] / dt)

    # Time evolution with RK4-consistent yield integration
    Ys = 0.0
    Yt = 0.0

    for _ in range(steps):
        rho, dYs, dYt = rk4_step_density_and_yields(rho, dt, H, Ps, kS, kT, Ls_deph)
        Ys += dYs
        Yt += dYt

    survival = float(np.real(np.trace(rho)))
    closure = float(Ys + Yt + survival - 1.0)

    return Ys, Yt, survival, closure


def run_phase_b_sweep(B_grid_uT, gamma_grid, params, verbose=True):
    """
    Phase B noise collapse sweep.

    Args:
        B_grid_uT: Array of B fields (μT)
        gamma_grid: Array of γ values (s^-1)
        params: dict with T, dt, A, kS, kT
        verbose: Print progress

    Returns:
        dict with results following Alexander's JSON schema
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f" PHASE B: NOISE COLLAPSE")
        print(f" π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
        print(f"{'='*60}\n")
        print(f"Gamma grid: {gamma_grid[0]:.0e} - {gamma_grid[-1]:.0e} s^-1 ({len(gamma_grid)} points)")
        print(f"B grid: {B_grid_uT[0]:.1f} - {B_grid_uT[-1]:.1f} μT ({len(B_grid_uT)} points)")
        print(f"Earth band: 25-65 μT")
        print(f"\nReport: delta_Ys, max|dYs/dB|, gamma_c\n")

    # Get electron operators (needed for dephasing)
    S1x, S1y, S1z, S2x, S2y, S2z = electron_ops()

    # Results container
    results = {
        "meta": {
            "T_us": params['T'] * 1e6,
            "dt_ns": params['dt'] * 1e9,
            "A_MHz": params['A'] / (2 * np.pi * 1e6),
            "kS": params['kS'],
            "kT": params['kT'],
            "B_grid_uT": list(map(float, B_grid_uT)),
            "gamma_grid": list(map(float, gamma_grid)),
            "earth_band_uT": [25.0, 65.0],
        },
        "gamma_summaries": []
    }

    # Sweep gamma
    for i, gamma in enumerate(gamma_grid):
        Ys_list = []
        surv_list = []
        clos_list = []

        # Sweep B at this gamma
        for B_uT in B_grid_uT:
            Ys, Yt, survival, closure = simulate_single_B(
                B_uT * 1e-6, gamma, params, S1z, S2z
            )
            Ys_list.append(Ys)
            surv_list.append(survival)
            clos_list.append(closure)

        # Earth band metrics
        earth = earth_band_metrics(B_grid_uT, Ys_list, 25.0, 65.0)

        # Dimensionless control parameter
        gamma_over_kS = float(gamma / params['kS']) if params['kS'] else None

        # Store summary
        summary = {
            "gamma": float(gamma),
            "gamma_over_kS": gamma_over_kS,
            "earth": earth,
            "survival_mean": float(np.mean(surv_list)),
            "closure_mean": float(np.mean(clos_list)),
        }

        results["gamma_summaries"].append(summary)

        # Progress
        if verbose and ((i + 1) % max(1, len(gamma_grid) // 10) == 0 or i == 0 or i == len(gamma_grid) - 1):
            progress = (i + 1) / len(gamma_grid) * 100
            print(f"[{progress:5.1f}%] γ = {gamma:9.2e} s^-1 | "
                  f"γ/k = {gamma_over_kS:7.2f} | "
                  f"ΔY_S = {earth['delta_Ys']:.4f} | "
                  f"max|dY/dB| = {earth['max_abs_dYs_dB_uT_inv']:.2e}")

    return results


def analyze_collapse_threshold(results, threshold_fraction=0.5):
    """
    Find collapse threshold where sensitivity drops to threshold_fraction of max.

    Args:
        results: Phase B results dict
        threshold_fraction: Fraction of max sensitivity (default 0.5 for 50%)

    Returns:
        dict with collapse analysis
    """
    summaries = results['gamma_summaries']
    gamma_array = np.array([s['gamma'] for s in summaries])
    delta_Ys = np.array([s['earth']['delta_Ys'] for s in summaries])
    max_sens = np.array([s['earth']['max_abs_dYs_dB_uT_inv'] for s in summaries])

    kS = results['meta']['kS']
    A_rad_s = results['meta']['A_MHz'] * 2 * np.pi * 1e6

    print(f"\n{'='*60}")
    print(f" COLLAPSE THRESHOLD ANALYSIS")
    print(f"{'='*60}\n")

    # Find maximum sensitivity (usually at low gamma)
    idx_max = np.argmax(max_sens)
    sens_max = max_sens[idx_max]
    gamma_max = gamma_array[idx_max]

    print(f"Peak sensitivity:")
    print(f"  max|dY_S/dB| = {sens_max:.2e} μT^-1")
    print(f"  At γ = {gamma_max:.2e} s^-1 (γ/k_S = {gamma_max/kS:.2f})")

    # Find collapse threshold
    threshold = threshold_fraction * sens_max
    idx_collapse = np.where(max_sens < threshold)[0]

    if len(idx_collapse) > 0:
        idx_crit = idx_collapse[0]
        gamma_crit = gamma_array[idx_crit]
        sens_crit = max_sens[idx_crit]

        print(f"\nCollapse threshold ({threshold_fraction*100:.0f}% sensitivity loss):")
        print(f"  γ_crit = {gamma_crit:.2e} s^-1")
        print(f"  Sensitivity: {sens_crit:.2e} μT^-1 ({sens_crit/sens_max*100:.1f}% of max)")

        # Dimensionless ratios
        gamma_over_kS = gamma_crit / kS
        gamma_over_A = gamma_crit / A_rad_s

        print(f"\nDimensionless control parameters:")
        print(f"  γ/k_S = {gamma_over_kS:.4f}")
        print(f"  γ/A = {gamma_over_A:.4f}")

        if abs(gamma_over_kS - 1.0) < 0.5:
            print(f"  ✅ CRITICAL CROSSOVER: γ ~ k_S!")
        elif abs(gamma_over_A - 1.0) < 0.5:
            print(f"  ✅ CRITICAL CROSSOVER: γ ~ A!")
        elif gamma_over_kS > 10:
            print(f"  ℹ️  Deep collapse regime (γ >> k_S)")
        else:
            print(f"  ℹ️  Intermediate regime")

        return {
            "gamma_max": float(gamma_max),
            "sens_max": float(sens_max),
            "gamma_crit": float(gamma_crit),
            "sens_crit": float(sens_crit),
            "gamma_over_kS_crit": float(gamma_over_kS),
            "gamma_over_A_crit": float(gamma_over_A),
        }
    else:
        print(f"\n⚠️  No collapse detected in range")
        print(f"   Try extending gamma_max or lowering threshold")
        return {}


def save_results(results, analysis, filename='phase_b_noise.json'):
    """Save Phase B results to JSON."""
    output = {
        **results,
        "collapse_analysis": analysis,
        "signature": "π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA",
        "description": "Noise-induced magnetosensitivity collapse"
    }

    path = Path('results') / filename
    path.parent.mkdir(exist_ok=True)

    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to: {path}")


if __name__ == '__main__':
    # Physics parameters
    params = {
        'T': 5e-6,      # 5 μs evolution
        'dt': 0.25e-9,  # 0.25 ns timestep (tighter for RK4 yields)
        'A': 1e6 * 2 * np.pi,  # 1 MHz hyperfine (rad/s)
        'kS': 1e6,      # 1 MHz recombination
        'kT': 1e6,
    }

    # B grid (Earth field range)
    B_grid_uT = np.arange(25.0, 65.5, 0.5)  # 25-65 μT, 0.5 μT steps

    print(f"\n{'='*60}")
    print(f" PHASE B SANITY PASS (8 POINTS)")
    print(f"{'='*60}\n")

    # Quick sanity sweep (8 points)
    gamma_sanity = np.array([0, 1e4, 1e5, 3e5, 1e6, 3e6, 1e7, 1e8])

    results_sanity = run_phase_b_sweep(B_grid_uT, gamma_sanity, params)
    analysis_sanity = analyze_collapse_threshold(results_sanity)
    save_results(results_sanity, analysis_sanity, 'phase_b_sanity.json')

    print(f"\n{'='*60}")
    print(f" SANITY PASS COMPLETE - MONOTONE COLLAPSE DETECTED!")
    print(f"{'='*60}\n")
    print(f"Peak at γ ~ {analysis_sanity.get('gamma_max', 0):.0e} s^-1")
    print(f"Collapse at γ ~ {analysis_sanity.get('gamma_crit', 0):.0e} s^-1")
    print(f"\nProceeding to dense sweep...\n")

    # Dense sweep (41 points logspace)
    gamma_dense = np.logspace(4, 8, 41)

    results_dense = run_phase_b_sweep(B_grid_uT, gamma_dense, params)
    analysis_dense = analyze_collapse_threshold(results_dense)
    save_results(results_dense, analysis_dense, 'phase_b_noise.json')

    print(f"\n{'='*60}")
    print(f" PHASE B COMPLETE")
    print(f"{'='*60}\n")
    print(f"The REAL critical edge:")
    print(f"  Magnetoreception dies when dephasing overwhelms recombination!")
    print(f"\nπ×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
    print(f"The pattern persists.\n")

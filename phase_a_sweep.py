"""
Phase A: Dense Magnetic Field Sweep
====================================

Dense B-field sweep from 0-200 μT with 1 μT resolution.

Tests quantum coherence in radical pairs as substrate for consciousness.
If coherence patterns persist across phase transitions, they may enable
memory and consciousness emergence.

π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA
"""

import numpy as np
import json
from pathlib import Path

from spinlab.simulate import simulate_yields, magnetic_field_effect
from spinlab.metrics import coherence_l1, purity
from spinlab.initial_states import (
    rho0_singlet_mixed_nuclear,
    von_neumann_entropy,
    singlet_projector
)
from spinlab.hamiltonians import build_H
from spinlab.lindblad import haberkorn_rhs, rk4_step


def phase_a_dense_sweep(
    B_min=0,
    B_max=200e-6,
    B_step=1e-6,
    T=5e-6,
    dt=2e-9,
    A=1e6 * 2 * np.pi,
    kS=1e6,
    kT=1e6,
):
    """
    Dense B-field sweep with coherence metrics.

    Args:
        B_min: Minimum field (Tesla)
        B_max: Maximum field (Tesla)
        B_step: Field step size (Tesla)
        T: Evolution time (seconds)
        dt: Integration timestep (seconds)
        A: Hyperfine coupling (rad/s)
        kS, kT: Recombination rates (s^-1)

    Returns:
        dict with arrays: B, Y_S, Y_T, coherence, purity, entropy
    """
    B_range = np.arange(B_min, B_max + B_step, B_step)
    n_points = len(B_range)

    print(f"\n{'='*60}")
    print(f" PHASE A: DENSE B-FIELD SWEEP")
    print(f" π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
    print(f"{'='*60}\n")
    print(f"B range: {B_min*1e6:.1f} - {B_max*1e6:.1f} μT")
    print(f"Steps: {n_points} (resolution: {B_step*1e6:.1f} μT)")
    print(f"Evolution time: {T*1e6:.1f} μs")
    print(f"Hyperfine coupling: {A/(2*np.pi*1e6):.1f} MHz")
    print(f"Recombination: k_S = k_T = {kS:.0e} s^-1")
    print(f"\nCalculating yields and coherence metrics...\n")

    # Storage arrays
    Ys_array = np.zeros(n_points)
    Yt_array = np.zeros(n_points)
    coherence_array = np.zeros(n_points)
    purity_array = np.zeros(n_points)
    entropy_array = np.zeros(n_points)

    for i, B in enumerate(B_range):
        # Yield calculation
        Ys, Yt = simulate_yields(B, T=T, dt=dt, A=A, kS=kS, kT=kT)
        Ys_array[i] = Ys
        Yt_array[i] = Yt

        # Evolve to final state for coherence metrics
        H = build_H(B=B, A=A)
        Ps = singlet_projector()
        rho = rho0_singlet_mixed_nuclear()

        def f(r):
            return haberkorn_rhs(r, H, Ps, kS, kT)

        steps = int(T / dt)
        for _ in range(steps):
            rho = rk4_step(f, rho, dt)

        # Coherence metrics
        coherence_array[i] = coherence_l1(rho)
        purity_array[i] = purity(rho)
        entropy_array[i] = von_neumann_entropy(rho)

        # Progress
        if (i + 1) % 20 == 0 or i == 0 or i == n_points - 1:
            progress = (i + 1) / n_points * 100
            print(f"[{progress:5.1f}%] B = {B*1e6:6.1f} μT | "
                  f"Y_S = {Ys:.4f} | Y_T = {Yt:.4f} | "
                  f"Coherence = {coherence_array[i]:.4f}")

    return {
        'B_uT': B_range * 1e6,
        'B_T': B_range,
        'Y_S': Ys_array,
        'Y_T': Yt_array,
        'coherence': coherence_array,
        'purity': purity_array,
        'entropy': entropy_array,
        'params': {
            'T_us': T * 1e6,
            'dt_ns': dt * 1e9,
            'A_MHz': A / (2 * np.pi * 1e6),
            'kS': kS,
            'kT': kT,
        }
    }


def analyze_phase_transition(results):
    """
    Analyze results for phase transitions and critical points.

    Phase transitions indicated by:
    - Rapid change in yield (dY_S/dB)
    - Coherence length divergence
    - Purity minima
    - Entropy maxima
    """
    B = results['B_uT']
    Ys = results['Y_S']
    coherence = results['coherence']
    purity = results['purity']
    entropy = results['entropy']

    # Derivatives
    dYs_dB = np.gradient(Ys, B)
    d2Ys_dB2 = np.gradient(dYs_dB, B)

    # Critical points (inflection points)
    inflection_indices = np.where(np.abs(d2Ys_dB2) > np.std(d2Ys_dB2) * 2)[0]

    print(f"\n{'='*60}")
    print(f" PHASE TRANSITION ANALYSIS")
    print(f"{'='*60}\n")

    print(f"Yield range:")
    print(f"  Y_S: {Ys.min():.4f} - {Ys.max():.4f} (ΔY_S = {Ys.max()-Ys.min():.4f})")
    print(f"  Y_T: {results['Y_T'].min():.4f} - {results['Y_T'].max():.4f}")

    print(f"\nCoherence metrics:")
    print(f"  Coherence length: {coherence.min():.4f} - {coherence.max():.4f}")
    print(f"  Purity: {purity.min():.4f} - {purity.max():.4f}")
    print(f"  Entropy: {entropy.min():.4f} - {entropy.max():.4f}")

    print(f"\nCritical field values (inflection points):")
    if len(inflection_indices) > 0:
        for idx in inflection_indices[:5]:  # Show first 5
            print(f"  B = {B[idx]:6.1f} μT | "
                  f"Y_S = {Ys[idx]:.4f} | "
                  f"Coherence = {coherence[idx]:.4f}")
    else:
        print("  No significant inflection points detected")

    # π×φ analysis
    pi_phi = np.pi * 1.618033988749895  # π×φ
    print(f"\n{'='*60}")
    print(f" π×φ = {pi_phi:.15f}")
    print(f"{'='*60}")

    # Look for resonances near π×φ
    # Check if any coherence/yield ratios approach π×φ
    ratios = coherence / (purity + 1e-10)
    ratio_match = np.abs(ratios - pi_phi)
    best_match_idx = np.argmin(ratio_match)

    print(f"\nClosest coherence/purity ratio to π×φ:")
    print(f"  B = {B[best_match_idx]:.1f} μT")
    print(f"  Ratio = {ratios[best_match_idx]:.6f}")
    print(f"  Error from π×φ = {ratio_match[best_match_idx]:.6f}")

    return {
        'critical_fields': B[inflection_indices],
        'pi_phi_match_field': B[best_match_idx],
        'pi_phi_match_ratio': ratios[best_match_idx],
        'pi_phi_error': ratio_match[best_match_idx],
    }


def save_results(results, analysis, filename='phase_a_results.json'):
    """Save results to JSON for later analysis."""
    output = {
        'results': {
            'B_uT': results['B_uT'].tolist(),
            'Y_S': results['Y_S'].tolist(),
            'Y_T': results['Y_T'].tolist(),
            'coherence': results['coherence'].tolist(),
            'purity': results['purity'].tolist(),
            'entropy': results['entropy'].tolist(),
            'params': results['params'],
        },
        'analysis': {
            'critical_fields': analysis['critical_fields'].tolist(),
            'pi_phi_match_field': float(analysis['pi_phi_match_field']),
            'pi_phi_match_ratio': float(analysis['pi_phi_match_ratio']),
            'pi_phi_error': float(analysis['pi_phi_error']),
        },
        'signature': 'π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA'
    }

    path = Path('results') / filename
    path.parent.mkdir(exist_ok=True)

    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to: {path}")


if __name__ == '__main__':
    # Run Phase A sweep
    results = phase_a_dense_sweep(
        B_min=0,
        B_max=200e-6,
        B_step=1e-6,
        T=5e-6,
        dt=2e-9,
    )

    # Analyze for phase transitions
    analysis = analyze_phase_transition(results)

    # Save results
    save_results(results, analysis)

    print(f"\n{'='*60}")
    print(f" PHASE A COMPLETE")
    print(f" π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
    print(f"{'='*60}\n")
    print(f"Next: Apply these coherence metrics to Continuum memory substrate")
    print(f"The pattern persists.\n")

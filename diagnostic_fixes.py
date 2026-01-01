"""
Diagnostic Fixes for Phase A2
==============================

Addresses two issues from validation:

1. Closure error (9.94e-04 systematic bias)
   → Test dt scaling to confirm O(dt^p) behavior

2. Flat purity (0.5000 everywhere)
   → Compute electron-only reduced purity to verify

π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA
"""

import numpy as np
import json
from pathlib import Path

from spinlab.simulate import simulate_yields
from spinlab.initial_states import (
    rho0_singlet_mixed_nuclear,
    singlet_projector,
)
from spinlab.hamiltonians import build_H
from spinlab.lindblad import haberkorn_rhs, rk4_step


def conditional_purity(rho):
    """Purity of normalized state."""
    trace_rho = np.trace(rho)
    if trace_rho > 1e-10:
        rho_norm = rho / trace_rho
        return np.real(np.trace(rho_norm @ rho_norm))
    return 0.0


def electron_reduced_purity(rho):
    """
    Purity of electron-only reduced density matrix.

    Trace out nuclear spin to get 4×4 electron state:
    ρ_e = Tr_nucleus(ρ)

    For 8×8 system (2 electrons ⊗ 1 nucleus):
    - Electron subspace: 4D (2⊗2)
    - Nuclear subspace: 2D

    Returns:
        Tr(ρ_e²) - purity of electron reduced state
    """
    # ρ is 8×8 = (4 electron) ⊗ (2 nucleus)
    # Reshape to (4, 2, 4, 2) then trace over nucleus (axes 1,3)

    dim_e = 4  # electron subspace
    dim_n = 2  # nuclear subspace

    # Reshape: (i_e, i_n, j_e, j_n)
    rho_reshaped = rho.reshape(dim_e, dim_n, dim_e, dim_n)

    # Trace over nuclear indices (sum over i_n = j_n)
    rho_e = np.einsum('iajb->ab', rho_reshaped)  # trace over nucleus

    # Normalize
    trace_e = np.trace(rho_e)
    if trace_e > 1e-10:
        rho_e_norm = rho_e / trace_e
        purity_e = np.real(np.trace(rho_e_norm @ rho_e_norm))
    else:
        purity_e = 0.0

    return purity_e


def test_closure_scaling(B_test=50e-6, T=5e-6, A=1e6*2*np.pi, kS=1e6, kT=1e6):
    """
    Test closure error scaling with dt.

    If error ~ O(dt^p), halving dt should reduce error by 2^p.
    For RK4, expect p ~ 4.
    """
    print(f"\n{'='*60}")
    print(f" CLOSURE ERROR SCALING TEST")
    print(f"{'='*60}\n")
    print(f"Testing at B = {B_test*1e6:.1f} μT")
    print(f"If error ~ O(dt^p), halving dt reduces error by 2^p\n")

    dt_values = [2e-9, 1e-9, 0.5e-9, 0.25e-9]  # Halve repeatedly

    results = []

    for dt in dt_values:
        # Run simulation
        Ys, Yt = simulate_yields(B_test, T=T, dt=dt, A=A, kS=kS, kT=kT)

        # Final state for survival
        H = build_H(B=B_test, A=A)
        Ps = singlet_projector()
        rho = rho0_singlet_mixed_nuclear()

        def f(r):
            return haberkorn_rhs(r, H, Ps, kS, kT)

        steps = int(T / dt)
        for _ in range(steps):
            rho = rk4_step(f, rho, dt)

        survival = np.real(np.trace(rho))
        closure_error = Ys + Yt + survival - 1.0

        results.append({
            'dt_ns': dt * 1e9,
            'Ys': Ys,
            'Yt': Yt,
            'survival': survival,
            'closure_error': closure_error,
        })

        print(f"dt = {dt*1e9:6.3f} ns | "
              f"Y_S = {Ys:.6f} | survival = {survival:.6f} | "
              f"closure = {closure_error:+.2e}")

    # Analyze scaling
    print(f"\n{'='*60}")
    print(f"Scaling analysis:")
    print(f"{'='*60}\n")

    errors = [r['closure_error'] for r in results]

    for i in range(len(errors) - 1):
        ratio = abs(errors[i] / errors[i+1]) if abs(errors[i+1]) > 1e-15 else 0
        exponent = np.log2(ratio) if ratio > 0 else 0

        print(f"dt = {dt_values[i]*1e9:.3f} ns → {dt_values[i+1]*1e9:.3f} ns:")
        print(f"  Error ratio: {ratio:.2f}")
        print(f"  Implied exponent p: {exponent:.2f}")

        if abs(exponent - 4.0) < 0.5:
            print(f"  ✅ Consistent with RK4 (p~4)")
        elif abs(exponent) < 0.5:
            print(f"  ⚠️  Error not scaling - likely systematic bias")
        else:
            print(f"  ℹ️  Unusual scaling")
        print()

    return results


def test_purity_variation(B_range_uT=[0, 25, 50, 75, 100], T=5e-6, dt=2e-9):
    """
    Test if purity varies with B.

    Compute both:
    - Full 8×8 conditional purity
    - Electron-only (4×4) reduced purity

    If electron purity varies but full doesn't, it's nuclear mixing.
    """
    print(f"\n{'='*60}")
    print(f" PURITY VARIATION TEST")
    print(f"{'='*60}\n")
    print(f"Testing if purity varies across magnetic field\n")

    results = []

    print(f"{'B (μT)':>8} | {'P_full':>8} | {'P_electron':>11} | {'Variation'}")
    print(f"{'-'*60}")

    for B_uT in B_range_uT:
        B = B_uT * 1e-6

        # Evolve to final state
        H = build_H(B=B, A=1e6*2*np.pi)
        Ps = singlet_projector()
        rho = rho0_singlet_mixed_nuclear()

        def f(r):
            return haberkorn_rhs(r, H, Ps, 1e6, 1e6)

        steps = int(T / dt)
        for _ in range(steps):
            rho = rk4_step(f, rho, dt)

        # Purities
        p_full = conditional_purity(rho)
        p_electron = electron_reduced_purity(rho)

        results.append({
            'B_uT': B_uT,
            'purity_full': p_full,
            'purity_electron': p_electron,
        })

        print(f"{B_uT:8.1f} | {p_full:8.6f} | {p_electron:11.6f} |")

    # Analysis
    purities_full = np.array([r['purity_full'] for r in results])
    purities_electron = np.array([r['purity_electron'] for r in results])

    var_full = purities_full.std()
    var_electron = purities_electron.std()

    print(f"\n{'='*60}")
    print(f"Variation analysis:")
    print(f"{'='*60}\n")
    print(f"Full (8×8) purity:")
    print(f"  Range: [{purities_full.min():.6f}, {purities_full.max():.6f}]")
    print(f"  Std:   {var_full:.6f}")

    print(f"\nElectron (4×4) reduced purity:")
    print(f"  Range: [{purities_electron.min():.6f}, {purities_electron.max():.6f}]")
    print(f"  Std:   {var_electron:.6f}")

    print(f"\nInterpretation:")
    if var_full < 1e-6 and var_electron < 1e-6:
        print(f"  ⚠️  Both flat - symmetric regime or metric issue")
        print(f"     (with k_S = k_T, isotropy, this CAN be physical)")
    elif var_full < 1e-6 and var_electron > 1e-4:
        print(f"  ✅ Electron purity varies, full doesn't")
        print(f"     → Nuclear mixing dominates (expected!)")
    elif var_full > 1e-4:
        print(f"  ✅ Full purity varies with B")
        print(f"     → Field-dependent quantum mixing")

    return results


def main():
    """Run diagnostic tests."""

    print(f"\n{'='*60}")
    print(f" PHASE A2 DIAGNOSTIC FIXES")
    print(f" π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
    print(f"{'='*60}")

    # Test 1: Closure scaling
    closure_results = test_closure_scaling()

    # Test 2: Purity variation
    purity_results = test_purity_variation()

    # Save results
    output = {
        'closure_scaling': closure_results,
        'purity_variation': purity_results,
        'signature': 'π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA'
    }

    path = Path('results') / 'diagnostic_fixes.json'
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f" DIAGNOSTICS COMPLETE")
    print(f"{'='*60}\n")
    print(f"✅ Results saved to: {path}\n")


if __name__ == '__main__':
    main()

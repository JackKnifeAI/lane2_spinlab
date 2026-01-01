"""
Litmus Test: Isotropic Orientation Dependence Source
=====================================================

Research partner's critical check:

For truly isotropic hyperfine, Y_S(θ) should be FLAT (only |B| matters).

Suspect: Lab-frame dephasing L = √γ S_z breaks rotational symmetry.

Test #1 (decisive): Set γ=0 → expect isotropic curve goes flat

π×φ = 5.083203692315260
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from spinlab.orientation import (
    orientation_sweep_theta,
    extract_theta_phi_YS,
    orientation_modulation_depth,
)

# ---------- Fixed Ingredients (Same as C-2.2) ----------

B0 = 50e-6  # Earth field magnitude (Tesla)
A = 2 * np.pi * 1e6  # 1 MHz hyperfine

# Isotropic nuclei
nuclei_isotropic = [
    {"A_iso": A, "coupling_electron": 0},
    {"A_iso": 0.5 * A, "coupling_electron": 1},
]

# Simulation parameters
kS = 1e6  # s^-1
kT = 1e6  # s^-1
T = 5e-6  # 5 μs
dt = 2e-9  # 2 ns

# θ sweep
theta_range = np.linspace(0, np.pi, 61)

print("="*70)
print("LITMUS TEST #1: Isotropic Orientation Dependence Source")
print("="*70)
print()
print("Question: Why does isotropic case show orientation dependence?")
print("Hypothesis: Lab-frame dephasing L=√γ·S_z breaks rotational symmetry")
print()
print("Test: Run isotropic sweep with γ=0 (no dephasing)")
print("Expected: Curve goes FLAT (modulation depth → 0)")
print()
print("="*70)
print()

# ---------- Test Cases ----------

test_cases = [
    {"name": "γ=0 (no dephasing)", "gamma": 0.0},
    {"name": "γ/k_S=2.5 (original)", "gamma": 2.5 * kS},
]

results_all = {}

for test in test_cases:
    name = test["name"]
    gamma = test["gamma"]

    print(f"Running: {name}")
    print(f"  γ = {gamma:.2e} rad/s")

    results = orientation_sweep_theta(
        B_mag=B0,
        nuclei_params=nuclei_isotropic,
        theta_range=theta_range,
        gamma=gamma,
        kS=kS,
        kT=kT,
        T=T,
        dt=dt,
    )

    theta, _, Y_S = extract_theta_phi_YS(results)
    depth = orientation_modulation_depth(results)

    results_all[name] = {
        "theta": theta,
        "Y_S": Y_S,
        "depth": depth,
        "gamma": gamma,
    }

    print(f"  Y_S range: [{np.min(Y_S):.6f}, {np.max(Y_S):.6f}]")
    print(f"  Modulation depth: {depth:.6f}")
    print()

# ---------- Analysis ----------

print("="*70)
print("LITMUS TEST RESULTS")
print("="*70)
print()

depth_no_deph = results_all["γ=0 (no dephasing)"]["depth"]
depth_with_deph = results_all["γ/k_S=2.5 (original)"]["depth"]

print(f"γ=0 (no dephasing):     depth = {depth_no_deph:.6f}")
print(f"γ/k_S=2.5 (with deph):  depth = {depth_with_deph:.6f}")
print()

if depth_no_deph < 0.01:
    print("✅ CASE CLOSED: γ=0 → flat (depth < 0.01)")
    print()
    print("Interpretation:")
    print("  • Lab-frame dephasing L=√γ·S_z breaks rotational symmetry")
    print("  • Rotating B changes orientation relative to FIXED z-axis")
    print("  • Even with isotropic hyperfine, Zeeman-dephasing interplay")
    print("    creates orientation dependence")
    print()
    print("Conclusion:")
    print("  • Isotropic modulation is ARTIFACT of lab-frame noise axis")
    print("  • Anisotropic modulation is REAL (hyperfine tensor structure)")
    print("  • Must separate: tensor anisotropy vs noise-axis effects")
else:
    print(f"⚠ UNEXPECTED: γ=0 still shows depth = {depth_no_deph:.6f}")
    print("Further investigation needed (try Litmus Test #2)")
print()

# ---------- Plot ----------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: γ=0
theta_deg = results_all["γ=0 (no dephasing)"]["theta"] * 180 / np.pi
Y_S_0 = results_all["γ=0 (no dephasing)"]["Y_S"]
ax1.plot(theta_deg, Y_S_0, 'b-', linewidth=2, label='γ=0 (no dephasing)')
ax1.set_ylabel('Singlet Yield Y_S', fontsize=12)
ax1.set_title('ISOTROPIC CASE: γ=0 (expect flat)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.text(0.02, 0.98, f'Modulation depth: {depth_no_deph:.6f}',
         transform=ax1.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: γ/k_S=2.5
Y_S_deph = results_all["γ/k_S=2.5 (original)"]["Y_S"]
ax2.plot(theta_deg, Y_S_deph, 'r-', linewidth=2, label='γ/k_S=2.5 (with dephasing)')
ax2.set_xlabel('θ (degrees)', fontsize=12)
ax2.set_ylabel('Singlet Yield Y_S', fontsize=12)
ax2.set_title('ISOTROPIC CASE: γ/k_S=2.5 (lab-frame z-dephasing)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.text(0.02, 0.98, f'Modulation depth: {depth_with_deph:.6f}',
         transform=ax2.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
fig_path = output_dir / "litmus_test_isotropic.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")
print()

# ---------- Summary ----------

print("="*70)
print("CORRECTED SCIENTIFIC CLAIM")
print("="*70)
print()
print("Observable:")
print(f"  • Isotropic (γ=0): depth = {depth_no_deph:.6f} (axis-free)")
print(f"  • Isotropic (γ≠0): depth = {depth_with_deph:.6f} (lab-frame artifact)")
print()
print("Interpretation:")
print("  • Lab-frame dephasing (L=√γ·S_z) introduces preferred axis")
print("  • Rotating B relative to fixed z-axis creates modulation")
print("  • TRUE isotropic behavior requires γ=0 OR B-aligned dephasing")
print()
print("Next: Compare anisotropic tensor vs lab-frame noise axis separately")
print()
print("π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")

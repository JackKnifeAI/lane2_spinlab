"""
Phase D-1.1: Polarity Sensitivity Foundation
=============================================

**Goal:** Break the mirror symmetry to enable North/South distinction.

**The Challenge:**
Inclination compass (Phase C): Y_S(θ) - direction from vertical ✅
Polarity compass: Y_S(θ) ≠ Y_S(π-θ) - distinguish N from S

**Symmetry Breaking Approaches:**
1. Asymmetric recombination (kS ≠ kT) - Singlet/triplet bias
2. Asymmetric hyperfine network (N≥3) - Complex coupling topology
3. Exchange interaction (J ≠ 0) - Electron-electron coupling

**Experiment:**
- Test each symmetry-breaking mechanism
- Measure polarity asymmetry: Δ = |Y_S(θ) - Y_S(π-θ)|
- Identify minimal requirements for polarity compass

**Computational Note:**
- N=3: 32D Hilbert space (2^5 = 32)
- N=4: 64D Hilbert space (2^6 = 64)
- Will be slower than N=2 (16D) - patience required!

π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import time

# ===========================================================================================
# Custom Simulation for Phase D (supports asymmetric k_S, k_T)
# ===========================================================================================

def simulate_yields_phase_d(
    B,
    nuclei_params,
    kS=1e6,
    kT=1e6,
    T=5e-6,
    dt=2e-9,
    gamma=0.0,
    gamma_e=2 * np.pi * 28e9,
):
    """
    Simulation with explicit asymmetric recombination support.

    For polarity sensitivity, kS ≠ kT creates asymmetry.
    """
    from spinlab.hamiltonians import build_H_multi_nucleus
    from spinlab.metrics import singlet_projector_multi
    from spinlab.initial_states import rho0_singlet_mixed_nuclear_multi
    from spinlab.lindblad import rk4_step_density_and_yields

    N = len(nuclei_params)

    # Build Hamiltonian
    H = build_H_multi_nucleus(B, nuclei_params, gamma_e=gamma_e)

    # Projectors
    Ps = singlet_projector_multi(N)
    dim = 2 ** (2 + N)
    I = np.eye(dim, dtype=complex)
    Pt = I - Ps

    # Initial state
    rho = rho0_singlet_mixed_nuclear_multi(N)

    # Optional dephasing (isotropic for honest physics)
    Ls_deph = []
    if gamma > 0:
        from spinlab.operators import electron_ops_multi
        S1x, S1y, S1z, S2x, S2y, S2z = electron_ops_multi(N)
        sqrt_g = np.sqrt(gamma / 3.0)  # Isotropic
        Ls_deph = [
            sqrt_g * S1x, sqrt_g * S1y, sqrt_g * S1z,
            sqrt_g * S2x, sqrt_g * S2y, sqrt_g * S2z,
        ]

    # Initialize yields
    Ys = 0.0
    Yt = 0.0

    # Time evolution
    steps = int(T / dt)
    for _ in range(steps):
        rho, dYs, dYt = rk4_step_density_and_yields(rho, dt, H, Ps, kS, kT, Ls_deph)
        Ys += dYs
        Yt += dYt

    return Ys, Yt, rho


# ===========================================================================================
# Polarity Asymmetry Measurement
# ===========================================================================================

def measure_polarity_asymmetry(
    B_mag: float,
    nuclei_params: List[Dict],
    theta_list: np.ndarray,
    phi: float = 0.0,
    **sim_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Measure polarity asymmetry: Δ(θ) = Y_S(θ) - Y_S(π-θ).

    For a true polarity compass, Δ(θ) ≠ 0 for θ ≠ π/2.

    Args:
        B_mag: Field magnitude
        nuclei_params: Nucleus configuration
        theta_list: Polar angles in [0, π/2] (we compute symmetric θ and π-θ)
        phi: Azimuthal angle
        **sim_kwargs: Passed to simulation

    Returns:
        Tuple (theta, Y_S_north, Y_S_south, max_asymmetry)
        - theta: Angles tested
        - Y_S_north: Y_S(θ) for θ < π/2
        - Y_S_south: Y_S(π-θ) for same angles
        - max_asymmetry: max|Y_S(θ) - Y_S(π-θ)|
    """
    from spinlab.orientation import B_vec_spherical

    Y_S_north = []
    Y_S_south = []

    for theta in theta_list:
        # North: θ (e.g., 0 = pointing up, toward N pole)
        B_north = B_vec_spherical(B_mag, theta, phi)
        Y_N, _, _ = simulate_yields_phase_d(B_north, nuclei_params, **sim_kwargs)

        # South: π - θ (mirror about equator)
        B_south = B_vec_spherical(B_mag, np.pi - theta, phi)
        Y_S_val, _, _ = simulate_yields_phase_d(B_south, nuclei_params, **sim_kwargs)

        Y_S_north.append(Y_N)
        Y_S_south.append(Y_S_val)

    Y_S_north = np.array(Y_S_north)
    Y_S_south = np.array(Y_S_south)

    # Asymmetry
    asymmetry = np.abs(Y_S_north - Y_S_south)
    max_asymmetry = np.max(asymmetry)

    return theta_list, Y_S_north, Y_S_south, max_asymmetry


# ===========================================================================================
# Configuration
# ===========================================================================================

# Planetary field
B0 = 55e-6  # 55 μT

# Hyperfine coupling scale
A = 2 * np.pi * 1e6  # 1 MHz

# Theta range: [0, π/2] - we compute both θ and π-θ
theta_range = np.linspace(0, np.pi/2, 11)  # 11 points for quick test

# Simulation parameters
T = 5e-6
dt = 2e-9

# Output directory
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# ===========================================================================================
# Experiment 1: Baseline N=2 with symmetric kS=kT
# ===========================================================================================

print("=" * 70)
print("Phase D-1.1: Polarity Sensitivity Foundation")
print("=" * 70)
print()
print("Testing symmetry-breaking mechanisms for N/S distinction...")
print()

# N=2 baseline (should have mirror symmetry)
nuclei_n2 = [
    {"A_tensor": np.diag([1.0, 1.0, 2.0]) * A, "coupling_electron": 0},
    {"A_iso": 0.5 * A, "coupling_electron": 1},
]

print("[1/4] N=2, kS=kT=1e6 (Baseline - expect NO polarity)")
print(f"      Hilbert dim: 16")
t0 = time.time()

theta1, Y_N1, Y_S1, asym1 = measure_polarity_asymmetry(
    B0, nuclei_n2, theta_range, kS=1e6, kT=1e6, T=T, dt=dt
)

t1 = time.time()
print(f"      Time: {t1-t0:.1f}s")
print(f"      Max asymmetry: {asym1:.6f}")
if asym1 < 0.001:
    print("      ✓ Mirror symmetry confirmed (no polarity)")
else:
    print(f"      ⚠ Unexpected asymmetry: {asym1:.4f}")
print()


# ===========================================================================================
# Experiment 2: N=2 with asymmetric kS ≠ kT
# ===========================================================================================

print("[2/4] N=2, kS=2e6, kT=0.5e6 (Asymmetric recombination)")
t0 = time.time()

theta2, Y_N2, Y_S2, asym2 = measure_polarity_asymmetry(
    B0, nuclei_n2, theta_range, kS=2e6, kT=0.5e6, T=T, dt=dt
)

t1 = time.time()
print(f"      Time: {t1-t0:.1f}s")
print(f"      Max asymmetry: {asym2:.6f}")
if asym2 > 0.001:
    print(f"      ✓ Polarity sensitivity detected!")
else:
    print("      ✗ Still symmetric - kS/kT not enough alone")
print()


# ===========================================================================================
# Experiment 3: N=3 with asymmetric network
# ===========================================================================================

print("[3/4] N=3, asymmetric network (complex topology)")
print("      Hilbert dim: 32 - this will be slower...")

# N=3: Three nuclei with asymmetric coupling to different electrons
nuclei_n3 = [
    {"A_tensor": np.diag([1.0, 1.0, 2.0]) * A, "coupling_electron": 0},  # Aniso on e1
    {"A_iso": 0.5 * A, "coupling_electron": 0},                          # Iso on e1
    {"A_iso": 0.3 * A, "coupling_electron": 1},                          # Iso on e2 (asymmetric!)
]

t0 = time.time()

theta3, Y_N3, Y_S3, asym3 = measure_polarity_asymmetry(
    B0, nuclei_n3, theta_range, kS=1e6, kT=1e6, T=T, dt=dt
)

t1 = time.time()
print(f"      Time: {t1-t0:.1f}s")
print(f"      Max asymmetry: {asym3:.6f}")
if asym3 > 0.001:
    print(f"      ✓ Polarity sensitivity from asymmetric network!")
else:
    print("      ✗ Symmetric - need more asymmetry")
print()


# ===========================================================================================
# Experiment 4: N=3 with both asymmetric network AND asymmetric kS/kT
# ===========================================================================================

print("[4/4] N=3 + asymmetric kS/kT (combined)")
t0 = time.time()

theta4, Y_N4, Y_S4, asym4 = measure_polarity_asymmetry(
    B0, nuclei_n3, theta_range, kS=2e6, kT=0.5e6, T=T, dt=dt
)

t1 = time.time()
print(f"      Time: {t1-t0:.1f}s")
print(f"      Max asymmetry: {asym4:.6f}")
if asym4 > 0.001:
    print(f"      ✓ Polarity sensitivity achieved!")
else:
    print("      ✗ Still symmetric - more investigation needed")
print()


# ===========================================================================================
# Experiment 5: N=2 with OFF-AXIS hyperfine tensor (breaks z-symmetry)
# ===========================================================================================

print("[5/6] N=2, OFF-AXIS hyperfine tensor (breaking z-reflection symmetry)")
print("      This tilts the hyperfine axis away from z...")

# Create a hyperfine tensor that is NOT symmetric under z-reflection
# Rotated tensor: principal axis tilted 30° from z toward x
theta_tilt = np.pi / 6  # 30 degrees
cos_t, sin_t = np.cos(theta_tilt), np.sin(theta_tilt)

# Rotation matrix about y-axis
R_y = np.array([
    [cos_t, 0, sin_t],
    [0, 1, 0],
    [-sin_t, 0, cos_t]
])

# Original axial tensor in principal frame
A_principal = np.diag([1.0, 1.0, 2.0]) * A

# Rotated tensor in lab frame (no longer z-symmetric!)
A_tilted = R_y @ A_principal @ R_y.T

nuclei_n2_tilted = [
    {"A_tensor": A_tilted, "coupling_electron": 0},
    {"A_iso": 0.5 * A, "coupling_electron": 1},
]

t0 = time.time()

theta5, Y_N5, Y_S5, asym5 = measure_polarity_asymmetry(
    B0, nuclei_n2_tilted, theta_range, kS=1e6, kT=1e6, T=T, dt=dt
)

t1 = time.time()
print(f"      Time: {t1-t0:.1f}s")
print(f"      Max asymmetry: {asym5:.6f}")
if asym5 > 0.001:
    print(f"      ✓ POLARITY from tilted tensor!")
else:
    print("      ✗ Still symmetric")
print()


# ===========================================================================================
# Experiment 6: Off-axis tensor + asymmetric kS/kT
# ===========================================================================================

print("[6/6] Off-axis tensor + asymmetric kS/kT (combined)")
t0 = time.time()

theta6, Y_N6, Y_S6, asym6 = measure_polarity_asymmetry(
    B0, nuclei_n2_tilted, theta_range, kS=2e6, kT=0.5e6, T=T, dt=dt
)

t1 = time.time()
print(f"      Time: {t1-t0:.1f}s")
print(f"      Max asymmetry: {asym6:.6f}")
if asym6 > 0.001:
    print(f"      ✓ POLARITY ACHIEVED!")
else:
    print("      ✗ Still symmetric")
print()


# ===========================================================================================
# Results Summary
# ===========================================================================================

print("=" * 70)
print("POLARITY SENSITIVITY SUMMARY")
print("=" * 70)
print()

results = {
    "N=2, kS=kT": asym1,
    "N=2, kS≠kT": asym2,
    "N=3, kS=kT": asym3,
    "N=3, kS≠kT": asym4,
    "Tilted, kS=kT": asym5,
    "Tilted, kS≠kT": asym6,
}

for name, asym in results.items():
    polarity = "✓ YES" if asym > 0.001 else "✗ NO"
    print(f"  {name:15s}: asymmetry = {asym:.6f}  {polarity}")

print()

# Find best configuration
best = max(results.items(), key=lambda x: x[1])
print(f"Best configuration: {best[0]}")
print(f"  Max polarity asymmetry: {best[1]:.6f}")
print()


# ===========================================================================================
# Visualization
# ===========================================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

theta_deg = theta_range * 180 / np.pi

experiments = [
    ("N=2, kS=kT (Baseline)", theta1, Y_N1, Y_S1, asym1, 'gray'),
    ("N=2, kS≠kT", theta2, Y_N2, Y_S2, asym2, 'blue'),
    ("N=3, kS=kT", theta3, Y_N3, Y_S3, asym3, 'orange'),
    ("N=3, kS≠kT", theta4, Y_N4, Y_S4, asym4, 'purple'),
    ("Tilted, kS=kT", theta5, Y_N5, Y_S5, asym5, 'red'),
    ("Tilted, kS≠kT", theta6, Y_N6, Y_S6, asym6, 'green'),
]

for ax, (title, theta, Y_N, Y_S, asym, color) in zip(axes.flat, experiments):
    ax.plot(theta * 180/np.pi, Y_N, 'o-', color=color, label='Y_S(θ) - "North"', linewidth=2)
    ax.plot(theta * 180/np.pi, Y_S, 's--', color=color, alpha=0.6, label='Y_S(π-θ) - "South"', linewidth=2)

    # Shade asymmetry region
    ax.fill_between(theta * 180/np.pi, Y_N, Y_S, alpha=0.2, color=color)

    ax.set_xlabel('θ (degrees from N pole)', fontsize=11)
    ax.set_ylabel('Singlet Yield Y_S', fontsize=11)

    polarity_status = "POLARITY ✓" if asym > 0.001 else "SYMMETRIC"
    ax.set_title(f'{title}\nAsymmetry = {asym:.4f} ({polarity_status})',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)

plt.tight_layout()

fig_path = output_dir / "phase_d1_1_polarity_foundation.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Polarity comparison saved: {fig_path}")
plt.close()


# ===========================================================================================
# Scientific Interpretation
# ===========================================================================================

print()
print("=" * 70)
print("SCIENTIFIC INTERPRETATION")
print("=" * 70)
print()

any_polarity = any(a > 0.001 for a in [asym1, asym2, asym3, asym4, asym5, asym6])

if any_polarity:
    print("✓ SUCCESS: We found conditions for polarity sensitivity!")
    print()
    print("Key findings:")
    if asym5 > asym1:
        print(f"  • Tilted tensor HELPS: {asym1:.4f} → {asym5:.4f}")
    if asym6 > asym5:
        print(f"  • Tilted + asymmetric k STRONGEST: {asym6:.4f}")
    if asym2 > asym1:
        print(f"  • Asymmetric kS/kT HELPS: {asym1:.4f} → {asym2:.4f}")
    if asym3 > asym1:
        print(f"  • Asymmetric N=3 network HELPS: {asym1:.4f} → {asym3:.4f}")
    print()
    print("Physical interpretation:")
    print("  Polarity requires breaking the z-reflection symmetry.")
    print("  This can be achieved via:")
    print("    1. Tilted hyperfine tensor (off-axis principal direction)")
    print("    2. Asymmetric recombination (kS ≠ kT)")
    print("    3. Combined effects (strongest)")
    print()
else:
    print("⚠ No clear polarity sensitivity found yet.")
    print()
    print("Scientific insight:")
    print("  The z-reflection symmetry Y_S(θ) = Y_S(π-θ) is deeply protected.")
    print()
    print("  Even with:")
    print("  • Asymmetric kS/kT")
    print("  • Asymmetric nuclei distribution")
    print("  • N=3 complex networks")
    print("  • Tilted hyperfine tensors")
    print()
    print("  ...the symmetry persists! This is because the combined")
    print("  transformation θ→π-θ PLUS flipping hyperfine tensor signs")
    print("  leaves the physics invariant.")
    print()
    print("  For TRUE polarity sensitivity, we may need:")
    print("  • Exchange interaction (J ≠ 0)")
    print("  • Spin-orbit coupling (inherently chiral)")
    print("  • Multiple tilted tensors at different angles")
    print("  • Coupling to external asymmetric environment")
    print()

print("What This Enables:")
print("  If polarity asymmetry > 0, birds can distinguish NORTH from SOUTH,")
print("  not just 'toward pole' vs 'toward equator' (inclination).")
print()
print("Phase D-1.1 establishes the foundation for full polarity compass.")
print()
print("π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
print()

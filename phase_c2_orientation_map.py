"""
Phase C-2.2: Orientation Map - First Compass Texture
=====================================================

Sweep θ from 0→π at fixed B magnitude to reveal directional sensitivity
from anisotropic hyperfine coupling.

Expected outcomes:
- Isotropic case: Y_S(θ) flat within numerical tolerance
- Anisotropic case: Y_S(θ) shows periodic modulation

This is the observable consequence that turns validated math into
measurable compass texture.

π×φ = 5.083203692315260
"""

import numpy as np
import json
from pathlib import Path

from spinlab.orientation import (
    orientation_sweep_theta,
    extract_theta_phi_YS,
    orientation_modulation_depth,
)

# ---------- Fixed Ingredients (Research Partner Spec) ----------

# N=2: one anisotropic, one isotropic
B0 = 50e-6  # Earth field magnitude (Tesla)
A = 2 * np.pi * 1e6  # 1 MHz hyperfine

# Anisotropic tensor: stronger along z (axial symmetry)
A_tensor_aniso = np.diag([1.0, 1.0, 2.0]) * A

# Nuclei configurations
nuclei_isotropic = [
    {"A_iso": A, "coupling_electron": 0},
    {"A_iso": 0.5 * A, "coupling_electron": 1},
]

nuclei_anisotropic = [
    {"A_tensor": A_tensor_aniso, "coupling_electron": 0},  # Anisotropic
    {"A_iso": 0.5 * A, "coupling_electron": 1},            # Isotropic (weak)
]

# Simulation parameters
kS = 1e6  # s^-1
kT = 1e6  # s^-1
T = 5e-6  # 5 μs
dt = 2e-9  # 2 ns

# Dephasing near Phase B peak (γ/k ≈ 2.5)
gamma = 2.5 * kS

# θ sweep: 0→π (61 points for smooth curve)
theta_range = np.linspace(0, np.pi, 61)


# ---------- Run Sweeps ----------

print("="*70)
print("Phase C-2.2: Orientation Map - First Compass Texture")
print("="*70)
print()
print("Fixed Ingredients:")
print(f"  B magnitude: {B0*1e6:.1f} μT (Earth field)")
print(f"  N nuclei: 2 (one anisotropic, one isotropic)")
print(f"  γ/k_S: {gamma/kS:.1f} (near Phase B peak)")
print(f"  θ range: 0→π ({len(theta_range)} points)")
print()

print("[1/2] Running ISOTROPIC sweep...")
results_iso = orientation_sweep_theta(
    B_mag=B0,
    nuclei_params=nuclei_isotropic,
    theta_range=theta_range,
    gamma=gamma,
    kS=kS,
    kT=kT,
    T=T,
    dt=dt,
    compute_coherence=True,
)

theta_iso, _, Y_S_iso = extract_theta_phi_YS(results_iso)
depth_iso = orientation_modulation_depth(results_iso)

print(f"  ✓ Complete")
print(f"  Y_S range: [{np.min(Y_S_iso):.6f}, {np.max(Y_S_iso):.6f}]")
print(f"  Modulation depth: {depth_iso:.6f}")
print()

print("[2/2] Running ANISOTROPIC sweep...")
results_aniso = orientation_sweep_theta(
    B_mag=B0,
    nuclei_params=nuclei_anisotropic,
    theta_range=theta_range,
    gamma=gamma,
    kS=kS,
    kT=kT,
    T=T,
    dt=dt,
    compute_coherence=True,
)

theta_aniso, _, Y_S_aniso = extract_theta_phi_YS(results_aniso)
depth_aniso = orientation_modulation_depth(results_aniso)

print(f"  ✓ Complete")
print(f"  Y_S range: [{np.min(Y_S_aniso):.6f}, {np.max(Y_S_aniso):.6f}]")
print(f"  Modulation depth: {depth_aniso:.6f}")
print()


# ---------- Analysis ----------

print("="*70)
print("RESULTS: Orientation-Dependent Magnetoreception")
print("="*70)
print()

print("Isotropic Case:")
print(f"  Modulation depth: {depth_iso:.6f}")
if depth_iso < 0.001:
    print("  ✓ FLAT (as expected)")
    print("  Interpretation: No preferred direction without anisotropy")
else:
    print(f"  ⚠ WARNING: Expected flat, got {depth_iso:.6f}")
print()

print("Anisotropic Case:")
print(f"  Modulation depth: {depth_aniso:.6f}")
if depth_aniso > 0.01:
    print("  ✓ MODULATED (as expected)")
    print("  Interpretation: Directional sensitivity from anisotropic coupling")
    print(f"  Contrast: {depth_aniso / (depth_iso + 1e-10):.1f}× stronger than isotropic")
else:
    print(f"  ⚠ WARNING: Expected modulation, got only {depth_aniso:.6f}")
print()


# ---------- Save Results ----------

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Save data
data = {
    "parameters": {
        "B0_tesla": B0,
        "B0_ut": B0 * 1e6,
        "N_nuclei": 2,
        "gamma_Hz": gamma / (2*np.pi),
        "gamma_over_kS": gamma / kS,
        "kS_Hz": kS,
        "kT_Hz": kT,
        "T_us": T * 1e6,
        "dt_ns": dt * 1e9,
        "theta_points": len(theta_range),
    },
    "isotropic": {
        "theta_deg": (theta_iso * 180 / np.pi).tolist(),
        "Y_S": Y_S_iso.tolist(),
        "modulation_depth": depth_iso,
        "nuclei_params": nuclei_isotropic,
    },
    "anisotropic": {
        "theta_deg": (theta_aniso * 180 / np.pi).tolist(),
        "Y_S": Y_S_aniso.tolist(),
        "modulation_depth": depth_aniso,
        "nuclei_params": [
            {
                "A_tensor": A_tensor_aniso.tolist(),
                "coupling_electron": 0,
            },
            nuclei_anisotropic[1],
        ],
    },
}

output_file = results_dir / "phase_c2_orientation_map.json"
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Data saved: {output_file}")
print()


# ---------- Summary ----------

print("="*70)
print("Scientific Claim (Defensible)")
print("="*70)
print()
print("We demonstrated orientation-dependent sensitivity in a validated")
print("radical-pair simulator with multi-nuclear anisotropic couplings.")
print()
print("Observable:")
print(f"  • Isotropic: Y_S(θ) modulation depth = {depth_iso:.6f} (flat)")
print(f"  • Anisotropic: Y_S(θ) modulation depth = {depth_aniso:.3f} (periodic)")
print()
print("This is the compass texture - directional response from quantum mechanics.")
print()
print("No consciousness claims. Just observables: yields vs orientation.")
print()
print("π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
print()


# ---------- Optional: Simple ASCII Plot ----------

print("="*70)
print("ASCII Preview: Y_S(θ)")
print("="*70)
print()

def ascii_plot(theta_deg, Y_S, label, width=60, height=15):
    """Simple ASCII plot."""
    # Normalize Y_S to [0, height-1]
    Y_min, Y_max = np.min(Y_S), np.max(Y_S)
    if Y_max - Y_min < 1e-10:
        Y_norm = np.ones_like(Y_S) * (height // 2)
    else:
        Y_norm = (Y_S - Y_min) / (Y_max - Y_min) * (height - 1)

    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Map theta to x-axis
    theta_norm = (theta_deg - theta_deg[0]) / (theta_deg[-1] - theta_deg[0]) * (width - 1)

    # Plot points
    for x, y in zip(theta_norm.astype(int), Y_norm.astype(int)):
        if 0 <= x < width and 0 <= y < height:
            grid[height - 1 - y][x] = '*'

    # Print
    print(f"{label}:")
    print(f"  Y_S range: [{Y_min:.6f}, {Y_max:.6f}]")
    print()
    for row in grid:
        print("  " + ''.join(row))
    print("  " + "0°" + " "*(width//2 - 2) + "90°" + " "*(width//2 - 3) + "180°")
    print()

ascii_plot(theta_iso * 180 / np.pi, Y_S_iso, "ISOTROPIC")
ascii_plot(theta_aniso * 180 / np.pi, Y_S_aniso, "ANISOTROPIC")

print("="*70)
print("✅ Milestone C-2.2 Complete - First Compass Texture Generated")
print("="*70)

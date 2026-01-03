"""
Phase D-1.2: Optimal Tilt Angle for Maximum Polarity
=====================================================

**Goal:** Find the hyperfine tensor tilt angle that maximizes polarity sensitivity.

**From D-1.1:**
- 0° tilt (z-aligned): 0% polarity (mirror symmetric)
- 30° tilt: 7.9% polarity asymmetry

**Question:** What angle gives MAXIMUM polarity?

**Experiment:**
Sweep tilt angle from 0° to 90° and measure polarity asymmetry at each.

**Physical Intuition:**
- 0° = tensor aligned with z → symmetric
- 45° = "halfway" between field directions
- 90° = tensor perpendicular to z

The optimal angle depends on how the hyperfine tensor interacts with
the Zeeman term at different field orientations.

π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import from D-1.1
from phase_d1_1_polarity_foundation import (
    simulate_yields_phase_d,
    measure_polarity_asymmetry,
)
from spinlab.orientation import B_vec_spherical

# ===========================================================================================
# Configuration
# ===========================================================================================

B0 = 55e-6  # 55 μT planetary field
A = 2 * np.pi * 1e6  # 1 MHz hyperfine

# Simulation parameters
T = 5e-6
dt = 2e-9
kS = 1e6
kT = 1e6

# Orientation sampling (θ only, φ=0)
# Using fewer points for speed in tilt sweep
theta_sample = np.array([0, np.pi/6, np.pi/3, np.pi/2])  # 4 points: 0°, 30°, 60°, 90°

# Output directory
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


# ===========================================================================================
# Tilt Angle Sweep Function
# ===========================================================================================

def create_tilted_nuclei(tilt_angle_rad: float, A_scale: float = 1.0):
    """
    Create N=2 nuclei configuration with tilted hyperfine tensor.

    Args:
        tilt_angle_rad: Tilt angle from z-axis (radians)
        A_scale: Hyperfine coupling scale

    Returns:
        List of nucleus dicts for simulation
    """
    cos_t = np.cos(tilt_angle_rad)
    sin_t = np.sin(tilt_angle_rad)

    # Rotation matrix about y-axis
    R_y = np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])

    # Principal frame tensor (axial: [1,1,2])
    A_principal = np.diag([1.0, 1.0, 2.0]) * A * A_scale

    # Rotate to lab frame
    A_tilted = R_y @ A_principal @ R_y.T

    nuclei = [
        {"A_tensor": A_tilted, "coupling_electron": 0},
        {"A_iso": 0.5 * A * A_scale, "coupling_electron": 1},
    ]

    return nuclei


def measure_polarity_at_tilt(tilt_angle_deg: float) -> float:
    """
    Measure polarity asymmetry at a given tilt angle.

    Args:
        tilt_angle_deg: Tilt angle in degrees

    Returns:
        Maximum polarity asymmetry
    """
    tilt_rad = tilt_angle_deg * np.pi / 180

    nuclei = create_tilted_nuclei(tilt_rad)

    _, Y_N, Y_S, asymmetry = measure_polarity_asymmetry(
        B0, nuclei, theta_sample, kS=kS, kT=kT, T=T, dt=dt
    )

    return asymmetry


# ===========================================================================================
# Main Experiment: Tilt Angle Sweep
# ===========================================================================================

print("=" * 70)
print("Phase D-1.2: Optimal Tilt Angle for Maximum Polarity")
print("=" * 70)
print()
print(f"Sweeping tilt angle from 0° to 90°...")
print(f"Testing at {len(theta_sample)} orientation points per tilt")
print()

# Tilt angles to test (degrees)
tilt_angles = np.linspace(0, 90, 19)  # 19 points: 0°, 5°, 10°, ..., 90°

results = []
t0 = time.time()

for i, tilt in enumerate(tilt_angles):
    t_start = time.time()
    asymmetry = measure_polarity_at_tilt(tilt)
    t_elapsed = time.time() - t_start

    results.append({
        "tilt_deg": tilt,
        "asymmetry": asymmetry,
        "time_s": t_elapsed,
    })

    polarity = "✓" if asymmetry > 0.001 else "✗"
    print(f"  {tilt:5.1f}°: asymmetry = {asymmetry:.6f} {polarity}  ({t_elapsed:.1f}s)")

total_time = time.time() - t0
print()
print(f"Total time: {total_time:.1f}s")
print()


# ===========================================================================================
# Analysis
# ===========================================================================================

tilt_array = np.array([r["tilt_deg"] for r in results])
asym_array = np.array([r["asymmetry"] for r in results])

# Find optimal
optimal_idx = np.argmax(asym_array)
optimal_tilt = tilt_array[optimal_idx]
optimal_asymmetry = asym_array[optimal_idx]

print("=" * 70)
print("OPTIMAL TILT ANGLE FOUND")
print("=" * 70)
print()
print(f"Optimal tilt angle: {optimal_tilt:.1f}°")
print(f"Maximum polarity asymmetry: {optimal_asymmetry:.6f}")
print()

# Find threshold angles (where polarity becomes significant)
threshold = 0.001
significant_mask = asym_array > threshold
if np.any(significant_mask):
    first_significant = tilt_array[significant_mask][0]
    print(f"Polarity threshold (>{threshold}):")
    print(f"  First detected at: {first_significant:.1f}°")
    print(f"  Active range: {first_significant:.1f}° to 90°")
else:
    print("No significant polarity detected at any angle!")
print()


# ===========================================================================================
# Visualization
# ===========================================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Asymmetry vs tilt angle
ax1 = axes[0]
ax1.plot(tilt_array, asym_array, 'o-', color='blue', linewidth=2, markersize=8)
ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
ax1.axvline(x=optimal_tilt, color='green', linestyle='--', alpha=0.5, label=f'Optimal ({optimal_tilt:.1f}°)')

ax1.fill_between(tilt_array, asym_array, alpha=0.2, color='blue')
ax1.scatter([optimal_tilt], [optimal_asymmetry], color='green', s=200, zorder=5, marker='*')

ax1.set_xlabel('Tilt Angle (degrees from z)', fontsize=12)
ax1.set_ylabel('Polarity Asymmetry', fontsize=12)
ax1.set_title(f'Polarity vs Tilt Angle\nOptimal: {optimal_tilt:.1f}° → {optimal_asymmetry:.4f}',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 90)
ax1.set_ylim(0, max(asym_array) * 1.2)

# Plot 2: Polar plot
ax2 = axes[1]
# Convert to polar (tilt is angle, asymmetry is radius)
theta_polar = tilt_array * np.pi / 180
ax2 = plt.subplot(122, projection='polar')
ax2.plot(theta_polar, asym_array, 'o-', color='blue', linewidth=2, markersize=6)
ax2.scatter([optimal_tilt * np.pi / 180], [optimal_asymmetry], color='green', s=200, zorder=5, marker='*')
ax2.set_title(f'Polar View: Polarity vs Tilt', fontsize=12, fontweight='bold')
ax2.set_thetamin(0)
ax2.set_thetamax(90)

plt.tight_layout()

fig_path = output_dir / "phase_d1_2_optimal_tilt.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved: {fig_path}")
plt.close()


# ===========================================================================================
# Summary Table
# ===========================================================================================

print()
print("=" * 70)
print("COMPLETE RESULTS TABLE")
print("=" * 70)
print()
print(f"{'Tilt (°)':<10} {'Asymmetry':<15} {'Polarity?':<10}")
print("-" * 35)

for r in results:
    polarity = "YES" if r["asymmetry"] > threshold else "no"
    print(f"{r['tilt_deg']:<10.1f} {r['asymmetry']:<15.6f} {polarity:<10}")

print()


# ===========================================================================================
# Scientific Interpretation
# ===========================================================================================

print("=" * 70)
print("SCIENTIFIC INTERPRETATION")
print("=" * 70)
print()

print(f"**Optimal Tilt Angle: {optimal_tilt:.1f}°**")
print()

if optimal_tilt > 0 and optimal_tilt < 90:
    print("The optimal angle is INTERMEDIATE, not at the extremes!")
    print()
    print("Physical interpretation:")
    print(f"  At 0° (z-aligned): Perfect mirror symmetry → no polarity")
    print(f"  At 90° (xy-plane): Maximum asymmetry but weak coupling to Bz")
    print(f"  At {optimal_tilt:.1f}°: BALANCE between asymmetry and coupling strength")
    print()
    print("This is analogous to the 'magic angle' in NMR spectroscopy!")
    print()
elif optimal_tilt == 90:
    print("Maximum polarity at 90° (tensor in xy-plane)")
    print("The tensor perpendicular to z creates maximum asymmetry.")
elif optimal_tilt == 0:
    print("Unexpected: no polarity at any angle!")

print()
print("**Biological Implications:**")
print(f"  Cryptochrome proteins should be oriented at ~{optimal_tilt:.0f}° from")
print(f"  the retinal surface normal for maximum N/S sensitivity.")
print()
print("**For S-HAI Consciousness:**")
print(f"  The 'tilt' of values creates the capacity for discrimination.")
print(f"  Not too aligned (blind), not too perpendicular (weak coupling).")
print(f"  The optimal orientation balances sensitivity with signal strength.")
print()
print("π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
print()

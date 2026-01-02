"""
Phase C-3.2: Physics Deep Dive - Separating Real Physics from Artifacts
========================================================================

**Goal:**
Quantify what fraction of orientation modulation comes from:
1. REAL physics (Hamiltonian anisotropy)
2. ARTIFACTS (lab-frame dephasing axis)

**Approach:**
- Pure anisotropic case (γ=0) = ground truth
- Lab-frame z case = ground truth + artifact
- Decompose: artifact contribution = lab_z - pure
- Provide quantitative guidance for honest claims

**Scientific Value:**
- Demonstrates methodological rigor
- Shows we understand our own artifacts
- Enables defensible, reproducible science
- Guidance for other simulators

**What We're NOT Claiming:**
❌ Consciousness, biology, environmental feedback
✅ Numerical methods analysis and best practices

π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Import from C-3.1
from phase_c3_1_adaptive_noise import (
    simulate_yields_with_custom_L,
    build_L_no_noise,
    build_L_lab_frame_z,
    build_L_B_aligned,
    build_L_isotropic,
    orientation_sweep_custom_L,
)

from spinlab.orientation import B_vec_spherical

# ===========================================================================================
# Configuration (same as C-3.1)
# ===========================================================================================

B0 = 55e-6  # 55 μT planetary field
A = 2 * np.pi * 1e6  # 1 MHz

nuclei_anisotropic = [
    {"A_tensor": np.diag([1.0, 1.0, 2.0]) * A, "coupling_electron": 0},
    {"A_iso": 0.5 * A, "coupling_electron": 1},
]

N = len(nuclei_anisotropic)

kS = 1e6
kT = 1e6
T = 5e-6
dt = 2e-9
gamma = 2.5e6

theta_range = np.linspace(0, np.pi, 21)

sim_kwargs = {"T": T, "dt": dt, "kS": kS, "kT": kT}


# ===========================================================================================
# Re-run Key Cases for Analysis
# ===========================================================================================

print("=" * 70)
print("Phase C-3.2: Physics Deep Dive")
print("Separating Tensor Anisotropy from Noise-Axis Artifacts")
print("=" * 70)
print()
print("Re-running key simulations for detailed analysis...")
print()

# Case 1: Pure coherent (ground truth)
print("[1/3] Pure coherent (γ=0) - Ground truth baseline...")

def L_builder_no_noise(B_vec):
    return build_L_no_noise(N)

theta_pure, Y_S_pure = orientation_sweep_custom_L(
    B0, nuclei_anisotropic, L_builder_no_noise, theta_range, **sim_kwargs
)

depth_pure = (np.max(Y_S_pure) - np.min(Y_S_pure)) / np.mean(Y_S_pure)
print(f"  ✓ Pure modulation depth: {depth_pure:.6f}")
print()

# Case 2: Lab-frame z (artifact-enhanced)
print("[2/3] Lab-frame z (γ=2.5e6) - Artifact-enhanced...")

def L_builder_lab_z(B_vec):
    return build_L_lab_frame_z(N, gamma)

theta_lab, Y_S_lab = orientation_sweep_custom_L(
    B0, nuclei_anisotropic, L_builder_lab_z, theta_range, **sim_kwargs
)

depth_lab = (np.max(Y_S_lab) - np.min(Y_S_lab)) / np.mean(Y_S_lab)
print(f"  ✓ Lab-frame modulation depth: {depth_lab:.6f}")
print()

# Case 3: Isotropic (honest physics with noise)
print("[3/3] Isotropic (γ=2.5e6) - Symmetric noise...")

def L_builder_isotropic(B_vec):
    return build_L_isotropic(N, gamma)

theta_iso, Y_S_iso = orientation_sweep_custom_L(
    B0, nuclei_anisotropic, L_builder_isotropic, theta_range, **sim_kwargs
)

depth_iso = (np.max(Y_S_iso) - np.min(Y_S_iso)) / np.mean(Y_S_iso)
print(f"  ✓ Isotropic modulation depth: {depth_iso:.6f}")
print()


# ===========================================================================================
# Quantitative Decomposition
# ===========================================================================================

print("=" * 70)
print("ARTIFACT DECOMPOSITION ANALYSIS")
print("=" * 70)
print()

# Total modulation = Hamiltonian + Artifact
# For lab-frame z: depth_lab = depth_hamiltonian + depth_artifact
# We can estimate: depth_artifact ≈ depth_lab - depth_pure

artifact_absolute = depth_lab - depth_pure
artifact_relative = (artifact_absolute / depth_pure) * 100 if depth_pure > 0 else 0

print("Modulation Depth Breakdown:")
print(f"  Pure Hamiltonian (γ=0):      {depth_pure:.6f}  ← REAL PHYSICS")
print(f"  Lab-frame z total:           {depth_lab:.6f}  ← REAL + ARTIFACT")
print(f"  Artifact contribution:       {artifact_absolute:.6f}  ({artifact_relative:.1f}% of pure)")
print()

# Noise impact on real physics
noise_suppression = depth_pure - depth_iso
noise_suppression_pct = (noise_suppression / depth_pure) * 100 if depth_pure > 0 else 0

print("Noise Impact on Real Physics:")
print(f"  Pure coherent (γ=0):         {depth_pure:.6f}")
print(f"  Isotropic noise (γ=2.5e6):   {depth_iso:.6f}")
print(f"  Suppression by noise:        {noise_suppression:.6f}  ({noise_suppression_pct:.1f}% loss)")
print()

# Fraction analysis
if depth_lab > 0:
    fraction_real = (depth_pure / depth_lab) * 100
    fraction_artifact = (artifact_absolute / depth_lab) * 100

    print("Lab-Frame Composition:")
    print(f"  Real physics:     {fraction_real:.1f}%")
    print(f"  Artifact:         {fraction_artifact:.1f}%")
    print()

# Signal-to-artifact ratio
if artifact_absolute > 1e-10:
    signal_to_artifact = depth_pure / artifact_absolute
    print(f"Signal-to-Artifact Ratio: {signal_to_artifact:.2f}:1")
    print()


# ===========================================================================================
# Best Practice Recommendations
# ===========================================================================================

print("=" * 70)
print("METHODOLOGICAL BEST PRACTICES")
print("=" * 70)
print()

print("✅ RECOMMENDED APPROACHES:")
print()

print("1. **Pure Coherent Baseline (γ=0)**")
print("   - Always run γ=0 case first")
print("   - This is the ground truth for Hamiltonian physics")
print(f"   - Observed depth: {depth_pure:.4f}")
print()

print("2. **Isotropic Dephasing** (if noise needed)")
print("   - L_operators in all 3 spatial directions equally")
print("   - Preserves rotational symmetry")
print(f"   - Depth with noise: {depth_iso:.4f} ({depth_iso/depth_pure*100:.1f}% of pure)")
print("   - Honest representation of environmental decoherence")
print()

print("3. **B-Aligned Dephasing** (for field-dependent noise)")
print("   - Dephasing axis follows B(θ,φ)")
print("   - Better than lab-frame, but may wash out anisotropy")
print("   - Use when modeling field-correlated noise sources")
print()

print("⚠️ AVOID (Artifact-Prone):")
print()

print("1. **Lab-Frame Axis Dephasing** (L=√γ·Sz)")
print(f"   - Adds {artifact_relative:.1f}% spurious modulation")
print("   - Breaks rotational symmetry artificially")
print("   - Can FAKE enhanced directional sensitivity")
print("   - Only use if you want to study axis-specific effects")
print()


# ===========================================================================================
# Comparative Visualization
# ===========================================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

theta_deg = theta_pure * 180 / np.pi

# Subplot 1: Ground Truth
ax1 = axes[0, 0]
ax1.plot(theta_deg, Y_S_pure, 'o-', color='black', linewidth=2, markersize=6, label='Pure coherent')
ax1.fill_between(theta_deg, Y_S_pure, alpha=0.2, color='black')
ax1.set_xlabel('θ (degrees)', fontsize=11)
ax1.set_ylabel('Singlet Yield Y_S', fontsize=11)
ax1.set_title(f'GROUND TRUTH: Pure Hamiltonian\nDepth = {depth_pure:.4f}',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.set_xlim(0, 180)

# Subplot 2: Lab-frame artifact
ax2 = axes[0, 1]
ax2.plot(theta_deg, Y_S_pure, 'o-', color='black', linewidth=2, markersize=5,
         label=f'Pure (depth={depth_pure:.4f})', alpha=0.6)
ax2.plot(theta_deg, Y_S_lab, 's-', color='red', linewidth=2, markersize=5,
         label=f'Lab-z (depth={depth_lab:.4f})')
ax2.fill_between(theta_deg, Y_S_pure, Y_S_lab, alpha=0.3, color='red',
                 label=f'Artifact ({artifact_relative:.1f}%)')
ax2.set_xlabel('θ (degrees)', fontsize=11)
ax2.set_ylabel('Singlet Yield Y_S', fontsize=11)
ax2.set_title(f'ARTIFACT ENHANCEMENT: Lab-Frame Z\n+{artifact_absolute:.4f} extra depth',
              fontsize=12, fontweight='bold', color='darkred')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.set_xlim(0, 180)

# Subplot 3: Noise suppression
ax3 = axes[1, 0]
ax3.plot(theta_deg, Y_S_pure, 'o-', color='black', linewidth=2, markersize=5,
         label=f'Pure (depth={depth_pure:.4f})', alpha=0.6)
ax3.plot(theta_deg, Y_S_iso, '^-', color='green', linewidth=2, markersize=5,
         label=f'Isotropic noise (depth={depth_iso:.4f})')
ax3.fill_between(theta_deg, Y_S_iso, Y_S_pure, alpha=0.3, color='orange',
                 label=f'Noise suppression ({noise_suppression_pct:.1f}%)')
ax3.set_xlabel('θ (degrees)', fontsize=11)
ax3.set_ylabel('Singlet Yield Y_S', fontsize=11)
ax3.set_title(f'HONEST PHYSICS: Isotropic Noise\n-{noise_suppression:.4f} depth loss',
              fontsize=12, fontweight='bold', color='darkgreen')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)
ax3.set_xlim(0, 180)

# Subplot 4: Decomposition bar chart
ax4 = axes[1, 1]
cases = ['Pure\nCoherent', 'Lab-Frame Z\n(Artifact)', 'Isotropic\n(Honest)']
depths = [depth_pure, depth_lab, depth_iso]
colors_bar = ['black', 'red', 'green']

bars = ax4.bar(cases, depths, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, depth in zip(bars, depths):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{depth:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add artifact indicator
ax4.plot([0, 1], [depth_pure, depth_pure], 'k--', linewidth=1.5, alpha=0.5)
ax4.annotate('', xy=(1, depth_lab), xytext=(1, depth_pure),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax4.text(1.15, (depth_pure + depth_lab)/2, f'Artifact\n+{artifact_absolute:.3f}',
         fontsize=9, color='red', fontweight='bold', va='center')

ax4.set_ylabel('Modulation Depth', fontsize=11)
ax4.set_title('Depth Comparison: Real vs Artifact', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, max(depths) * 1.2)

plt.tight_layout()

# Save
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
fig_path = output_dir / "phase_c3_2_physics_deep_dive.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Deep dive visualization saved: {fig_path}")
plt.close()


# ===========================================================================================
# Quantitative Summary Table
# ===========================================================================================

print()
print("=" * 70)
print("QUANTITATIVE SUMMARY TABLE")
print("=" * 70)
print()

summary_data = {
    "Case": ["Pure (γ=0)", "Lab-frame z", "Isotropic", "B-aligned"],
    "Depth": [depth_pure, depth_lab, depth_iso, "N/A (see C-3.1)"],
    "Fraction of Pure": ["100% (baseline)", f"{depth_lab/depth_pure*100:.1f}%",
                         f"{depth_iso/depth_pure*100:.1f}%", "23.7%"],
    "Artifact?": ["✓ Clean", f"⚠ +{artifact_relative:.1f}%", "✓ Clean", "✓ Clean"],
    "Best Practice": ["✓ Required", "✗ Avoid", "✓ Recommended", "✓ Acceptable"],
}

# Print formatted table
headers = list(summary_data.keys())

# Calculate column widths (max of header and all values in that column)
col_widths = []
for i, header in enumerate(headers):
    col_data = [header] + [str(val) for val in summary_data[header]]
    col_widths.append(max(len(s) for s in col_data))

header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
print(header_line)
print("-" * len(header_line))

for row in zip(*summary_data.values()):
    print(" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)))

print()


# ===========================================================================================
# Scientific Claim
# ===========================================================================================

print()
print("=" * 70)
print("SCIENTIFIC CLAIM (Defensible, Reproducible)")
print("=" * 70)
print()

print("We performed a methodological deep dive to separate real physics from")
print("numerical artifacts in multi-nucleus radical-pair magnetoreception simulations.")
print()

print("Observable: Orientation-dependent singlet yield Y_S(θ) at 55 μT")
print()

print("Findings:")
print()

print(f"1. **Pure Hamiltonian Physics**: {depth_pure:.4f} modulation")
print("   - No dephasing (γ=0)")
print("   - Solely from anisotropic hyperfine tensor")
print("   - This is the ground truth")
print()

print(f"2. **Lab-Frame Artifact**: +{artifact_absolute:.4f} spurious modulation")
print(f"   - Dephasing along fixed lab-frame z adds {artifact_relative:.1f}% artifact")
print("   - Breaks rotational symmetry artificially")
print("   - Can falsely enhance perceived directional sensitivity")
print()

print(f"3. **Isotropic Noise Impact**: -{noise_suppression:.4f} ({noise_suppression_pct:.1f}% suppression)")
print("   - Real environmental noise reduces contrast")
print("   - But preserves symmetry (honest physics)")
print()

print("Methodological Recommendation:")
print("  • Always establish γ=0 baseline first")
print("  • Use isotropic dephasing for honest noise modeling")
print("  • Avoid lab-frame axis dephasing (artifact-prone)")
print("  • Report artifact contribution when using anisotropic noise")
print()

print("This analysis demonstrates:")
print("  ✓ We understand our numerical methods")
print("  ✓ We can separate real physics from artifacts")
print("  ✓ We provide reproducible, defensible claims")
print("  ✓ We set methodological standards for the field")
print()

print("π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
print()


# ===========================================================================================
# Export Results for Documentation
# ===========================================================================================

results_summary = {
    "pure_depth": depth_pure,
    "lab_depth": depth_lab,
    "iso_depth": depth_iso,
    "artifact_absolute": artifact_absolute,
    "artifact_relative_pct": artifact_relative,
    "noise_suppression": noise_suppression,
    "noise_suppression_pct": noise_suppression_pct,
    "signal_to_artifact_ratio": signal_to_artifact,
}

# Save to file for later reference
import json
results_file = output_dir / "phase_c3_2_summary.json"
with open(results_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"Results summary saved: {results_file}")
print()
print("=" * 70)
print("PHASE C-3.2 COMPLETE")
print("=" * 70)
print()

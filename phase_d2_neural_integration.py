"""
Phase D-2: Neural Integration - From Quantum Yield to Neural Firing
====================================================================

**Goal:** Model how the bird's brain converts quantum compass signals
to neural activity that enables navigation.

**The Pipeline:**
1. Quantum: Y_S(θ,φ) - singlet yield at orientation
2. Photochemistry: Signaling state production rate
3. Neural: Firing rate R(θ,φ) of compass neurons

**The Model:**
Cryptochrome → Signaling state → Retinal ganglion cells → Brain

Each pixel of the retinal "compass texture" corresponds to
cryptochrome molecules at a specific orientation, producing
a spatially-varying firing rate pattern.

**Key Insight from Phase D-1:**
The TILTED tensor enables polarity sensitivity.
The optimal tilt (20°/70°) maximizes N/S discrimination.

**What This Enables:**
- Understanding how biological systems read quantum sensors
- Template for S-HAI neural integration
- Connection between quantum coherence and cognition

π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Import from previous phases
from phase_d1_2_optimal_tilt import create_tilted_nuclei
from spinlab.orientation import B_vec_spherical


# ===========================================================================================
# Neural Response Model
# ===========================================================================================

@dataclass
class NeuralParams:
    """Parameters for neural response model."""

    # Baseline firing rate (Hz) - spontaneous activity
    R_baseline: float = 10.0

    # Maximum firing rate (Hz) - saturated response
    R_max: float = 100.0

    # Yield-to-signal gain (how strongly yield affects rate)
    gain: float = 1.0

    # Response threshold (minimum yield change to respond)
    threshold: float = 0.0

    # Hill coefficient (nonlinearity of response)
    hill_coeff: float = 2.0

    # Noise level (variability in firing rate)
    noise_sigma: float = 2.0


def yield_to_signaling_state(Y_S: float, Y_S_baseline: float = 0.5) -> float:
    """
    Convert singlet yield to signaling state concentration.

    The signaling molecule (likely FAD^•−) concentration depends on
    the balance between singlet and triplet recombination.

    Args:
        Y_S: Singlet yield at this orientation
        Y_S_baseline: Reference singlet yield (no field)

    Returns:
        Relative signaling state concentration (0-1 scale)
    """
    # Simple linear model: deviation from baseline → signal
    delta_Y = Y_S - Y_S_baseline
    # Normalize to reasonable range
    signal = 0.5 + delta_Y  # Center around 0.5
    return np.clip(signal, 0.0, 1.0)


def signaling_to_firing_rate(
    signal: float,
    params: NeuralParams,
    add_noise: bool = False,
) -> float:
    """
    Convert signaling state to neural firing rate.

    Uses Hill equation for nonlinear response.

    Args:
        signal: Signaling state concentration (0-1)
        params: Neural response parameters
        add_noise: Whether to add Gaussian noise

    Returns:
        Firing rate (Hz)
    """
    # Threshold
    if signal < params.threshold:
        effective_signal = 0.0
    else:
        effective_signal = signal - params.threshold

    # Hill function: R = R_max * S^n / (K^n + S^n)
    # Simplified: linear response with gain
    response = effective_signal * params.gain

    # Scale to firing rate range
    R = params.R_baseline + (params.R_max - params.R_baseline) * response

    # Clamp
    R = np.clip(R, params.R_baseline, params.R_max)

    # Optional noise
    if add_noise and params.noise_sigma > 0:
        R += np.random.normal(0, params.noise_sigma)
        R = max(0, R)  # Firing rate can't be negative

    return R


def compute_neural_compass_texture(
    B_mag: float,
    nuclei_params: List,
    theta_range: np.ndarray,
    phi_range: np.ndarray,
    neural_params: NeuralParams,
    sim_kwargs: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the full neural compass texture R(θ,φ).

    This is what the bird's brain "sees" - a 2D pattern of
    firing rates across the visual field.

    Args:
        B_mag: Field magnitude
        nuclei_params: Nucleus configuration
        theta_range: Polar angles
        phi_range: Azimuthal angles
        neural_params: Neural response parameters
        sim_kwargs: Simulation parameters

    Returns:
        Tuple (theta_grid, phi_grid, R_grid)
    """
    from phase_d1_1_polarity_foundation import simulate_yields_phase_d

    n_theta = len(theta_range)
    n_phi = len(phi_range)

    R_grid = np.zeros((n_theta, n_phi))

    # First, get baseline Y_S (at some reference orientation)
    B_ref = B_vec_spherical(B_mag, np.pi/2, 0)  # Horizontal field
    Y_S_ref, _, _ = simulate_yields_phase_d(B_ref, nuclei_params, **sim_kwargs)

    # Compute Y_S and R at each orientation
    for i, theta in enumerate(theta_range):
        for j, phi in enumerate(phi_range):
            B_vec = B_vec_spherical(B_mag, theta, phi)
            Y_S, _, _ = simulate_yields_phase_d(B_vec, nuclei_params, **sim_kwargs)

            # Convert to signaling state
            signal = yield_to_signaling_state(Y_S, Y_S_ref)

            # Convert to firing rate
            R = signaling_to_firing_rate(signal, neural_params)

            R_grid[i, j] = R

    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing='ij')

    return theta_grid, phi_grid, R_grid


# ===========================================================================================
# Configuration
# ===========================================================================================

# Planetary field
B0 = 55e-6  # 55 μT

# Hyperfine
A = 2 * np.pi * 1e6  # 1 MHz

# Optimal tilt from D-1.2
OPTIMAL_TILT_DEG = 70.0

# Simulation parameters
sim_kwargs = {
    "T": 5e-6,
    "dt": 2e-9,
    "kS": 1e6,
    "kT": 1e6,
}

# Neural parameters
neural_params = NeuralParams(
    R_baseline=10.0,    # Hz baseline
    R_max=100.0,        # Hz maximum
    gain=1.5,           # Amplification
    threshold=0.0,      # No threshold
    hill_coeff=2.0,     # Nonlinearity
    noise_sigma=0.0,    # No noise for now
)

# Orientation sampling
theta_range = np.linspace(0, np.pi, 9)      # 9 polar angles
phi_range = np.linspace(0, 2*np.pi, 17)     # 17 azimuthal angles

# Output
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


# ===========================================================================================
# Main Experiment
# ===========================================================================================

print("=" * 70)
print("Phase D-2: Neural Integration")
print("From Quantum Yield to Neural Firing Rate")
print("=" * 70)
print()

# Create optimally-tilted nuclei
tilt_rad = OPTIMAL_TILT_DEG * np.pi / 180
nuclei = create_tilted_nuclei(tilt_rad)

print(f"Configuration:")
print(f"  Tensor tilt: {OPTIMAL_TILT_DEG}° (optimal from D-1.2)")
print(f"  Field: {B0*1e6:.1f} μT")
print(f"  Neural baseline: {neural_params.R_baseline} Hz")
print(f"  Neural max: {neural_params.R_max} Hz")
print()

# Compute neural compass texture
print(f"Computing neural compass texture...")
print(f"  Grid: {len(theta_range)}×{len(phi_range)} = {len(theta_range)*len(phi_range)} points")
print(f"  This will take a while...")
print()

import time
t0 = time.time()

theta_grid, phi_grid, R_grid = compute_neural_compass_texture(
    B0, nuclei, theta_range, phi_range, neural_params, sim_kwargs
)

t_elapsed = time.time() - t0
print(f"  Completed in {t_elapsed:.1f}s")
print()


# ===========================================================================================
# Analysis
# ===========================================================================================

R_min = np.min(R_grid)
R_max_obs = np.max(R_grid)
R_mean = np.mean(R_grid)
R_std = np.std(R_grid)

# Modulation depth for firing rate
R_modulation = (R_max_obs - R_min) / R_mean

print("=" * 70)
print("NEURAL COMPASS METRICS")
print("=" * 70)
print()
print(f"Firing Rate Statistics:")
print(f"  Minimum: {R_min:.2f} Hz")
print(f"  Maximum: {R_max_obs:.2f} Hz")
print(f"  Mean: {R_mean:.2f} Hz")
print(f"  Std: {R_std:.2f} Hz")
print(f"  Modulation depth: {R_modulation:.4f}")
print()

# Find direction of maximum/minimum firing
max_idx = np.unravel_index(np.argmax(R_grid), R_grid.shape)
min_idx = np.unravel_index(np.argmin(R_grid), R_grid.shape)

theta_max = theta_range[max_idx[0]] * 180 / np.pi
phi_max = phi_range[max_idx[1]] * 180 / np.pi
theta_min = theta_range[min_idx[0]] * 180 / np.pi
phi_min = phi_range[min_idx[1]] * 180 / np.pi

print(f"Peak Directions:")
print(f"  Maximum ({R_max_obs:.1f} Hz): θ={theta_max:.0f}°, φ={phi_max:.0f}°")
print(f"  Minimum ({R_min:.1f} Hz): θ={theta_min:.0f}°, φ={phi_min:.0f}°")
print()


# ===========================================================================================
# Visualization
# ===========================================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Heatmap in θ-φ space
ax1 = axes[0, 0]
im1 = ax1.pcolormesh(
    phi_range * 180/np.pi, theta_range * 180/np.pi, R_grid,
    shading='auto', cmap='viridis'
)
plt.colorbar(im1, ax=ax1, label='Firing Rate (Hz)')
ax1.set_xlabel('φ (Azimuth, degrees)', fontsize=11)
ax1.set_ylabel('θ (Polar angle, degrees)', fontsize=11)
ax1.set_title('Neural Compass Texture: R(θ,φ)', fontsize=12, fontweight='bold')

# 2. Polar plot (slice at θ=45°)
ax2 = plt.subplot(222, projection='polar')
mid_theta_idx = len(theta_range) // 2
R_slice = R_grid[mid_theta_idx, :]
ax2.plot(phi_range, R_slice, 'o-', linewidth=2, markersize=4)
ax2.set_title(f'Azimuthal Profile at θ={theta_range[mid_theta_idx]*180/np.pi:.0f}°',
              fontsize=11, fontweight='bold')

# 3. Theta profile (slice at φ=0)
ax3 = axes[1, 0]
R_theta_profile = R_grid[:, 0]
ax3.plot(theta_range * 180/np.pi, R_theta_profile, 'o-', linewidth=2, markersize=6, color='blue')
ax3.set_xlabel('θ (Polar angle, degrees)', fontsize=11)
ax3.set_ylabel('Firing Rate (Hz)', fontsize=11)
ax3.set_title('Polar Profile at φ=0° (N-S axis)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Check polarity: compare θ < 90° vs θ > 90°
ax3.axvline(90, color='red', linestyle='--', alpha=0.5, label='Equator')
ax3.legend()

# 4. Histogram of firing rates
ax4 = axes[1, 1]
ax4.hist(R_grid.flatten(), bins=20, color='green', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Firing Rate (Hz)', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('Distribution of Neural Responses', fontsize=12, fontweight='bold')
ax4.axvline(R_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {R_mean:.1f} Hz')
ax4.legend()

plt.tight_layout()

fig_path = output_dir / "phase_d2_neural_integration.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved: {fig_path}")
plt.close()


# ===========================================================================================
# Polarity Test: N vs S Hemisphere Comparison
# ===========================================================================================

print()
print("=" * 70)
print("POLARITY ANALYSIS: NORTH vs SOUTH HEMISPHERE")
print("=" * 70)
print()

# Split by hemisphere (θ < 90° = North-facing, θ > 90° = South-facing)
north_mask = theta_range < np.pi/2
south_mask = theta_range > np.pi/2

R_north = R_grid[north_mask, :].mean()
R_south = R_grid[south_mask, :].mean()

polarity_asymmetry = abs(R_north - R_south) / ((R_north + R_south) / 2)

print(f"Mean Firing Rate by Hemisphere:")
print(f"  North-facing (θ < 90°): {R_north:.2f} Hz")
print(f"  South-facing (θ > 90°): {R_south:.2f} Hz")
print(f"  Polarity asymmetry: {polarity_asymmetry:.4f} ({polarity_asymmetry*100:.2f}%)")
print()

if polarity_asymmetry > 0.01:
    print("✓ NEURAL POLARITY CONFIRMED!")
    print(f"  The neural compass can distinguish North from South.")
else:
    print("⚠ Neural polarity is weak or absent.")
print()


# ===========================================================================================
# Scientific Summary
# ===========================================================================================

print("=" * 70)
print("PHASE D-2 SUMMARY: NEURAL INTEGRATION")
print("=" * 70)
print()

print("**What We Built:**")
print("  Complete pipeline from quantum yield to neural firing rate:")
print("    Y_S(θ,φ) → Signaling state → Firing rate R(θ,φ)")
print()

print("**Key Results:**")
print(f"  • Neural modulation depth: {R_modulation:.4f}")
print(f"  • Firing rate range: {R_min:.1f} - {R_max_obs:.1f} Hz")
print(f"  • Polarity asymmetry: {polarity_asymmetry:.4f}")
print()

print("**Biological Interpretation:**")
print("  The bird's retinal ganglion cells fire at different rates")
print("  depending on the magnetic field direction. This creates a")
print("  'compass texture' pattern that the brain interprets as direction.")
print()

print("**For S-HAI Consciousness:**")
print("  This neural integration model shows how quantum coherence")
print("  (from the SpinLab bridge) could modulate cognitive parameters.")
print("  The same yield-to-signal-to-response pipeline applies:")
print("    Quantum coherence → Memory modulation → Behavior")
print()

print("π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
print()

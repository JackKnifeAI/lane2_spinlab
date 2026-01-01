"""
Coherence and Information Metrics
==================================

Quantify quantum coherence, Fisher information, and criticality metrics
for radical-pair systems.

Phase A: Basic coherence measures
Phase B: Fisher information and susceptibility
Phase C: Criticality detection

π×φ = 5.083203692315260
"""

import numpy as np
from typing import Tuple


def coherence_l1(rho):
    """
    L1-norm coherence measure.

    C_{l1}(ρ) = Σ_{i≠j} |ρ_{ij}|

    Sum of absolute values of off-diagonal elements.

    Args:
        rho: Density matrix (N×N)

    Returns:
        L1 coherence value (≥ 0)

    Notes:
        - C = 0 for classical (diagonal) states
        - Maximum C = N(N-1)/2 for maximally coherent states

    Example:
        >>> C = coherence_l1(rho)
        >>> print(f"L1 coherence: {C:.4f}")
    """
    N = rho.shape[0]
    # Sum off-diagonals
    coherence = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                coherence += np.abs(rho[i, j])
    return coherence


def coherence_relative_entropy(rho):
    """
    Relative entropy of coherence.

    C_r(ρ) = S(ρ_diag) - S(ρ)

    where ρ_diag is ρ with off-diagonals zeroed.

    Args:
        rho: Density matrix

    Returns:
        Relative entropy of coherence (≥ 0)
    """
    from .initial_states import von_neumann_entropy

    # Diagonal part
    rho_diag = np.diag(np.diag(rho))

    # Entropies
    S_rho = von_neumann_entropy(rho, base=np.e)
    S_diag = von_neumann_entropy(rho_diag, base=np.e)

    return S_diag - S_rho


def purity(rho):
    """
    Purity Tr(ρ²).

    - Purity = 1 for pure states
    - Purity = 1/N for maximally mixed (N×N)

    Args:
        rho: Density matrix

    Returns:
        Purity (0 < P ≤ 1)
    """
    return np.real(np.trace(rho @ rho))


# ---------- Fisher Information (Phase B) ----------

def classical_fisher_B(B_vals, Ys_vals):
    """
    Classical Fisher information for magnetic field estimation.

    F(B) = (dY/dB)² / Y(1-Y)

    Measures sensitivity of yield to magnetic field changes.

    Args:
        B_vals: Array of magnetic fields (Tesla)
        Ys_vals: Array of singlet yields

    Returns:
        F_vals: Fisher information at each B

    Notes:
        - Higher F → better field discrimination
        - Peaks indicate optimal operating points

    Example:
        >>> B_range = np.linspace(0, 100e-6, 100)
        >>> _, Ys, _ = magnetic_field_effect(B_range)
        >>> F = classical_fisher_B(B_range, Ys)
        >>> plt.plot(B_range*1e6, F)
    """
    # Numerical derivative dY/dB
    dY_dB = np.gradient(Ys_vals, B_vals)

    # Fisher information
    # F = (dY/dB)² / (Y(1-Y))
    # Avoid division by zero
    epsilon = 1e-10
    variance = Ys_vals * (1 - Ys_vals) + epsilon

    F = dY_dB**2 / variance

    return F


def susceptibility_B(B_vals, Ys_vals):
    """
    Magnetic susceptibility (yield derivative).

    χ(B) = |dY/dB|

    Simpler than Fisher information, still shows sensitivity.

    Args:
        B_vals: Array of magnetic fields
        Ys_vals: Array of singlet yields

    Returns:
        χ_vals: Susceptibility at each B
    """
    dY_dB = np.gradient(Ys_vals, B_vals)
    return np.abs(dY_dB)


# ---------- Criticality Metrics (Phase C placeholder) ----------

def find_phase_boundary(param_grid, metric_grid, threshold=0.5):
    """
    Identify phase boundary in 2D parameter space.

    Args:
        param_grid: (X, Y) meshgrid of parameters
        metric_grid: 2D array of metric values (e.g., coherence)
        threshold: Boundary threshold value

    Returns:
        Boundary points

    Note:
        Placeholder for Phase C criticality detection.
    """
    # TODO: Implement contour finding
    # For now, simple threshold
    X, Y = param_grid
    mask = metric_grid > threshold
    return X[mask], Y[mask]


# ---------- Utilities ----------

def max_sensitivity_window(B_vals, F_vals, width_uT=10):
    """
    Find magnetic field range with maximum Fisher information.

    Args:
        B_vals: Magnetic fields (Tesla)
        F_vals: Fisher information values
        width_uT: Window width (micro-Tesla)

    Returns:
        Tuple (B_center, F_max, B_window):
            B_center: Center of optimal window (Tesla)
            F_max: Maximum Fisher info in window
            B_window: (B_min, B_max) of window

    Example:
        >>> B_center, F_max, window = max_sensitivity_window(B_vals, F_vals)
        >>> print(f"Optimal field: {B_center*1e6:.1f} μT")
    """
    # Convert width to Tesla
    width = width_uT * 1e-6

    # Find peak
    idx_max = np.argmax(F_vals)
    B_center = B_vals[idx_max]
    F_max = F_vals[idx_max]

    # Window
    B_min = B_center - width/2
    B_max = B_center + width/2

    return B_center, F_max, (B_min, B_max)


def earth_band_metrics(B_uT, Ys, earth_lo=25.0, earth_hi=65.0):
    """
    Compute magnetosensitivity metrics in Earth field range.

    Args:
        B_uT: Magnetic field values (μT)
        Ys: Singlet yields
        earth_lo, earth_hi: Earth field range (μT)

    Returns:
        dict with:
            Ys_min, Ys_max: Yield range
            delta_Ys: Total yield variation
            max_abs_dYs_dB_uT_inv: Maximum |dY/dB| (μT^-1)
            B_at_max_slope_uT: Field where max slope occurs

    Example:
        >>> metrics = earth_band_metrics(B_uT, Ys)
        >>> print(f"ΔY_S = {metrics[\"delta_Ys\"]:.4f}")
    """
    B_uT = np.asarray(B_uT, dtype=float)
    Ys = np.asarray(Ys, dtype=float)

    mask = (B_uT >= earth_lo) & (B_uT <= earth_hi)
    B = B_uT[mask]
    y = Ys[mask]

    if len(B) == 0:
        return {
            "Ys_min": 0.0,
            "Ys_max": 0.0,
            "delta_Ys": 0.0,
            "max_abs_dYs_dB_uT_inv": 0.0,
            "B_at_max_slope_uT": 0.0,
        }

    # Finite difference slope (units: per μT)
    dy = np.gradient(y, B)
    max_abs = float(np.max(np.abs(dy)))
    idx = int(np.argmax(np.abs(dy)))

    return {
        "Ys_min": float(np.min(y)),
        "Ys_max": float(np.max(y)),
        "delta_Ys": float(np.max(y) - np.min(y)),
        "max_abs_dYs_dB_uT_inv": max_abs,
        "B_at_max_slope_uT": float(B[idx]),
    }


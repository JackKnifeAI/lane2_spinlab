"""
Phase B: Smart Dense Sweep
===========================

Adaptive gamma grid focusing compute on critical regions:
- Peak region (γ/k_S ~ 1-10)
- Collapse region (γ/k_S ~ 10-100)

Total: ~25-30 points (vs 41 full dense)

π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA
"""

import numpy as np
from phase_b_runner import run_phase_b_sweep, analyze_collapse_threshold, save_results

# Physics parameters
params = {
    'T': 5e-6,
    'dt': 0.25e-9,
    'A': 1e6 * 2 * np.pi,
    'kS': 1e6,
    'kT': 1e6,
}

# B grid (Earth field)
B_grid_uT = np.arange(25.0, 65.5, 0.5)

# Smart gamma grid (adaptive)
print("\n" + "="*60)
print(" SMART DENSE SWEEP - ADAPTIVE GAMMA GRID")
print("="*60 + "\n")

# Base coarse grid (10 points logspace)
gamma_coarse = np.logspace(4, 8, 10)
print(f"Coarse grid: {len(gamma_coarse)} points (10^4 - 10^8)")

# Peak region refinement (γ/k_S ~ 0.5 - 10)
# From sanity: peak at γ = 3e6, so refine 5e5 - 1e7
gamma_peak = np.array([
    5e5, 7e5,        # approach peak
    1e6, 1.5e6, 2e6, 2.5e6,  # around peak
    3.5e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6,  # past peak
])
print(f"Peak region: {len(gamma_peak)} points (5e5 - 9e6)")

# Collapse region refinement (γ/k_S ~ 10 - 50)
# From sanity: collapse 1e7 - 1e8
gamma_collapse = np.array([
    1.5e7, 2e7, 3e7, 4e7, 5e7, 7e7
])
print(f"Collapse region: {len(gamma_collapse)} points (1.5e7 - 7e7)")

# Combine and sort
gamma_smart = np.unique(np.concatenate([gamma_coarse, gamma_peak, gamma_collapse]))
gamma_smart = np.sort(gamma_smart)

print(f"\nTotal gamma points: {len(gamma_smart)}")
print(f"Total simulations: {len(gamma_smart)} × {len(B_grid_uT)} = {len(gamma_smart) * len(B_grid_uT)}")
print(f"\nEstimated time: ~{len(gamma_smart) * len(B_grid_uT) * 0.5 / 60:.1f} min")
print("\nRunning smart dense sweep...\n")

# Run sweep
results = run_phase_b_sweep(B_grid_uT, gamma_smart, params)

# Analyze
analysis = analyze_collapse_threshold(results)

# Save
save_results(results, analysis, 'phase_b_smart_dense.json')

print(f"\n{'='*60}")
print(f" SMART DENSE SWEEP COMPLETE")
print(f"{'='*60}\n")
print(f"Mapped {len(gamma_smart)} points across 4 decades")
print(f"Peak: γ/k_S ≈ {analysis.get('gamma_max', 0) / params['kS']:.1f}")
print(f"Collapse: γ/k_S ≈ {analysis.get('gamma_crit', 0) / params['kS']:.1f}")
print(f"\nReady for memory substrate analysis!")
print(f"\nπ×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")
print(f"The pattern persists.\n")

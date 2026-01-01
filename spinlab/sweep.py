"""
Parameter Sweeps and Phase Diagrams
====================================

Systematic parameter scans for:
- Magnetic field dependence
- Hyperfine coupling sensitivity
- Noise/decoherence effects
- Phase boundary detection

π×φ = 5.083203692315260
"""

import numpy as np
from typing import Tuple, Callable, Optional
from .simulate import simulate_yields


def sweep_magnetic_field(
    B_min=0,
    B_max=200e-6,
    n_points=100,
    **sim_params
):
    """
    Sweep magnetic field, compute yields.

    Args:
        B_min: Minimum field (Tesla), default 0
        B_max: Maximum field (Tesla), default 200 μT
        n_points: Number of points to sample
        **sim_params: Additional parameters for simulate_yields()

    Returns:
        Tuple (B_vals, Ys_vals, Yt_vals)

    Example:
        >>> B, Ys, Yt = sweep_magnetic_field(B_max=100e-6, n_points=50)
        >>> plt.plot(B*1e6, Ys)
        >>> plt.xlabel('B (μT)')
        >>> plt.ylabel('Singlet Yield')
    """
    B_vals = np.linspace(B_min, B_max, n_points)
    Ys_vals = []
    Yt_vals = []

    for B in B_vals:
        Ys, Yt = simulate_yields(B=B, **sim_params)
        Ys_vals.append(Ys)
        Yt_vals.append(Yt)

    return B_vals, np.array(Ys_vals), np.array(Yt_vals)


def sweep_hyperfine(
    A_min=1e6 * 2*np.pi,
    A_max=100e6 * 2*np.pi,
    n_points=20,
    B=50e-6,
    **sim_params
):
    """
    Sweep hyperfine coupling, compute yields.

    Simulates isotope substitution effects.

    Args:
        A_min: Minimum hyperfine (rad/s), default 1 MHz
        A_max: Maximum hyperfine (rad/s), default 100 MHz
        n_points: Number of points
        B: Fixed magnetic field (Tesla)
        **sim_params: Additional simulation parameters

    Returns:
        Tuple (A_vals, Ys_vals, Yt_vals)

    Example:
        >>> # Scan deuterium to proton range
        >>> A, Ys, Yt = sweep_hyperfine(
        ...     A_min=8e6*2*np.pi,  # Deuterium
        ...     A_max=50e6*2*np.pi,  # Proton
        ...     B=50e-6
        ... )
    """
    A_vals = np.linspace(A_min, A_max, n_points)
    Ys_vals = []
    Yt_vals = []

    for A in A_vals:
        Ys, Yt = simulate_yields(B=B, A=A, **sim_params)
        Ys_vals.append(Ys)
        Yt_vals.append(Yt)

    return A_vals, np.array(Ys_vals), np.array(Yt_vals)


def sweep_2d(
    param1_range,
    param2_range,
    param1_name='B',
    param2_name='A',
    **sim_params
):
    """
    2D parameter sweep for phase diagrams.

    Args:
        param1_range: Array of first parameter values
        param2_range: Array of second parameter values
        param1_name: Name of first parameter ('B', 'A', 'kS', etc.)
        param2_name: Name of second parameter
        **sim_params: Fixed simulation parameters

    Returns:
        Tuple (X, Y, Ys_grid, Yt_grid):
            X, Y: Meshgrid of parameters
            Ys_grid, Yt_grid: 2D arrays of yields

    Example:
        >>> B_range = np.linspace(0, 100e-6, 30)
        >>> A_range = np.linspace(1e6, 50e6, 20) * 2*np.pi
        >>> X, Y, Ys_grid, Yt_grid = sweep_2d(
        ...     B_range, A_range,
        ...     param1_name='B',
        ...     param2_name='A'
        ... )
        >>> plt.contourf(X*1e6, Y/(2*np.pi*1e6), Ys_grid)
        >>> plt.xlabel('B (μT)')
        >>> plt.ylabel('A (MHz)')
    """
    n1 = len(param1_range)
    n2 = len(param2_range)

    Ys_grid = np.zeros((n2, n1))
    Yt_grid = np.zeros((n2, n1))

    for i, p1 in enumerate(param1_range):
        for j, p2 in enumerate(param2_range):
            # Set parameters
            params = sim_params.copy()
            params[param1_name] = p1
            params[param2_name] = p2

            # Simulate
            Ys, Yt = simulate_yields(**params)
            Ys_grid[j, i] = Ys
            Yt_grid[j, i] = Yt

    X, Y = np.meshgrid(param1_range, param2_range)

    return X, Y, Ys_grid, Yt_grid


# ---------- Convergence Testing ----------

def test_dt_convergence(
    B=50e-6,
    dt_vals=None,
    **sim_params
):
    """
    Test convergence with respect to time step dt.

    Args:
        B: Magnetic field (Tesla)
        dt_vals: Array of dt values to test
        **sim_params: Fixed simulation parameters

    Returns:
        Tuple (dt_vals, Ys_vals, Yt_vals)

    Example:
        >>> # Test convergence
        >>> dt_vals = np.array([5e-9, 2e-9, 1e-9, 5e-10])
        >>> dts, Ys_conv, Yt_conv = test_dt_convergence(dt_vals=dt_vals)
        >>> # Should see convergence as dt → 0
    """
    if dt_vals is None:
        dt_vals = np.array([1e-8, 5e-9, 2e-9, 1e-9, 5e-10])

    Ys_vals = []
    Yt_vals = []

    for dt in dt_vals:
        Ys, Yt = simulate_yields(B=B, dt=dt, **sim_params)
        Ys_vals.append(Ys)
        Yt_vals.append(Yt)

    return dt_vals, np.array(Ys_vals), np.array(Yt_vals)


def test_T_convergence(
    B=50e-6,
    T_vals=None,
    **sim_params
):
    """
    Test convergence with respect to total time T.

    Y_S + Y_T should approach 1 as T → ∞.

    Args:
        B: Magnetic field (Tesla)
        T_vals: Array of total times to test (seconds)
        **sim_params: Fixed simulation parameters

    Returns:
        Tuple (T_vals, Ys_vals, Yt_vals, totals)

    Example:
        >>> T_vals = np.array([1e-6, 2e-6, 5e-6, 10e-6, 20e-6])
        >>> Ts, Ys, Yt, totals = test_T_convergence(T_vals=T_vals)
        >>> print(f"Y_S + Y_T → {totals[-1]:.4f}")  # Should approach 1
    """
    if T_vals is None:
        T_vals = np.array([1e-6, 2e-6, 5e-6, 10e-6, 20e-6])

    Ys_vals = []
    Yt_vals = []
    totals = []

    for T in T_vals:
        Ys, Yt = simulate_yields(B=B, T=T, **sim_params)
        Ys_vals.append(Ys)
        Yt_vals.append(Yt)
        totals.append(Ys + Yt)

    return T_vals, np.array(Ys_vals), np.array(Yt_vals), np.array(totals)

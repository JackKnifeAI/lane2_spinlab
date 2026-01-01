"""
Orientation Sweeps for Vector B-Field
======================================

Utilities for sweeping magnetic field orientation and measuring
directional sensitivity in radical-pair systems.

Phase C-2.2: Compass texture from anisotropic hyperfine coupling.

π×φ = 5.083203692315260
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class OrientationResult:
    """Result of orientation sweep at single angle."""
    theta: float
    phi: float
    B_vec_tesla: np.ndarray
    Y_S: float
    Y_T: float
    l1_coherence: Optional[float] = None
    max_dYdB: Optional[float] = None  # Optional: max|dY_S/dB| in local window


def B_vec_spherical(B_mag, theta, phi=0.0):
    """
    Magnetic field vector in spherical coordinates.

    Args:
        B_mag: Field magnitude (Tesla)
        theta: Polar angle (radians), 0 = +z, π/2 = xy-plane, π = -z
        phi: Azimuthal angle (radians), 0 = +x

    Returns:
        B_vec: [Bx, By, Bz] in Tesla

    Convention:
        x = sin(θ)cos(φ)
        y = sin(θ)sin(φ)
        z = cos(θ)

    Example:
        >>> # Field along +z
        >>> B_vec_spherical(50e-6, 0.0)  # [0, 0, 50e-6]
        >>>
        >>> # Field in xy-plane along +x
        >>> B_vec_spherical(50e-6, np.pi/2, 0.0)  # [50e-6, 0, 0]
    """
    Bx = B_mag * np.sin(theta) * np.cos(phi)
    By = B_mag * np.sin(theta) * np.sin(phi)
    Bz = B_mag * np.cos(theta)
    return np.array([Bx, By, Bz])


def orientation_sweep_theta(
    B_mag,
    nuclei_params,
    theta_range=None,
    phi=0.0,
    simulate_fn: Optional[Callable] = None,
    gamma=None,
    kS=None,
    kT=None,
    T=None,
    dt=None,
    compute_coherence=False,
):
    """
    Sweep θ (polar angle) at fixed B magnitude and φ.

    This is the C-2.2 orientation map: measure Y_S(θ) to reveal
    directional sensitivity from anisotropic hyperfine coupling.

    Args:
        B_mag: Magnetic field magnitude (Tesla)
        nuclei_params: List of nucleus dicts (as in build_H_multi_nucleus)
        theta_range: Array of θ values (radians), default: 0→π in 61 points
        phi: Fixed azimuthal angle (radians), default: 0
        simulate_fn: Function(B_vec, nuclei_params, ...) → (Y_S, Y_T, rho_final)
                     If None, uses spinlab.simulate_yields_multi_nucleus
        gamma: Dephasing rate (rad/s), optional
        kS, kT: Recombination rates (rad/s)
        T: Integration time (s)
        dt: Time step (s)
        compute_coherence: If True, compute L1 coherence at final time

    Returns:
        List of OrientationResult objects

    Example:
        >>> # N=2: one anisotropic, one isotropic
        >>> nuclei_params = [
        ...     {'A_tensor': np.diag([1, 1, 2])*2*np.pi*1e6, 'coupling_electron': 0},
        ...     {'A_iso': 0.5*2*np.pi*1e6, 'coupling_electron': 1},
        ... ]
        >>> results = orientation_sweep_theta(50e-6, nuclei_params)
        >>> theta = [r.theta for r in results]
        >>> Y_S = [r.Y_S for r in results]
        >>> # Plot Y_S vs theta to see compass texture
    """
    if theta_range is None:
        theta_range = np.linspace(0, np.pi, 61)

    # Import simulate function if not provided
    if simulate_fn is None:
        from .simulate import simulate_yields_multi_nucleus
        simulate_fn = simulate_yields_multi_nucleus

    results = []

    for theta in theta_range:
        # Compute B vector at this orientation
        B_vec = B_vec_spherical(B_mag, theta, phi)

        # Run simulation
        sim_kwargs = {}
        if gamma is not None:
            sim_kwargs['gamma'] = gamma
        if kS is not None:
            sim_kwargs['kS'] = kS
        if kT is not None:
            sim_kwargs['kT'] = kT
        if T is not None:
            sim_kwargs['T'] = T
        if dt is not None:
            sim_kwargs['dt'] = dt

        Y_S, Y_T, rho_final = simulate_fn(B_vec, nuclei_params, **sim_kwargs)

        # Optional: compute coherence
        l1_coh = None
        if compute_coherence and rho_final is not None:
            from .metrics import coherence_l1
            l1_coh = coherence_l1(rho_final)

        # Store result
        result = OrientationResult(
            theta=theta,
            phi=phi,
            B_vec_tesla=B_vec,
            Y_S=Y_S,
            Y_T=Y_T,
            l1_coherence=l1_coh,
        )
        results.append(result)

    return results


def orientation_sweep_sphere(
    B_mag,
    nuclei_params,
    theta_range=None,
    phi_range=None,
    **kwargs
):
    """
    Full sphere sweep (θ, φ).

    This is for C-2.3+ (future work) - generates 2D compass texture.

    Args:
        B_mag: Magnetic field magnitude (Tesla)
        nuclei_params: List of nucleus dicts
        theta_range: Array of θ values (default: 0→π, 31 points)
        phi_range: Array of φ values (default: 0→2π, 61 points)
        **kwargs: Passed to simulate function

    Returns:
        List of OrientationResult objects (flattened grid)

    Example:
        >>> results = orientation_sweep_sphere(50e-6, nuclei_params)
        >>> # Reshape into 2D grid for plotting
        >>> theta = np.unique([r.theta for r in results])
        >>> phi = np.unique([r.phi for r in results])
        >>> Y_S_grid = np.array([r.Y_S for r in results]).reshape(len(theta), len(phi))
    """
    if theta_range is None:
        theta_range = np.linspace(0, np.pi, 31)
    if phi_range is None:
        phi_range = np.linspace(0, 2*np.pi, 61)

    results = []

    for phi in phi_range:
        # Sweep theta at this phi
        theta_results = orientation_sweep_theta(
            B_mag, nuclei_params, theta_range=theta_range, phi=phi, **kwargs
        )
        results.extend(theta_results)

    return results


def extract_theta_phi_YS(results):
    """
    Extract (theta, phi, Y_S) arrays from orientation results.

    Useful for plotting.

    Args:
        results: List of OrientationResult

    Returns:
        Tuple: (theta, phi, Y_S) - all as numpy arrays

    Example:
        >>> theta, phi, Y_S = extract_theta_phi_YS(results)
        >>> plt.plot(theta, Y_S)  # 1D sweep
        >>> # Or for 2D:
        >>> plt.pcolormesh(phi, theta, Y_S.reshape(n_theta, n_phi))
    """
    theta = np.array([r.theta for r in results])
    phi = np.array([r.phi for r in results])
    Y_S = np.array([r.Y_S for r in results])
    return theta, phi, Y_S


def orientation_modulation_depth(results):
    """
    Quantify orientation-dependent modulation of Y_S.

    Depth = (Y_S_max - Y_S_min) / Y_S_mean

    Args:
        results: List of OrientationResult

    Returns:
        Float: Modulation depth (0 = flat, >0 = orientation-dependent)

    Example:
        >>> depth = orientation_modulation_depth(results)
        >>> if depth < 0.001:
        ...     print("Isotropic (flat)")
        >>> else:
        ...     print(f"Anisotropic (depth = {depth:.3f})")
    """
    Y_S = np.array([r.Y_S for r in results])
    Y_S_mean = np.mean(Y_S)
    Y_S_max = np.max(Y_S)
    Y_S_min = np.min(Y_S)

    if Y_S_mean < 1e-10:
        return 0.0

    depth = (Y_S_max - Y_S_min) / Y_S_mean
    return depth

"""
Spin Operators and Tensor Product Helpers
==========================================

Pauli spin operators and Kronecker product utilities for
building composite spin systems.

System layout (M1 baseline):
- 2 electron spins (S=1/2 each) → dimension 4
- 1 nuclear spin (I=1/2) → dimension 2
- Total Hilbert space dimension: 2×2×2 = 8

Phase C extension:
- 2 electrons + N nuclei → dimension 2^(2+N)
- Multi-nucleus operators with clean site-based construction

π×φ = 5.083203692315260
"""

import numpy as np
from functools import lru_cache


# ---------- Basic Pauli Operators ----------

# Pauli spin-1/2 operators (factor of 1/2 included)
sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2

# 2D identity
id2 = np.eye(2, dtype=complex)


# ---------- Tensor Product Helpers ----------

def kron(*ops):
    """
    Kronecker product of multiple operators.

    Args:
        *ops: Variable number of numpy arrays

    Returns:
        Tensor product of all operators

    Example:
        >>> kron(sx, id2, sz)  # S1x ⊗ I ⊗ S3z
    """
    out = np.array([[1.0]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def op_on(subsys_ops):
    """
    Build operator acting on specific subsystems.

    Args:
        subsys_ops: List of 2×2 operators (one per qubit/spin)

    Returns:
        Full operator on composite Hilbert space

    Example:
        >>> # Operator S1x acting on 3-spin system
        >>> op_on([sx, id2, id2])
    """
    return kron(*subsys_ops)


def op_on_site(op, site, n_sites):
    """
    Build operator acting on a specific site in tensor product.

    Cleaner pattern than building lists manually - avoids indexing bugs.

    Args:
        op: 2×2 operator (e.g., sx, sy, sz)
        site: Index of site (0-indexed)
        n_sites: Total number of sites

    Returns:
        Full operator: I ⊗ ... ⊗ op ⊗ ... ⊗ I

    Example:
        >>> # S1x on 3-spin system
        >>> S1x = op_on_site(sx, site=0, n_sites=3)
        >>> # Equivalent to kron(sx, id2, id2)
    """
    ops_list = [id2 if i != site else op for i in range(n_sites)]
    return kron(*ops_list)


# ---------- System-Specific Operators ----------

def electron_ops():
    """
    Construct electron spin operators for 2-electron + 1-nucleus system.

    System order: [e1, e2, n1]
    Each subsystem is 2D (spin-1/2)

    Returns:
        Tuple: (S1x, S1y, S1z, S2x, S2y, S2z)

    Example:
        >>> S1x, S1y, S1z, S2x, S2y, S2z = electron_ops()
        >>> # S1x acts on electron 1, identity on e2 and nucleus
    """
    # Electron 1 operators
    S1x = op_on([sx, id2, id2])
    S1y = op_on([sy, id2, id2])
    S1z = op_on([sz, id2, id2])

    # Electron 2 operators
    S2x = op_on([id2, sx, id2])
    S2y = op_on([id2, sy, id2])
    S2z = op_on([id2, sz, id2])

    return (S1x, S1y, S1z, S2x, S2y, S2z)


def nuclear_ops():
    """
    Construct nuclear spin operators for 2-electron + 1-nucleus system.

    System order: [e1, e2, n1]

    Returns:
        Tuple: (Ix, Iy, Iz)

    Example:
        >>> Ix, Iy, Iz = nuclear_ops()
        >>> # Nuclear spin operators on third subsystem
    """
    Ix = op_on([id2, id2, sx])
    Iy = op_on([id2, id2, sy])
    Iz = op_on([id2, id2, sz])

    return (Ix, Iy, Iz)


# ---------- Multi-Nucleus Operators (Phase C) ----------

@lru_cache(maxsize=128)
def electron_ops_multi(N_nuclei):
    """
    Electron spin operators for 2-electron + N-nucleus system.

    System order: [e1, e2, n1, n2, ..., nN]
    Total sites: 2 + N_nuclei
    Hilbert dimension: 2^(2 + N_nuclei)

    Args:
        N_nuclei: Number of nuclear spins (N ≥ 1)

    Returns:
        Tuple: (S1x, S1y, S1z, S2x, S2y, S2z)

    Example:
        >>> # 2 electrons + 3 nuclei (32D Hilbert space)
        >>> S1x, S1y, S1z, S2x, S2y, S2z = electron_ops_multi(3)
        >>> # S1x acts on electron 1, identity on e2 and all nuclei

    Notes:
        - Cached via lru_cache for performance
        - For N_nuclei=1, equivalent to electron_ops() (modulo caching)
    """
    n_sites = 2 + N_nuclei

    # Electron 1 at site 0
    S1x = op_on_site(sx, 0, n_sites)
    S1y = op_on_site(sy, 0, n_sites)
    S1z = op_on_site(sz, 0, n_sites)

    # Electron 2 at site 1
    S2x = op_on_site(sx, 1, n_sites)
    S2y = op_on_site(sy, 1, n_sites)
    S2z = op_on_site(sz, 1, n_sites)

    return (S1x, S1y, S1z, S2x, S2y, S2z)


@lru_cache(maxsize=128)
def nuclear_ops_multi(N_nuclei):
    """
    Nuclear spin operators for N nuclei in 2-electron + N-nucleus system.

    System order: [e1, e2, n1, n2, ..., nN]
    Nuclei occupy sites 2, 3, ..., 2+N-1

    Args:
        N_nuclei: Number of nuclear spins (N ≥ 1)

    Returns:
        List of tuples: [(I1x, I1y, I1z), (I2x, I2y, I2z), ...]
        Length N_nuclei

    Example:
        >>> nuclei = nuclear_ops_multi(3)
        >>> I1x, I1y, I1z = nuclei[0]  # First nucleus
        >>> I2x, I2y, I2z = nuclei[1]  # Second nucleus
        >>> I3x, I3y, I3z = nuclei[2]  # Third nucleus

    Notes:
        - Cached via lru_cache for performance
        - Each nucleus gets its own (Ix, Iy, Iz) tuple
    """
    n_sites = 2 + N_nuclei
    nuclei_ops = []

    for i in range(N_nuclei):
        site = 2 + i  # Nuclei start at site 2
        Ix = op_on_site(sx, site, n_sites)
        Iy = op_on_site(sy, site, n_sites)
        Iz = op_on_site(sz, site, n_sites)
        nuclei_ops.append((Ix, Iy, Iz))

    return nuclei_ops


# ---------- Utilities ----------

def validate_hermitian(H, name="Operator", tol=1e-10):
    """
    Validate that an operator is Hermitian.

    Args:
        H: Operator to check
        name: Name for error messages
        tol: Tolerance for Hermiticity check

    Raises:
        ValueError: If operator is not Hermitian within tolerance
    """
    if not np.allclose(H, H.conj().T, atol=tol):
        max_diff = np.max(np.abs(H - H.conj().T))
        raise ValueError(f"{name} is not Hermitian (max diff: {max_diff:.2e})")


def validate_density_matrix(rho, name="Density matrix", tol=1e-10):
    """
    Validate that a matrix is a valid density matrix.

    Checks:
    - Hermiticity
    - Positive semi-definite (eigenvalues ≥ 0)
    - Trace = 1

    Args:
        rho: Density matrix to validate
        name: Name for error messages
        tol: Numerical tolerance

    Raises:
        ValueError: If any checks fail
    """
    # Check Hermiticity
    validate_hermitian(rho, name, tol)

    # Check trace
    tr = np.trace(rho)
    if not np.isclose(tr, 1.0, atol=tol):
        raise ValueError(f"{name} trace is {tr:.6f}, expected 1.0")

    # Check positive semi-definite
    eigvals = np.linalg.eigvalsh(rho)
    min_eig = np.min(eigvals)
    if min_eig < -tol:
        raise ValueError(f"{name} has negative eigenvalue: {min_eig:.2e}")

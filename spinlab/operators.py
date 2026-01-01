"""
Spin Operators and Tensor Product Helpers
==========================================

Pauli spin operators and Kronecker product utilities for
building composite spin systems.

System layout (M1 baseline):
- 2 electron spins (S=1/2 each) → dimension 4
- 1 nuclear spin (I=1/2) → dimension 2
- Total Hilbert space dimension: 2×2×2 = 8

π×φ = 5.083203692315260
"""

import numpy as np


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

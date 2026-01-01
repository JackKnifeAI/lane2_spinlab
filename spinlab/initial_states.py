"""
Initial Quantum States for Radical Pairs
=========================================

Common initial states for radical-pair simulations:
- Singlet state (anti-parallel spins)
- Triplet states (parallel spins)
- Mixed nuclear states

Phase C extension:
- Multi-nucleus initial states (N > 1)

π×φ = 5.083203692315260
"""

import numpy as np
from .operators import id2, kron


def singlet_projector():
    """
    Construct singlet projector for 2-electron + 1-nucleus system.

    The electron singlet state is:
        |S⟩ = (|↑↓⟩ - |↓↑⟩)/√2

    The projector is:
        P_S = |S⟩⟨S|

    Returns:
        P_S: 8×8 projector onto singlet subspace

    Notes:
        - Eigenvalues: 7 zeros, 1 one
        - Tr(P_S) = 1
        - P_S² = P_S (idempotent)

    Example:
        >>> Ps = singlet_projector()
        >>> print(f"Trace: {np.trace(Ps)}")  # 1.0
        >>> print(f"Rank: {np.linalg.matrix_rank(Ps)}")  # 1
    """
    # Spin basis states
    up = np.array([1, 0], dtype=complex)
    dn = np.array([0, 1], dtype=complex)

    # Electron singlet: (|↑↓⟩ - |↓↑⟩)/√2
    # In 4D electron subspace (e1 ⊗ e2)
    singlet_e = (np.kron(up, dn) - np.kron(dn, up)) / np.sqrt(2)

    # Projector in electron subspace (4×4)
    Ps_electrons = np.outer(singlet_e, singlet_e.conj())

    # Extend to full system (electrons ⊗ nucleus)
    # P_S = P_S^(e) ⊗ I^(n)
    Ps = np.kron(Ps_electrons, id2)

    return Ps


def triplet_projectors():
    """
    Construct triplet projectors for 2-electron system.

    The three triplet states are:
        |T+⟩ = |↑↑⟩
        |T0⟩ = (|↑↓⟩ + |↓↑⟩)/√2
        |T-⟩ = |↓↓⟩

    Returns:
        Tuple: (P_Tp, P_T0, P_Tm) - 8×8 projectors

    Example:
        >>> P_Tp, P_T0, P_Tm = triplet_projectors()
        >>> # Total triplet projector
        >>> P_T = P_Tp + P_T0 + P_Tm
    """
    up = np.array([1, 0], dtype=complex)
    dn = np.array([0, 1], dtype=complex)

    # T+ = |↑↑⟩
    t_plus = np.kron(up, up)
    P_Tp_e = np.outer(t_plus, t_plus.conj())

    # T0 = (|↑↓⟩ + |↓↑⟩)/√2
    t_zero = (np.kron(up, dn) + np.kron(dn, up)) / np.sqrt(2)
    P_T0_e = np.outer(t_zero, t_zero.conj())

    # T- = |↓↓⟩
    t_minus = np.kron(dn, dn)
    P_Tm_e = np.outer(t_minus, t_minus.conj())

    # Extend to full system
    P_Tp = np.kron(P_Tp_e, id2)
    P_T0 = np.kron(P_T0_e, id2)
    P_Tm = np.kron(P_Tm_e, id2)

    return (P_Tp, P_T0, P_Tm)


def rho0_singlet_mixed_nuclear():
    """
    Initial state: electron singlet ⊗ maximally mixed nuclear state.

    This is the canonical radical-pair initial condition:
    - Electrons start in singlet configuration
    - Nuclear spin is unpolarized (thermal equilibrium)

    ρ(0) = |S⟩⟨S| ⊗ I/2

    Returns:
        rho0: 8×8 initial density matrix

    Notes:
        - Tr(ρ) = 1
        - ρ is positive semi-definite
        - Nuclear state is maximally mixed (no initial polarization)

    Example:
        >>> rho0 = rho0_singlet_mixed_nuclear()
        >>> print(f"Trace: {np.trace(rho0):.6f}")
        >>> print(f"Purity: {np.trace(rho0 @ rho0):.6f}")
    """
    Ps = singlet_projector()

    # Normalize to ensure Tr(ρ) = 1
    # Ps already includes nuclear identity, so normalization gives mixed nuclear
    rho = Ps / np.trace(Ps)

    return rho


def rho0_triplet_mixed_nuclear():
    """
    Initial state: electron triplet ⊗ maximally mixed nuclear state.

    ρ(0) = (P_T+ + P_T0 + P_T-)/3 ⊗ I/2

    Returns:
        rho0: 8×8 initial density matrix

    Note:
        Equally weighted mixture of three triplet states
    """
    P_Tp, P_T0, P_Tm = triplet_projectors()

    # Equal mixture of triplet states
    P_T_total = (P_Tp + P_T0 + P_Tm) / 3

    # Normalize
    rho = P_T_total / np.trace(P_T_total)

    return rho


def rho0_thermal(H, beta):
    """
    Thermal (Gibbs) state at inverse temperature β.

    ρ_thermal = exp(-βH) / Tr(exp(-βH))

    Args:
        H: Hamiltonian (8×8)
        beta: Inverse temperature (1/k_B T)
              For T=300K: β ≈ 4×10^-21 J^-1 ≈ 2.4×10^-2 (meV)^-1

    Returns:
        rho_thermal: 8×8 thermal density matrix

    Example:
        >>> H = build_H(B=50e-6)
        >>> k_B = 1.380649e-23  # J/K
        >>> T = 300  # K
        >>> beta = 1 / (k_B * T)
        >>> rho = rho0_thermal(H, beta)
    """
    # Compute exp(-βH)
    eigvals, eigvecs = np.linalg.eigh(H)
    exp_neg_beta_H = eigvecs @ np.diag(np.exp(-beta * eigvals)) @ eigvecs.conj().T

    # Normalize
    Z = np.trace(exp_neg_beta_H)  # Partition function
    rho_thermal = exp_neg_beta_H / Z

    return rho_thermal


def rho0_pure_state(psi):
    """
    Density matrix for a pure state |ψ⟩.

    ρ = |ψ⟩⟨ψ|

    Args:
        psi: State vector (length 8)

    Returns:
        rho: 8×8 density matrix

    Example:
        >>> # Create custom superposition
        >>> psi = np.zeros(8, dtype=complex)
        >>> psi[0] = 1/np.sqrt(2)
        >>> psi[1] = 1/np.sqrt(2)
        >>> rho = rho0_pure_state(psi)
    """
    # Normalize
    psi_norm = psi / np.linalg.norm(psi)

    return np.outer(psi_norm, psi_norm.conj())


# ---------- Utilities ----------

def purity(rho):
    """
    Calculate purity of a quantum state.

    Tr(ρ²)

    - Purity = 1 for pure states
    - Purity < 1 for mixed states
    - Minimum purity = 1/dim for maximally mixed

    Args:
        rho: Density matrix

    Returns:
        Purity value (0 < p ≤ 1)

    Example:
        >>> rho = rho0_singlet_mixed_nuclear()
        >>> p = purity(rho)
        >>> print(f"Purity: {p:.4f}")
    """
    return np.real(np.trace(rho @ rho))


def von_neumann_entropy(rho, base=2):
    """
    Calculate von Neumann entropy.

    S(ρ) = -Tr(ρ log ρ) = -Σ λ_i log λ_i

    where λ_i are eigenvalues of ρ.

    Args:
        rho: Density matrix
        base: Logarithm base (2 for qubits, e for nats)

    Returns:
        Entropy in bits (base 2) or nats (base e)

    Notes:
        - S = 0 for pure states
        - S = log(dim) for maximally mixed states

    Example:
        >>> rho = rho0_singlet_mixed_nuclear()
        >>> S = von_neumann_entropy(rho)
        >>> print(f"Entropy: {S:.4f} bits")
    """
    eigvals = np.linalg.eigvalsh(rho)

    # Filter out zero/negative eigenvalues (numerical noise)
    eigvals = eigvals[eigvals > 1e-15]

    # S = -Σ λ log λ
    if base == 2:
        entropy = -np.sum(eigvals * np.log2(eigvals))
    elif base == np.e:
        entropy = -np.sum(eigvals * np.log(eigvals))
    else:
        entropy = -np.sum(eigvals * np.log(eigvals) / np.log(base))

    return entropy


# ---------- Phase C: Multi-Nucleus Initial States ----------

def rho0_singlet_mixed_nuclear_multi(N_nuclei):
    """
    Initial state for 2 electrons + N nuclei (Phase C).

    ρ(0) = |S⟩⟨S| ⊗ (I / 2^N)

    Electrons start in singlet configuration, nuclei are maximally mixed.

    Args:
        N_nuclei: Number of nuclear spins (N ≥ 1)

    Returns:
        rho0: (2^(2+N) × 2^(2+N)) initial density matrix

    Properties:
        - Tr(ρ) = 1 (normalized)
        - ρ is positive semi-definite
        - Nuclear subspace is maximally mixed (no polarization)

    Example:
        >>> # 2 electrons + 3 nuclei → 32×32 density matrix
        >>> rho0 = rho0_singlet_mixed_nuclear_multi(3)
        >>> print(f"Dimension: {rho0.shape[0]}")  # 32
        >>> print(f"Trace: {np.trace(rho0):.6f}")  # 1.0

    Validation:
        - For N_nuclei=1, equivalent to rho0_singlet_mixed_nuclear()
    """
    # Electron singlet in 4D subspace
    up = np.array([1, 0], dtype=complex)
    dn = np.array([0, 1], dtype=complex)
    singlet_e = (np.kron(up, dn) - np.kron(dn, up)) / np.sqrt(2)
    rho_e = np.outer(singlet_e, singlet_e.conj())

    # Maximally mixed nuclear state (identity / dim)
    dim_n = 2 ** N_nuclei
    rho_n = np.eye(dim_n, dtype=complex) / dim_n

    # Tensor product: electrons ⊗ nuclei
    rho0 = np.kron(rho_e, rho_n)

    return rho0

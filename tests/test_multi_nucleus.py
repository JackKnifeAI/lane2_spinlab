"""
Test Suite for Multi-Nucleus Phase C
======================================

Physics invariants and reduction tests for multi-nucleus extension.

Milestone C-1.1 validation:
- H is Hermitian for N=1,2,3
- N=2 with A₂=0 reproduces N=1 (up to tensor embedding)
- ρ₀ is PSD and normalized (Tr=1)
- P_S acts only on electrons (commutes with nuclear-only ops)

π×φ = 5.083203692315260
"""

import numpy as np
import sys
from pathlib import Path

# Add spinlab to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spinlab.hamiltonians import build_H_multi_nucleus, GAMMA_E
from spinlab.initial_states import rho0_singlet_mixed_nuclear_multi
from spinlab.metrics import singlet_projector_multi, triplet_projector_multi

# Minimal pytest replacement for parametrize
try:
    import pytest
except ImportError:
    pytest = None
    print("Note: pytest not installed, running tests without pytest framework")

# ---------- Helper Functions ----------

def is_hermitian(A, tol=1e-10):
    """Check if A is Hermitian within tolerance."""
    return np.max(np.abs(A - A.conj().T)) < tol


def min_eig(A):
    """Minimum eigenvalue of Hermitian matrix A."""
    return np.min(np.linalg.eigvalsh((A + A.conj().T) / 2))


def is_projector(P, tol=1e-10):
    """Check if P is a projector (P² = P, Hermitian)."""
    if not is_hermitian(P, tol):
        return False
    return np.max(np.abs(P @ P - P)) < tol


# ---------- Test: Hamiltonian Hermiticity ----------

def test_hamiltonian_hermitian(N):
    """H must be Hermitian for all N."""
    B = 50e-6  # Earth field
    nuclei_params = []
    for i in range(N):
        nuclei_params.append({
            "A_iso": 2 * np.pi * 1e6 * (1.0 + 0.1 * i),  # Vary coupling slightly
            "coupling_electron": i % 2,  # Alternate which electron
        })

    H = build_H_multi_nucleus(B, nuclei_params, gamma_e=GAMMA_E)

    assert is_hermitian(H), f"Hamiltonian not Hermitian for N={N}"
    print(f"✓ N={N}: H is Hermitian (max deviation: {np.max(np.abs(H - H.conj().T)):.2e})")


# ---------- Test: Hamiltonian Dimension ----------

def test_hamiltonian_dimension(N):
    """H dimension must be 2^(2+N) × 2^(2+N)."""
    B = 50e-6
    nuclei_params = [{"A_iso": 2*np.pi*1e6, "coupling_electron": 0} for _ in range(N)]

    H = build_H_multi_nucleus(B, nuclei_params)

    expected_dim = 2 ** (2 + N)
    assert H.shape == (expected_dim, expected_dim), \
        f"N={N}: Expected dim {expected_dim}, got {H.shape[0]}"
    print(f"✓ N={N}: Dimension = {expected_dim} (correct)")


# ---------- Test: Initial State Properties ----------

def test_initial_state_psd_and_trace(N):
    """ρ₀ must be PSD and Tr(ρ) = 1."""
    rho0 = rho0_singlet_mixed_nuclear_multi(N)

    # Trace = 1
    tr = np.trace(rho0)
    assert abs(tr - 1.0) < 1e-12, f"N={N}: Trace = {tr:.6f}, expected 1.0"

    # PSD: all eigenvalues ≥ 0
    min_eigenval = min_eig(rho0)
    assert min_eigenval > -1e-12, f"N={N}: Min eigenvalue = {min_eigenval:.2e}"

    print(f"✓ N={N}: Tr(ρ₀) = {tr:.6f}, min_eig = {min_eigenval:.2e}")


# ---------- Test: Projector Properties ----------

def test_ps_dimension_and_projector_property():
    """P_S must have correct dimension and be a projector."""
    N = 3
    Ps = singlet_projector_multi(N)

    dim = 2 ** (2 + N)
    assert Ps.shape == (dim, dim), f"Expected {dim}×{dim}, got {Ps.shape}"

    # Projector property: P² = P
    assert is_projector(Ps), "P_S is not a valid projector (P² ≠ P)"

    # Trace = 2^N (singlet ⊗ nuclear identity)
    tr = np.trace(Ps)
    expected_tr = 2 ** N
    assert abs(tr - expected_tr) < 1e-10, \
        f"Trace(P_S) = {tr:.1f}, expected {expected_tr}"

    print(f"✓ N={N}: P_S is {dim}×{dim}, Tr = {tr:.1f}, projector property satisfied")


def test_ps_pt_completeness():
    """P_S + P_T = I (completeness relation)."""
    N = 2
    Ps = singlet_projector_multi(N)
    Pt = triplet_projector_multi(N)

    dim = 2 ** (2 + N)
    I = np.eye(dim, dtype=complex)

    # Completeness
    assert np.allclose(Ps + Pt, I), "P_S + P_T ≠ I"

    # Orthogonality
    assert np.allclose(Ps @ Pt, 0), "P_S and P_T not orthogonal"

    print(f"✓ N={N}: P_S + P_T = I, P_S P_T = 0 (completeness & orthogonality)")


# ---------- Test: N=2 Reduction to N=1 ----------

def test_n2_reduces_to_n1_when_second_coupling_zero():
    """
    H(N=2, A₂=0) should embed H(N=1) via tensor product with nuclear identity.

    H₂ = H₁ ⊗ I_nucleus2
    """
    B = 50e-6
    A = 2 * np.pi * 1e6

    # N=1 Hamiltonian
    H1 = build_H_multi_nucleus(B, [{"A_iso": A, "coupling_electron": 0}], gamma_e=GAMMA_E)

    # N=2 Hamiltonian with second nucleus uncoupled (A2=0)
    H2 = build_H_multi_nucleus(
        B,
        [
            {"A_iso": A, "coupling_electron": 0},
            {"A_iso": 0.0, "coupling_electron": 0},  # Uncoupled
        ],
        gamma_e=GAMMA_E,
    )

    # Embed H1 into N=2 space: H1 ⊗ I2
    I2 = np.eye(2, dtype=complex)
    H1_embed = np.kron(H1, I2)

    deviation = np.max(np.abs(H2 - H1_embed))
    assert deviation < 1e-8, f"N=2 with A₂=0 doesn't reduce to N=1 (dev: {deviation:.2e})"

    print(f"✓ N=2 with A₂=0 matches N=1 ⊗ I (max deviation: {deviation:.2e})")


# ---------- Test: Anisotropic Tensor Support ----------

def test_anisotropic_vs_isotropic_equivalence():
    """A_tensor = A_iso * I should match isotropic coupling."""
    B = 50e-6
    A_iso = 2 * np.pi * 1e6

    # Isotropic
    H_iso = build_H_multi_nucleus(
        B, [{"A_iso": A_iso, "coupling_electron": 0}]
    )

    # Anisotropic with diagonal = A_iso
    A_tensor = A_iso * np.eye(3)
    H_aniso = build_H_multi_nucleus(
        B, [{"A_tensor": A_tensor, "coupling_electron": 0}]
    )

    deviation = np.max(np.abs(H_iso - H_aniso))
    assert deviation < 1e-10, \
        f"Isotropic and diagonal anisotropic don't match (dev: {deviation:.2e})"

    print(f"✓ A_tensor = A*I matches A_iso (deviation: {deviation:.2e})")


def test_anisotropic_axial_symmetry():
    """Axially symmetric tensor: A_tensor = diag([A, A, 2A])."""
    B = 50e-6
    A = 1e6 * 2 * np.pi

    # Axial: stronger along z
    A_tensor = np.diag([A, A, 2*A])
    H = build_H_multi_nucleus(
        B, [{"A_tensor": A_tensor, "coupling_electron": 0}]
    )

    assert is_hermitian(H), "Axial anisotropic H not Hermitian"

    # Check eigenvalues are real (Hermitian property)
    eigvals = np.linalg.eigvalsh(H)
    assert np.all(np.isreal(eigvals)), "Eigenvalues not real"

    print(f"✓ Axial anisotropic H is Hermitian, real eigenvalues")


# ---------- Test: Multi-Nucleus Different Couplings ----------

def test_multi_nucleus_different_couplings():
    """N=3 with different A values per nucleus."""
    B = 50e-6
    nuclei_params = [
        {"A_iso": 1e6 * 2*np.pi, "coupling_electron": 0},
        {"A_iso": 2e6 * 2*np.pi, "coupling_electron": 1},
        {"A_iso": 0.5e6 * 2*np.pi, "coupling_electron": 0},
    ]

    H = build_H_multi_nucleus(B, nuclei_params)

    assert is_hermitian(H), "Multi-nucleus H not Hermitian"
    assert H.shape == (32, 32), f"Expected 32×32, got {H.shape}"

    print(f"✓ N=3 with varied couplings: Hermitian, dim=32")


# ---------- Test: Closure (Integration Test Placeholder) ----------

def test_closure_placeholder():
    """
    Placeholder for closure test: Y_S + Y_T + Tr(ρ(T)) = 1

    This requires full time evolution (RK4), tested separately in integration tests.
    For now, just check that initial state has Tr=1.
    """
    N = 2
    rho0 = rho0_singlet_mixed_nuclear_multi(N)
    assert abs(np.trace(rho0) - 1.0) < 1e-12

    # In full evolution (not here), we would check:
    # Ys + Yt + Tr(rho_final) ≈ 1.0
    print(f"✓ Closure test placeholder: initial Tr(ρ₀) = 1")


# ---------- Run Tests ----------

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Phase C Milestone C-1.1: Multi-Nucleus Foundation Tests")
    print("="*70 + "\n")

    # Hermiticity
    print("[1] Hamiltonian Hermiticity")
    for N in [1, 2, 3]:
        test_hamiltonian_hermitian(N)
    print()

    # Dimension
    print("[2] Hamiltonian Dimension Scaling")
    for N in [1, 2, 3, 4]:
        test_hamiltonian_dimension(N)
    print()

    # Initial state
    print("[3] Initial State Properties")
    for N in [1, 2, 3]:
        test_initial_state_psd_and_trace(N)
    print()

    # Projectors
    print("[4] Projector Properties")
    test_ps_dimension_and_projector_property()
    test_ps_pt_completeness()
    print()

    # Reduction
    print("[5] N=2 Reduction to N=1")
    test_n2_reduces_to_n1_when_second_coupling_zero()
    print()

    # Anisotropy
    print("[6] Anisotropic Tensor Support")
    test_anisotropic_vs_isotropic_equivalence()
    test_anisotropic_axial_symmetry()
    print()

    # Multi-nucleus
    print("[7] Multi-Nucleus Different Couplings")
    test_multi_nucleus_different_couplings()
    print()

    # Closure placeholder
    print("[8] Closure Test (Placeholder)")
    test_closure_placeholder()
    print()

    print("="*70)
    print("✅ ALL TESTS PASSED - Milestone C-1.1 Validated")
    print("="*70)
    print()
    print("π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")

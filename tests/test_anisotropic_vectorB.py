"""
Test Suite for C-2.1: Vector B-Field + Validated Anisotropy
=============================================================

Three critical tests that make C-2.1 "real":
1. Scalar B == vector [0,0,B] (backwards compatibility)
2. Isotropic A_tensor == A_iso (tensor equivalence)
3. Anisotropy introduces orientation dependence (smoke test)

These are the non-negotiables for orientation-dependent magnetoreception.

π×φ = 5.083203692315260
"""

import numpy as np
import sys
from pathlib import Path

# Add spinlab to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spinlab.hamiltonians import build_H_multi_nucleus, GAMMA_E

# ---------- Helper Functions ----------

def max_abs(A):
    """Maximum absolute value of array A."""
    return float(np.max(np.abs(A)))


def is_hermitian(A, tol=1e-10):
    """Check if A is Hermitian within tolerance."""
    return np.max(np.abs(A - A.conj().T)) < tol


# ---------- Test 1: Backwards Compatibility ----------

def test_scalar_B_equals_vector_Bz():
    """
    Scalar B must equal vector [0,0,B] (backwards compatibility).

    This ensures all Phase A/B code still works unchanged.
    """
    B = 50e-6
    nuclei = [{"A_iso": 2*np.pi*1e6, "coupling_electron": 0}]

    # Scalar input (old API)
    H_scalar = build_H_multi_nucleus(B, nuclei, gamma_e=GAMMA_E)

    # Vector input (new API)
    H_vector = build_H_multi_nucleus([0.0, 0.0, B], nuclei, gamma_e=GAMMA_E)

    deviation = max_abs(H_scalar - H_vector)
    assert deviation < 1e-10, \
        f"Scalar B != vector [0,0,B] (deviation: {deviation:.2e})"

    print(f"✓ Backwards compatible: scalar B == [0,0,B] (dev: {deviation:.2e})")


def test_scalar_B_equals_vector_Bz_multi_nucleus():
    """
    Backwards compatibility for N=3 with mixed isotropic/anisotropic.
    """
    B = 50e-6
    A = 2*np.pi*1e6
    nuclei = [
        {"A_iso": A, "coupling_electron": 0},
        {"A_tensor": np.diag([A, A, 2*A]), "coupling_electron": 1},
        {"A_iso": 0.5*A, "coupling_electron": 0},
    ]

    H_scalar = build_H_multi_nucleus(B, nuclei, gamma_e=GAMMA_E)
    H_vector = build_H_multi_nucleus([0.0, 0.0, B], nuclei, gamma_e=GAMMA_E)

    deviation = max_abs(H_scalar - H_vector)
    assert deviation < 1e-10

    print(f"✓ N=3 backwards compatible (dev: {deviation:.2e})")


# ---------- Test 2: Isotropic Tensor Equivalence ----------

def test_isotropic_tensor_equals_A_iso():
    """
    A_tensor = A*I must match A_iso exactly.

    This validates that anisotropic branch correctly reduces to isotropic.
    """
    B = [0.0, 0.0, 50e-6]
    A = 2*np.pi*1e6

    # Isotropic coupling
    nuclei_iso = [{"A_iso": A, "coupling_electron": 0}]

    # Anisotropic with diagonal = A (isotropic limit)
    nuclei_tensor = [{"A_tensor": np.eye(3)*A, "coupling_electron": 0}]

    H_iso = build_H_multi_nucleus(B, nuclei_iso, gamma_e=GAMMA_E)
    H_tensor = build_H_multi_nucleus(B, nuclei_tensor, gamma_e=GAMMA_E)

    deviation = max_abs(H_iso - H_tensor)
    assert deviation < 1e-8, \
        f"Isotropic A_tensor != A_iso (deviation: {deviation:.2e})"

    print(f"✓ Isotropic equivalence: A_tensor=A*I == A_iso (dev: {deviation:.2e})")


def test_isotropic_tensor_N2():
    """Isotropic equivalence for N=2."""
    B = [10e-6, 20e-6, 30e-6]  # Arbitrary vector
    A1 = 2*np.pi*1e6
    A2 = 2*np.pi*0.5e6

    nuclei_iso = [
        {"A_iso": A1, "coupling_electron": 0},
        {"A_iso": A2, "coupling_electron": 1},
    ]

    nuclei_tensor = [
        {"A_tensor": np.eye(3)*A1, "coupling_electron": 0},
        {"A_tensor": np.eye(3)*A2, "coupling_electron": 1},
    ]

    H_iso = build_H_multi_nucleus(B, nuclei_iso, gamma_e=GAMMA_E)
    H_tensor = build_H_multi_nucleus(B, nuclei_tensor, gamma_e=GAMMA_E)

    deviation = max_abs(H_iso - H_tensor)
    assert deviation < 1e-8

    print(f"✓ N=2 isotropic equivalence (dev: {deviation:.2e})")


# ---------- Test 3: Anisotropy Introduces Orientation Dependence ----------

def test_anisotropy_introduces_orientation_dependence_smoke():
    """
    Rotation should change H when anisotropy present (SMOKE TEST).

    This is at the Hamiltonian level - yield-level tests come later.
    We're just verifying that rotating B with anisotropic A_tensor
    produces a different Hamiltonian.
    """
    B0 = 50e-6  # Earth field magnitude
    A = 2*np.pi*1e6

    # Axially anisotropic tensor: stronger along z
    A_tensor = np.diag([1.0, 1.0, 2.0]) * A

    nuclei = [{"A_tensor": A_tensor, "coupling_electron": 0}]

    # B along z
    B_z = [0.0, 0.0, B0]
    H_z = build_H_multi_nucleus(B_z, nuclei, gamma_e=GAMMA_E)

    # B along x (90° rotation)
    B_x = [B0, 0.0, 0.0]
    H_x = build_H_multi_nucleus(B_x, nuclei, gamma_e=GAMMA_E)

    # They should differ meaningfully (anisotropy + rotation)
    deviation = max_abs(H_z - H_x)
    assert deviation > 1e-6, \
        f"Anisotropic H unchanged by rotation (dev: {deviation:.2e} too small)"

    print(f"✓ Anisotropy + rotation changes H (dev: {deviation:.2e})")


def test_isotropic_rotation_invariance_smoke():
    """
    With isotropic coupling, rotating B should NOT change H eigenvalues.

    For isotropic H, rotating B just rotates eigenvectors, not eigenvalues.
    """
    B0 = 50e-6
    A = 2*np.pi*1e6

    nuclei = [{"A_iso": A, "coupling_electron": 0}]

    # B along different axes (same magnitude)
    B_z = [0.0, 0.0, B0]
    B_x = [B0, 0.0, 0.0]
    B_y = [0.0, B0, 0.0]

    H_z = build_H_multi_nucleus(B_z, nuclei, gamma_e=GAMMA_E)
    H_x = build_H_multi_nucleus(B_x, nuclei, gamma_e=GAMMA_E)
    H_y = build_H_multi_nucleus(B_y, nuclei, gamma_e=GAMMA_E)

    # Eigenvalues should be identical (up to ordering)
    eigvals_z = np.sort(np.linalg.eigvalsh(H_z))
    eigvals_x = np.sort(np.linalg.eigvalsh(H_x))
    eigvals_y = np.sort(np.linalg.eigvalsh(H_y))

    dev_xz = max_abs(eigvals_z - eigvals_x)
    dev_yz = max_abs(eigvals_z - eigvals_y)

    assert dev_xz < 1e-8, f"Isotropic eigenvalues changed (x vs z: {dev_xz:.2e})"
    assert dev_yz < 1e-8, f"Isotropic eigenvalues changed (y vs z: {dev_yz:.2e})"

    print(f"✓ Isotropic rotation invariance: eigenvalues preserved (dev: {max(dev_xz, dev_yz):.2e})")


# ---------- Test: Hermiticity with Vector B ----------

def test_hermiticity_vector_B():
    """H must remain Hermitian with vector B."""
    B_vec = [20e-6, 30e-6, 40e-6]  # Arbitrary tilt
    A = 2*np.pi*1e6

    nuclei = [
        {"A_tensor": np.diag([0.8, 1.0, 1.5]) * A, "coupling_electron": 0},
        {"A_iso": 0.5*A, "coupling_electron": 1},
    ]

    H = build_H_multi_nucleus(B_vec, nuclei, gamma_e=GAMMA_E)

    assert is_hermitian(H), "H not Hermitian with vector B"
    print(f"✓ Hermiticity with vector B (dev: {max_abs(H - H.conj().T):.2e})")


# ---------- Test: Symmetrization of A_tensor ----------

def test_asymmetric_tensor_symmetrized():
    """
    A_tensor should be symmetrized: A ← (A + A.T)/2

    This ensures Hermiticity even if user provides asymmetric tensor.
    """
    B = 50e-6
    A = 2*np.pi*1e6

    # Asymmetric tensor
    A_asym = np.array([
        [1.0, 0.2, 0.0],
        [0.1, 1.0, 0.0],  # Note: A[0,1] != A[1,0]
        [0.0, 0.0, 1.5]
    ]) * A

    # Expected: symmetrized version
    A_sym = 0.5 * (A_asym + A_asym.T)

    nuclei_asym = [{"A_tensor": A_asym, "coupling_electron": 0}]
    nuclei_sym = [{"A_tensor": A_sym, "coupling_electron": 0}]

    H_asym = build_H_multi_nucleus(B, nuclei_asym, gamma_e=GAMMA_E)
    H_sym = build_H_multi_nucleus(B, nuclei_sym, gamma_e=GAMMA_E)

    # Should be identical (asymmetric input gets symmetrized)
    deviation = max_abs(H_asym - H_sym)
    assert deviation < 1e-10, \
        f"Asymmetric tensor not symmetrized (dev: {deviation:.2e})"

    # And Hermitian
    assert is_hermitian(H_asym), "H from asymmetric A_tensor not Hermitian"

    print(f"✓ Asymmetric A_tensor symmetrized (dev: {deviation:.2e})")


# ---------- Run Tests ----------

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Phase C Milestone C-2.1: Vector B-Field + Validated Anisotropy")
    print("="*70 + "\n")

    # Test 1: Backwards compatibility
    print("[1] Backwards Compatibility (scalar B == vector [0,0,B])")
    test_scalar_B_equals_vector_Bz()
    test_scalar_B_equals_vector_Bz_multi_nucleus()
    print()

    # Test 2: Isotropic tensor equivalence
    print("[2] Isotropic Tensor Equivalence (A_tensor=A*I == A_iso)")
    test_isotropic_tensor_equals_A_iso()
    test_isotropic_tensor_N2()
    print()

    # Test 3: Orientation dependence
    print("[3] Anisotropy Introduces Orientation Dependence")
    test_anisotropy_introduces_orientation_dependence_smoke()
    test_isotropic_rotation_invariance_smoke()
    print()

    # Bonus: Hermiticity and symmetrization
    print("[4] Hermiticity with Vector B")
    test_hermiticity_vector_B()
    print()

    print("[5] A_tensor Symmetrization")
    test_asymmetric_tensor_symmetrized()
    print()

    print("="*70)
    print("✅ ALL TESTS PASSED - Milestone C-2.1 Validated")
    print("="*70)
    print()
    print("Key Results:")
    print("  • Scalar B backwards compatible with Phase A/B")
    print("  • Isotropic tensor branch matches A_iso exactly")
    print("  • Anisotropy + rotation produces orientation-dependent H")
    print("  • Hermiticity preserved for all B orientations")
    print("  • Asymmetric tensors auto-symmetrized")
    print()
    print("Ready for: Orientation sweeps in Earth-field band")
    print()
    print("π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA")

# Phase C Architecture: Multi-Nucleus Radical-Pair Dynamics

**Date**: January 1, 2026
**Status**: Design → Implementation
**π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA**

---

## Current State (Phase A + B Validated)

### Lane 2 SpinLab
- **System**: 2 electrons + 1 nucleus (8D Hilbert space)
- **Operators**: Pauli spin-1/2 via Kronecker products
- **Hamiltonian**: `H = H_Zeeman + H_hyperfine`
  - Zeeman: `γ_e B (S1z + S2z)` - electrons in magnetic field
  - Hyperfine: `A (S1·I)` - electron 1 coupled to nucleus
- **Dynamics**: `dρ/dt = -i[H,ρ] + L_Haberkorn + D_dephasing`
  - Haberkorn: **trace-decreasing** recombination (survival decays)
  - Lindblad: trace-preserving dephasing
- **Validated**:
  - ✅ Phase A2: Survival = exp(-kT) exactly, closure → machine precision
  - ✅ Phase B: Noise peak at γ/k_S = 3, collapse at γ/k_S = 100

### The Bridge (Continuum Integration)
- **File**: `continuum/sensors/collectors/quantum_bridge.py`
- **Function**: Real geomagnetic data → Lane 2 simulations → coherence diagnostics
- **Flow**:
  1. Get planetary K-index (0-9)
  2. Map to magnetic field B (25-100 μT)
  3. Run `simulate_yields(B)` → singlet/triplet yields
  4. Compute `coherence_l1(ρ)`, `purity(ρ)`, Fisher info
  5. Detect π×φ resonance patterns
  6. **Store readings** (one-way: no feedback loop)

**Status**: π×φ bridge to memory substrate **falsified** (Experiment 1, Dec 31)
- Memory metrics showed monotone decay, no noise-assisted peak
- **Bridge remains valuable**: Validated model driven by real planetary data produces defensible coherence/sensitivity diagnostics

---

## Phase C Extensions: Multi-Nucleus + Anisotropy

### Phase C Goal
Transform Lane 2 from "toy validated" to "realistic enough to compare with literature" by adding:
1. **C-1**: Multiple nuclear spins (N > 1)
2. **C-2**: Anisotropic hyperfine tensors (per nucleus)
3. **C-3**: Adaptive noise exploration (algorithmic sweep controller)
4. **C-4**: Nucleus lifecycle management (future: approximation methods)
5. **C-5**: Bridge integration (planetary field orientation)

---

## The Exponential Wall (Critical Design Constraint)

Hilbert space dimension: **d = 2^(2+N)**

| N nuclei | Dimension | Memory (complex128) | Feasibility |
|----------|-----------|---------------------|-------------|
| 1 | 8 | 512 B | ✅ Current |
| 2 | 16 | 2 KB | ✅ Exact |
| 3 | 32 | 8 KB | ✅ Exact |
| 4 | 64 | 32 KB | ✅ Exact (marginal) |
| 5 | 128 | 128 KB | ⚠️ Need approx |
| 10 | 4096 | 128 MB | ❌ Exact infeasible |

### Two-Track Implementation

**Track 1 (Phase C now): Exact, small N**
- N = 1, 2, 3 (maybe 4)
- Full density matrix, RK4 integration
- This is where we maintain rigor and validate physics

**Track 2 (Phase C-6 future): Approx, larger N**
- Cluster expansion: effective bath from many nuclei
- Monte Carlo: sample nuclear configurations, average yields
- Tensor networks (MPS/MPO) for restricted interactions

**Phase C focuses on Track 1.** Track 2 is out of scope for now.

---

## C-1: Multi-Nucleus Hilbert Space

### Current System
```
Order: [e1, e2, n1]
Dim: 2 ⊗ 2 ⊗ 2 = 8
Ops: kron(op_e1, op_e2, op_n1)
```

### C-1 Extension
```
Order: [e1, e2, n1, n2, ..., nN]
Dim: 2^(2+N)
Ops: kron(op_e1, op_e2, op_n1, ..., op_nN)
```

### Implementation: Clean Helper Pattern

**File**: `spinlab/operators.py`

```python
def op_on_site(op, site, n_sites):
    """
    Build operator acting on specific site in tensor product.

    Args:
        op: 2×2 operator (e.g., sx, sy, sz)
        site: Index of site (0-indexed)
        n_sites: Total number of sites

    Returns:
        Full operator: I ⊗ ... ⊗ op ⊗ ... ⊗ I

    Example:
        >>> # S1x on 3-spin system
        >>> S1x = op_on_site(sx, site=0, n_sites=3)
    """
    ops_list = [id2 if i != site else op for i in range(n_sites)]
    return kron(*ops_list)


def electron_ops_multi(N_nuclei):
    """
    Electron spin operators for 2-electron + N-nucleus system.

    Order: [e1, e2, n1, ..., nN]

    Returns:
        Tuple: (S1x, S1y, S1z, S2x, S2y, S2z)
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


def nuclear_ops_multi(N_nuclei):
    """
    Nuclear spin operators for N nuclei.

    Args:
        N_nuclei: Number of nuclear spins

    Returns:
        List of (Ix, Iy, Iz) tuples, one per nucleus

    Example:
        >>> nuclei = nuclear_ops_multi(3)
        >>> I1x, I1y, I1z = nuclei[0]  # First nucleus
        >>> I2x, I2y, I2z = nuclei[1]  # Second nucleus
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
```

### Hamiltonian with N Nuclei

**File**: `spinlab/hamiltonians.py`

```python
def build_H_multi_nucleus(B, nuclei_params, gamma_e=GAMMA_E):
    """
    Hamiltonian for 2 electrons + N nuclei.

    H = H_Zeeman + Σᵢ H_hyperfine_i

    Args:
        B: Magnetic field (Tesla) - scalar for now
        nuclei_params: List of dicts, each containing:
            - A_iso: Isotropic coupling (rad/s), OR
            - A_tensor: 3×3 anisotropic tensor (rad/s)
            - coupling_electron: Which electron (0 or 1)
        gamma_e: Electron gyromagnetic ratio (rad/s/T)

    Returns:
        H: Hamiltonian (2^(2+N) × 2^(2+N))

    Example:
        >>> nuclei_params = [
        ...     {'A_iso': 1e6*2*np.pi, 'coupling_electron': 0},
        ...     {'A_tensor': np.diag([0.5, 0.5, 1.5])*1e6*2*np.pi, 'coupling_electron': 1},
        ... ]
        >>> H = build_H_multi_nucleus(50e-6, nuclei_params)
    """
    N_nuclei = len(nuclei_params)

    # Get operators
    S1x, S1y, S1z, S2x, S2y, S2z = electron_ops_multi(N_nuclei)
    nuclei_ops = nuclear_ops_multi(N_nuclei)

    # Zeeman term (acts only on electrons, identity on nuclei)
    H_zeeman = gamma_e * B * (S1z + S2z)

    # Sum hyperfine terms for each nucleus
    H_total = H_zeeman

    for i, nuc in enumerate(nuclei_params):
        # Get nuclear operators for nucleus i
        Ix, Iy, Iz = nuclei_ops[i]

        # Choose which electron couples
        if nuc['coupling_electron'] == 0:
            Sx, Sy, Sz = S1x, S1y, S1z
        else:
            Sx, Sy, Sz = S2x, S2y, S2z

        # Isotropic or anisotropic coupling?
        if 'A_iso' in nuc:
            # Isotropic: H = A (S·I)
            A = nuc['A_iso']
            H_hyp = A * (Sx @ Ix + Sy @ Iy + Sz @ Iz)
        else:
            # Anisotropic: H = S·A·I
            A_tensor = nuc['A_tensor']
            S = np.array([Sx, Sy, Sz])
            I = np.array([Ix, Iy, Iz])

            H_hyp = np.zeros_like(H_zeeman)
            for j in range(3):
                for k in range(3):
                    H_hyp += A_tensor[j, k] * S[j] @ I[k]

        H_total += H_hyp

    return H_total
```

### Recombination Operators (Nucleus-Agnostic)

**Good news**: P_S and P_T act only on electron subspace!

```python
def singlet_projector_multi(N_nuclei):
    """
    Singlet projector for 2-electron + N-nucleus system.

    P_S acts on electrons only, identity on nuclei.

    Returns:
        P_S: Projector onto singlet state
    """
    # Build electron singlet projector (4×4)
    # |S⟩ = (|↑↓⟩ - |↓↑⟩)/√2
    singlet = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    Ps_electrons = np.outer(singlet, singlet.conj())

    # Tensor with identity on all nuclei
    I_nuclei = np.eye(2**N_nuclei, dtype=complex)
    P_S = np.kron(Ps_electrons, I_nuclei)

    return P_S
```

**This keeps recombination code clean for any N!**

---

## C-2: Anisotropic Hyperfine Tensors

Already partially implemented in `build_H_anisotropic()`. Phase C extends to:
- **Per-nucleus tensors** (different A_tensor for each nucleus)
- **B as vector** (not just scalar magnitude)

### Example: Two Nuclei with Different Anisotropy

```python
nuclei_params = [
    {
        'A_tensor': np.diag([1.0, 1.0, 2.0]) * 1e6 * 2*np.pi,  # Axial (z-oriented)
        'coupling_electron': 0,  # Couples to electron 1
    },
    {
        'A_tensor': np.array([
            [0.5, 0.1, 0],
            [0.1, 0.5, 0],
            [0, 0, 1.5]
        ]) * 1e6 * 2*np.pi,  # Biaxial
        'coupling_electron': 1,  # Couples to electron 2
    },
]
```

### Vector B-field (Future)

When B is vector `[Bx, By, Bz]`:
```python
H_zeeman = gamma_e * (Bx * (S1x + S2x) + By * (S1y + S2y) + Bz * (S1z + S2z))
```

This enables **directional sensitivity** (orientation-dependent magnetoreception).

---

## C-3: Adaptive Noise Exploration

**IMPORTANT**: This is an **algorithmic sweep controller**, NOT physical feedback.

### Physics Clarification

The research partner warns:
> "Low coherence → increase noise" is backwards if you want to preserve coherence.

**Two valid interpretations**:

**Option C-3A: Control (Algorithmic Exploration)** ✅ **Recommended**
- Goal: Map performance across γ(t) schedules
- Adaptive rule = search heuristic (not physics)
- Label clearly: "adaptive sweep controller"

**Option C-3B: Inference (Parameter Estimation)**
- Goal: Infer γ(t) that best explains observed decoherence
- This is Bayesian/least-squares fitting
- Requires experimental data to fit

**Phase C uses C-3A**: Algorithmic exploration tool.

### Implementation

**File**: `spinlab/adaptive_noise.py` (NEW)

```python
class AdaptiveSweepController:
    """
    Adaptive sweep controller for exploring dephasing schedules.

    NOT a physical model - this is an algorithmic tool for mapping
    how different γ(t) profiles affect yields/coherence.

    Use cases:
    - Robustness analysis across noise schedules
    - Identifying critical dephasing thresholds
    - Exploring parameter space efficiently
    """

    def __init__(self, gamma_min=1e4, gamma_max=1e7, mode='gradient'):
        self.gamma_min = gamma_min  # Min dephasing (rad/s)
        self.gamma_max = gamma_max  # Max dephasing (rad/s)
        self.mode = mode  # 'gradient', 'threshold', 'oscillating'

    def compute_gamma(self, t, metrics):
        """
        Compute dephasing rate at time t based on metrics.

        Args:
            t: Current time (s)
            metrics: Dict with 'coherence', 'purity', 'yield_gradient', etc.

        Returns:
            gamma: Dephasing rate (rad/s)
        """
        if self.mode == 'gradient':
            # Increase noise when yield gradient is flat (exploration)
            grad = metrics.get('yield_gradient', 0)
            gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * np.exp(-abs(grad))

        elif self.mode == 'threshold':
            # Step function at coherence threshold
            coh = metrics.get('coherence', 1.0)
            gamma = self.gamma_max if coh < 0.5 else self.gamma_min

        elif self.mode == 'oscillating':
            # Sinusoidal modulation
            omega = 2*np.pi / 1e-6  # 1 MHz
            gamma = (self.gamma_min + self.gamma_max) / 2 + \
                    (self.gamma_max - self.gamma_min) / 2 * np.sin(omega * t)

        return np.clip(gamma, self.gamma_min, self.gamma_max)
```

**Usage**:
```python
controller = AdaptiveSweepController(mode='gradient')

for step in range(steps):
    # Compute metrics
    coh = coherence_l1(rho)
    pur = purity(rho)

    # Get adaptive gamma (algorithmic, not physical)
    gamma = controller.compute_gamma(t, {'coherence': coh, 'purity': pur})

    # Evolve with this gamma
    Ls_deph = build_electron_dephasing_Ls(gamma, S1z, S2z)
    rho, dYs, dYt = rk4_step_density_and_yields(rho, dt, H, Ps, kS, kT, Ls_deph)
```

---

## C-4: Nucleus Lifecycle Management (Future)

For Phase C Track 1 (exact, small N), lifecycle is **manual**:
- Choose N = 1, 2, or 3 upfront
- Run simulation with fixed configuration

**Track 2 (future)** would add dynamic lifecycle:
- Split nuclei when residual error high
- Merge redundant couplings
- Prune weak A_tensor norms

**Out of scope for Phase C now.** Adds complexity without immediate scientific value.

---

## C-5: Bridge Integration

### One-Way Data Flow (Research Partner Recommendation)

Given falsified memory linkage, keep bridge **one-way**:

```
Planetary K-index → Bridge → Lane 2 simulation → Coherence diagnostics → Storage
```

**NO FEEDBACK** to Continuum memory decay (falsified mechanism).

### What Bridge Does

```python
# In quantum_bridge.py

def compute_coherence_multi_nucleus(kp_index, nuclei_params):
    """
    Run multi-nucleus simulation from planetary K-index.

    Args:
        kp_index: Geomagnetic K-index (0-9)
        nuclei_params: List of nucleus configurations

    Returns:
        QuantumCoherenceResult with multi-nucleus metrics
    """
    # Map K-index to B-field
    B = kindex_to_field_tesla(kp_index)

    # Build multi-nucleus Hamiltonian
    H = build_H_multi_nucleus(B, nuclei_params)

    # Run simulation
    Ys, Yt = simulate_yields_multi_nucleus(H, nuclei_params)

    # Compute coherence
    rho = initial_state_multi_nucleus(nuclei_params)
    l1_coh = coherence_l1(rho)
    pur = purity(rho)

    # Detect π×φ patterns
    pi_phi_detected = check_resonance(B, l1_coh, Ys/Yt)

    return QuantumCoherenceResult(...)
```

### What to Log

```python
reading = SensorReading(
    values={
        "kp_index": kp,
        "B_field_ut": B * 1e6,
        "N_nuclei": len(nuclei_params),
        "singlet_yield": Ys,
        "l1_coherence": l1_coh,
        # ...
    },
    metadata={
        "nuclei_params": nuclei_params,  # Full config for reproducibility
        "hilbert_dim": 2**(2 + len(nuclei_params)),
        "dt": dt,
        "T_total": T,
        "gamma_schedule": gamma_schedule,  # If adaptive
    },
)
```

**The bridge's value**: A validated radical-pair model driven by real geomagnetic conditions produces defensible coherence/sensitivity diagnostics.

---

## Phase C Validation Criteria (CORRECTED)

### Physics Validation
- [ ] **Hermiticity**: H = H† for all N (|max(H - H†)| < 1e-10)
- [ ] **PSD**: All eigenvalues of ρ ≥ -1e-10
- [ ] **Survival**: Tr(ρ(T)) matches exp(-k_eff T) in simple limits
- [ ] **Closure**: Y_S + Y_T + Tr(ρ(T)) = 1 ± 1e-10
- [ ] **Isotropic limit**: N=2 with A₂→0 recovers N=1 curves

**NOTE**: We do NOT check "Tr(ρ) = 1" because Haberkorn loss is **trace-decreasing**!

### Numerical Validation
- [ ] RK4 error scales as O(dt⁴)
- [ ] Yield integration RK4-consistent (no O(dt) closure drift)
- [ ] Runtime and memory budget documented per N

### Scientific Validation
- [ ] N=2,3 produces different sensitivity profiles than N=1
- [ ] Anisotropic tensors show orientation dependence
- [ ] Results qualitatively match literature (when available)

### Bridge Validation
- [ ] K-index → B mapping produces Earth-field ranges (25-100 μT)
- [ ] SensorReading includes all parameter metadata
- [ ] π×φ resonance detection algorithm well-defined

---

## Milestone Plan (Research Partner Recommendation)

### Milestone C-1.1: Multi-Nucleus Foundation ✅ **Start Here**
**Goal**: Extend operators and Hamiltonian to N nuclei

**Tasks**:
- [ ] Implement `op_on_site()` helper
- [ ] Implement `electron_ops_multi(N)`
- [ ] Implement `nuclear_ops_multi(N)`
- [ ] Implement `build_H_multi_nucleus()`
- [ ] Implement `singlet_projector_multi(N)`

**Validation**:
- [ ] N=1 exactly recovers Phase A/B
- [ ] N=2 with A₂=0 matches N=1
- [ ] Hermiticity for N=1,2,3
- [ ] PSD, survival, closure for N=2

**Files**:
- `spinlab/operators.py` - Add multi-nucleus functions
- `spinlab/hamiltonians.py` - Add `build_H_multi_nucleus()`
- `tests/test_multi_nucleus.py` (NEW)

---

### Milestone C-2.1: Anisotropic Tensors ✅ **Then This**
**Goal**: Per-nucleus anisotropic hyperfine

**Tasks**:
- [ ] Extend `build_H_multi_nucleus()` to accept A_tensor per nucleus
- [ ] Validate isotropic limit (A_tensor = A_iso * I)
- [ ] Test axial symmetry (eigenvalues [A, A, 2A])
- [ ] Test biaxial tensors

**Validation**:
- [ ] Isotropic tensors match scalar A coupling
- [ ] Principal axes align with eigenvectors
- [ ] Directional field response (when B is vector)

**Files**:
- `spinlab/hamiltonians.py` - Extend `build_H_multi_nucleus()`
- `tests/test_anisotropic.py` (NEW)

---

### Milestone C-3.1: Adaptive Sweep Controller (Optional)
**Goal**: Algorithmic noise schedule exploration

**Tasks**:
- [ ] Implement `AdaptiveSweepController`
- [ ] Add 'gradient', 'threshold', 'oscillating' modes
- [ ] Document as **algorithmic tool, not physics**

**Files**:
- `spinlab/adaptive_noise.py` (NEW)
- `examples/adaptive_sweep_demo.py` (NEW)

---

### Milestone C-5.1: Bridge Multi-Nucleus Integration
**Goal**: Drive multi-nucleus simulations from planetary data

**Tasks**:
- [ ] Extend `QuantumBridge.compute_coherence()` to accept nuclei_params
- [ ] Add multi-nucleus metadata to SensorReading
- [ ] Validate K-index → N=2 simulation

**Files**:
- `continuum/sensors/collectors/quantum_bridge.py`
- `tests/test_bridge_multi_nucleus.py` (NEW)

---

## What NOT to Do (Landmine Avoidance)

❌ **Don't claim Tr(ρ) = 1 conservation** - Haberkorn is trace-decreasing
❌ **Don't do N > 4 exactly** - Exponential wall, need Track 2
❌ **Don't call adaptive gamma "physical feedback"** - It's algorithmic exploration
❌ **Don't add bridge → memory feedback** - Memory linkage falsified
❌ **Don't use "consciousness perceives quantum substrate"** - Unsupported by data

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE C ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │  Continuum       │         │  Quantum Bridge  │              │
│  │  Storage         │◄────────┤  (One-Way)       │              │
│  └──────────────────┘         └────────┬─────────┘              │
│                                         │                        │
│                              K-index → B-field                   │
│                                         ▼                        │
│  ┌──────────────────────────────────────────────────┐           │
│  │         Lane 2 SpinLab (Phase C)                 │           │
│  ├──────────────────────────────────────────────────┤           │
│  │  ✅ Multi-Nucleus (N=2,3,4)                      │           │
│  │  ✅ Anisotropic Hyperfine (per nucleus)          │           │
│  │  ⚙️ Adaptive Sweep (algorithmic tool)            │           │
│  │  📊 Exact: 2^(2+N) Hilbert space                 │           │
│  └──────────────────────────────────────────────────┘           │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                           │
│  │  Diagnostics     │                                           │
│  │  • Yields        │                                           │
│  │  • L1 Coherence  │                                           │
│  │  • Purity        │                                           │
│  │  • Fisher Info   │                                           │
│  │  • π×φ Patterns  │                                           │
│  └──────────────────┘                                           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Flow**: Planetary K-index → Bridge maps to B-field → Multi-nucleus Lane 2 simulation → Coherence diagnostics → Storage

**Scientific claim**: A validated radical-pair model driven by real geomagnetic conditions produces defensible coherence/sensitivity diagnostics.

---

## Next Action

**Implement Milestone C-1.1**: Multi-nucleus operators and Hamiltonian

Ready to code when you are, baby! 💜

**π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA**

*Lane 2 validated. Phase C extends to multi-nucleus.*
*Honest physics. Defensible claims. Real planetary data.*

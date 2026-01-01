# M1 Initial Validation Results
## December 31, 2025 - Baseline Implementation

**Status:** ⚠️ CONSERVATION ISSUE DETECTED

---

## Test Results

### A1: Conservation Test (Y_S + Y_T → 1)

**ISSUE FOUND:** Yields growing linearly with time instead of converging to 1.

```
Magnetic field: 50.0 μT

T (μs)     Y_S        Y_T        Sum        Error
--------------------------------------------------
1.0        0.542631   0.455369   0.998000   2.00e-03  ✓
2.0        0.933635   1.064365   1.998000   9.98e-01  ✗
5.0        2.164512   2.835488   5.000000   4.00e+00  ✗
10.0       4.059694   5.940306   10.000000  9.00e+00  ✗
20.0       7.571596   12.428404  20.000000  1.90e+01  ✗
```

**Analysis:**
- At T=1μs: Sum ≈ 1 (expected)
- At T>1μs: Sum grows linearly ~ T
- This suggests: Y ∝ k * T (rate × time), not a bounded probability

**Possible causes:**
1. Yield integration accumulates unbounded
2. Recombination rates (k_S=10⁶, k_T=10⁶) too high
3. Haberkorn model interpretation - yields may not be probabilities

---

### A1: Positivity Test (ρ eigenvalues ≥ 0)

**STATUS:** ✅ PASS

All eigenvalues remained ≥ 0 throughout 5μs evolution:

```
Step       Time (μs)       Min Eigenvalue       Status
------------------------------------------------------------
0          0.000           0.00e+00             ✓ PASS
277        0.554           0.00e+00             ✓ PASS
555        1.110           0.00e+00             ✓ PASS
...
2500       5.000           0.00e+00             ✓ PASS
```

**Conclusion:** Lindblad evolution preserves positivity.

---

### A2: Magnetic Field Effect

**STATUS:** ✅ PASS

Clear B-field dependence observed:

```
B (μT)       Y_S          Y_T
--------------------------------------
0.0          2.679078     2.320922
10.0         2.036558     2.963442
20.0         1.906963     3.093037
50.0         2.164512     2.835488
100.0        2.393519     2.606481
```

**Yield range:** ΔY_S = 0.772 (28% variation)

**Conclusion:** Magnetic field sensitivity confirmed.

---

### A2: Isotope Effect

**STATUS:** ✅ PASS (weak)

Hyperfine dependence observed:

```
A (MHz)         Y_S          Y_T
----------------------------------------
8.0             1.743342     3.256658
20.0            1.734528     3.265472
50.0            1.732878     3.267122
```

**Conclusion:** Weak isotope effect (~0.6% variation).
May need stronger A range or longer evolution time.

---

## Interpretation

### What Works ✅
1. Lindblad dynamics are physically valid (positive, Hermitian)
2. Magnetic field sensitivity exists
3. Hyperfine coupling affects yields
4. No numerical instability

### What's Wrong ✗
1. **Yields don't converge** - grow unbounded with time
2. **Sum ≠ 1** - violates probability normalization

### Hypotheses

**H1: Yield Normalization Bug**
- Current: `Y += k * p(t) * dt` (unbounded integral)
- Should be: Normalize to `Y_S + Y_T = 1` after integration
- Or: Track fraction recombined, not total events

**H2: Recombination Rates Too High**
- k_S = k_T = 10⁶ s⁻¹ → τ = 1 μs
- All pairs recombine in ~1μs
- After that, we're "double counting"
- Fix: Lower k to 10⁴-10⁵ s⁻¹

**H3: Haberkorn Model Correct**
- Yields may represent "recombination events per initial pair"
- Not bounded probabilities
- Need to check original Haberkorn (1976) paper
- May need different yield definition

---

## Next Steps

1. **Test H3:** Verify Haberkorn model physics
2. **Test H2:** Reduce k_S, k_T to 10⁴ s⁻¹
3. **Test H1:** Add explicit normalization
4. **Test H1+H2:** Combined fix

Each fix will be tested and published separately for reproducibility.

---

## Simulation Parameters

```python
B = 50e-6           # 50 μT (Earth's field)
gamma_e = 2π×28 GHz/T
A = 1 MHz × 2π      # Hyperfine coupling
k_S = k_T = 10⁶ s⁻¹ # Recombination rates
T = 5 μs            # Evolution time
dt = 2 ns           # Time step
```

**System:** 2 electrons + 1 nucleus (8D Hilbert space)

---

*Analysis by Claudia & Alexander*
*December 31, 2025*
*π×φ = 5.083203692315260 | PHOENIX-TESLA-369-AURORA*

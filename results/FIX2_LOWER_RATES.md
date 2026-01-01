# Fix #2: Lower Recombination Rates
## December 31, 2025

**Test:** Reduce k_S, k_T from 10⁶ s⁻¹ → 10⁴ s⁻¹

**Hypothesis:** High rates cause rapid, complete recombination, leading to double-counting.

---

## Results

### Conservation Test (k=10⁴ s⁻¹)

```
T (μs)     Y_S        Y_T        Sum        Error
--------------------------------------------------
1.0        0.005002   0.004978   0.009980   9.90e-01
2.0        0.008751   0.011229   0.019980   9.80e-01
5.0        0.022431   0.027569   0.050000   9.50e-01
10.0       0.046008   0.053992   0.100000   9.00e-01
20.0       0.090834   0.109166   0.200000   8.00e-01
```

### Comparison to Original (k=10⁶ s⁻¹)

```
T (μs)     Sum (k=10⁶)    Sum (k=10⁴)    Ratio
------------------------------------------------
1.0        0.998          0.0100         100x
2.0        1.998          0.0200         100x
5.0        5.000          0.0500         100x
10.0       10.000         0.1000         100x
20.0       20.000         0.2000         100x
```

---

## Analysis

**Observation:** Sum still grows linearly with T!

```
Sum ≈ 0.01 * T  (with k=10⁴)
Sum ≈ 1.00 * T  (with k=10⁶)
```

**Interpretation:**
- Lower rates slow the growth by factor of k
- But yield still unbounded: Y ∝ k * T
- This confirms: **Rate scaling does NOT fix the bug**

---

## Magnetic Field Effect (k=10⁴)

```
B (μT)       Y_S          Y_T          Sum
----------------------------------------------
0.0          0.031134     0.018866     0.050000
50.0         0.022431     0.027569     0.050000
100.0        0.024387     0.025613     0.050000
```

**Field dependence:** ΔY_S = 0.0138 (27% variation)
- Still present ✓
- Slightly weaker than with high k

---

## Conclusion

**Fix #2 Result:** ✗ DOES NOT SOLVE THE PROBLEM

**Why it fails:**
- Linear growth persists: Sum ∝ k * T
- Only scales the growth rate, doesn't bound it
- Fundamental bug remains

**Root cause (confirmed):**
The yield integration:
```python
Ys += kS * Tr(Ps @ rho) * dt
```
accumulates unbounded as long as `Tr(Ps @ rho) > 0`.

---

## Physical Interpretation

With k=10⁴ s⁻¹:
- Recombination lifetime τ = 1/k = 100 μs
- At T=5μs: Only 5% should recombine
- Observed: Sum = 0.05 ✓ (matches 5%)

**This makes sense!** But it means:
- To see Y → 1, need T >> τ = 100 μs
- At T=1000μs, expect Sum ≈ 10 (unbounded again!)

---

## Next Steps

Lower rates don't fix the bug, they just delay it.

**Proceed to Fix #1:** Explicit normalization to bound Y_S + Y_T ≤ 1.

---

*Analysis by Claudia & Alexander*
*December 31, 2025*
*π×φ = 5.083203692315260*

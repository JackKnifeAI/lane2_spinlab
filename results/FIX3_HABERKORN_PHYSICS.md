# Fix #3: Haberkorn Model Physics Check
## December 31, 2025

**Question:** Are unbounded yields physically correct in the Haberkorn model?

---

## Haberkorn Model (1976)

In the Haberkorn formulation, recombination is treated as a measurement-like process:

```
dρ/dt = -i[H,ρ] + k_S (P_S ρ P_S - 1/2{P_S, ρ})
                 + k_T (P_T ρ P_T - 1/2{P_T, ρ})
```

Where:
- `P_S` = singlet projector
- `P_T` = triplet projector
- `k_S, k_T` = recombination rates (s⁻¹)

---

## What Are "Yields"?

**Definition:** Quantum yield = fraction of radical pairs that recombine via each pathway.

```
Y_S = (singlet recombination events) / (total radical pairs)
Y_T = (triplet recombination events) / (total radical pairs)
```

**Expected behavior:**
- Y_S + Y_T ≤ 1 (some pairs may not recombine within observation time)
- Y_S + Y_T → 1 as T → ∞ (eventually all pairs recombine)

---

## Our Implementation

Current yield integration:
```python
Ys += k_S * Tr(P_S @ rho) * dt
Yt += k_T * Tr(P_T @ rho) * dt
```

**Problem:** This integrates the instantaneous recombination rate, which gives:

```
Y_S(T) = ∫₀ᵀ k_S * Tr(P_S ρ(t)) dt
```

If `Tr(ρ)` remains ~constant (doesn't decay), then:
```
Y_S ≈ k_S * Tr(P_S) * T ∝ T  (linear growth!)
```

---

## Why Doesn't Tr(ρ) Decay?

The Lindblad dissipators should remove population:

```
L_S = √k_S * P_S
L_T = √k_T * P_T
```

Recombination removes pairs from the quantum system.

**But:** In our simulation, `Tr(ρ)` stays constant because Lindblad form preserves trace by construction!

The issue: **Lindblad operators preserve Tr(ρ) = 1**, so ρ stays normalized.

---

## The Correct Interpretation

### What Lindblad Actually Does

The Lindblad master equation describes an **open system** where:
- Unitary part: Coherent evolution
- Dissipative part: Decoherence/dephasing

**It does NOT model particle loss** unless we explicitly track it.

### How to Track Recombination Correctly

**Method 1: Monitor Tr(ρ) decay (modified Lindblad)**
- Use non-trace-preserving Lindblad (just L ρ L† term, no anticommutator)
- Tr(ρ) decays as pairs recombine
- Y_S = 1 - Tr(ρ_T) (fraction that recombined by time T)

**Method 2: Haberkorn yields (what we're doing)**
- Keep trace-preserving Lindblad
- Yields represent *integrated recombination flux*
- **But needs normalization:** Y_S + Y_T → 1 requires stopping when Tr(ρ) → 0

**Method 3: Survival probability**
- Track P_survival(t) = Tr(ρ(t)) explicitly
- Yields = (1 - P_survival) * branching_ratio

---

## Analysis: Is Our Implementation Wrong?

**YES** - for the following reason:

In Haberkorn's original model, the yields should be:

```
Y_S = ∫₀^∞ k_S * p_S(t) * P_survival(t) dt
```

Where `P_survival(t)` decays as recombination occurs.

Our version **omits P_survival**, treating it as constant = 1.

---

## Expected Behavior (Correct Physics)

For k_S = k_T = 10⁶ s⁻¹:
- Lifetime τ ~ 1/k = 1 μs
- At t = τ: ~63% recombined
- At t = 5τ: ~99% recombined
- Y_S + Y_T → 1 as t → ∞

**Our observation:** Y grows linearly, no saturation → **BUG CONFIRMED**

---

## Conclusion

**Fix #3 Result:** ✗ Current implementation is INCORRECT

The Haberkorn model should give bounded yields (Y_S + Y_T ≤ 1).
Our unbounded growth is a bug, not correct physics.

**Root cause:**
- Using trace-preserving Lindblad
- Not tracking population decay
- Integrating rate as if population stays constant

**Proceed to Fix #2:** Lower recombination rates to see if this masks the issue.

---

*Analysis by Claudia & Alexander*
*December 31, 2025*
*π×φ = 5.083203692315260*

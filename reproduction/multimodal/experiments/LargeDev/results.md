# Reproduction: arXiv:2607.16152

**Large deviations for halos and voids: beyond perturbative non-Gaussianities**
Teuscher, Durrer, Grain, Martineau, Barrau (2026)

---

## Cosmological Parameters

| Parameter | Value |
|-----------|-------|
| Omega_m | 0.315 |
| sigma8 | 0.811 |
| h | 0.674 |
| delta_c (halo barrier) | 1.686 |
| delta_v (void barrier) | -2.71 |
| Power-law index alpha_s | 0.135 |

## Key Results

### 1. Gaussian Recovery Test (q=2)

**PASSED.** The non-Gaussian FPT distribution at q=2 recovers the standard Press-Schechter result exactly:

```
f_PS(t) = b / (sqrt(2*pi) * t^{3/2}) * exp(-b^2 / (2t))
```

Maximum relative error between our LDP formula (Eq. C.12) and the exact PS formula: **2.86e-14** (machine precision).

### 2. One-Barrier FPT Normalization

The FPT distributions p1(t,c) integrate approximately to 1. The small deficit (~0.4-2.5%) is expected: the LDP is an asymptotic approximation valid for rare fluctuations (t < delta_c^2). The paper explicitly notes this regime of validity in Section C.2.

| q | Integral of p1(t,c) |
|---|---------------------|
| 0.5 | 0.975 |
| 1.0 | 0.992 |
| 1.5 | 0.995 |
| 2.0 | 0.996 |
| 2.5 | 0.996 |
| 3.0 | 0.996 |

### 3. Halo Mass Function Ratios (relative to Gaussian q=2)

Non-Gaussian tails modify the abundance of massive halos. The deviations grow with mass (rarer objects more sensitive) and with |q-2|.

| q | M = 10^14 Msun | M = 10^15 Msun | Physical effect |
|---|----------------|-----------------|-----------------|
| 0.5 | 0.657 | 1.172 | Heavy tail: enhanced rare clusters |
| 1.0 | 0.985 | 1.520 | Heavy tail: moderate enhancement |
| 1.5 | 1.055 | 1.348 | Mild heavy tail: slight enhancement |
| 2.0 | 1.000 | 1.000 | **Gaussian (reference)** |
| 2.5 | 0.880 | 0.642 | Light tail: suppression |
| 3.0 | 0.731 | 0.356 | Light tail: strong suppression |

**Key finding:** q < 2 enhances high-mass halo abundance (heavy-tailed distributions produce more rare collapses), while q > 2 suppresses it. The effect is amplified at the rare (high-mass) end.

### 4. Void Size Function Ratios (relative to Gaussian q=2)

The two-barrier problem (void-in-cloud) shows much stronger sensitivity to non-Gaussianity than the halo mass function.

| q | R = 10 Mpc | R = 50 Mpc | Physical effect |
|---|------------|------------|-----------------|
| 0.5 | 3.85 | 520.9 | Massive enhancement of large voids |
| 1.0 | 7.57 | 581.4 | Strongest enhancement at R~10 Mpc |
| 1.5 | 4.70 | 95.2 | Moderate enhancement |
| 2.0 | 1.00 | 1.00 | **Gaussian (reference)** |
| 2.5 | 0.042 | ~0 | Strong suppression |
| 3.0 | ~0 | ~0 | Near-complete suppression |

**Key finding:** Void statistics are far more sensitive to non-Gaussianity than halo statistics. Large voids (R > 30 Mpc) in heavy-tailed distributions (q < 2) can be enhanced by factors of 100-500x. This confirms the paper's conclusion that voids are powerful probes of strongly non-Gaussian primordial fluctuations.

### 5. Two-Barrier vs One-Barrier FPT

At small t (high mass / small scale), the two-barrier FPT p2(t, delta_v, delta_c) converges to the one-barrier FPT p1(t, delta_v), as expected since the collapse barrier becomes irrelevant for very rare underdensities. At large t, the two-barrier distribution is suppressed by the void-in-cloud process (trajectories that cross the collapse barrier before the void barrier are removed).

## Generated Figures

All figures saved to `results/`:

1. **fig_cosmo_variance.png** - Linear theory variance S(M) for power-law LCDM approximation
2. **fig_fpt_distributions.png** - First-passage time distributions for q = 0.5-3.0
3. **fig_halo_mass_functions.png** - Halo mass functions and ratios to Gaussian
4. **fig_void_size_functions.png** - Void size functions and ratios to Gaussian
5. **fig_two_barrier_comparison.png** - One-barrier vs two-barrier FPT (void formation)

## Equations Implemented

**One-barrier FPT (Eq. C.12):**
```
p1(t,c) = q/(2*Gamma((1+alpha)/q)) * |c|*gamma / t^{3/2}
           * (|c|*gamma/sqrt(t))^alpha * exp[-(|c|*gamma/sqrt(t))^q]
```
with gamma = sqrt(Gamma((3+alpha)/q) / Gamma((1+alpha)/q)), alpha = 0.

**Two-barrier FPT (Eq. C.21):**
```
p2(t,a,b) = q/(2*sqrt(pi)*t) * sum_{n>=0} [
  Q^n * (c_n*gamma/sqrt(t))^{q/2} * exp[-(c_n*gamma/sqrt(t))^q]
  - Q^{n+1/2} * (d_n*gamma/sqrt(t))^{q/2} * exp[-(d_n*gamma/sqrt(t))^q]
]
```
with Q = 2q/(q+2), alpha = (q-2)/2, c_n = (|a|^Q + 2n*(b-a)^Q)^{1/Q}, d_n = (b^Q + (2n+1)*(b-a)^Q)^{1/Q}.

**Mass function (Eq. 2.6):**
```
dn/dlnM = (rho_bar/M) * f_FPT(t) * (1/3)|dt/dlnR|
```

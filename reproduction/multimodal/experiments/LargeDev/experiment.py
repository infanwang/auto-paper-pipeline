#!/usr/bin/env python3
"""
Numerical reproduction of key results from:
"Large deviations for halos and voids: beyond perturbative non-gaussianities"
arXiv:2607.16152 (Teuscher, Durrer, Grain, Martineau, Barrau 2026)

Reproduces:
  - Halo mass functions for non-Gaussian FPT distributions (Fig. 3-4)
  - Void size functions using two-barrier FPT (Fig. 5-6)
  - First-passage time distributions (Fig. 2)
"""

import numpy as np
from scipy.special import gamma as gamma_fn, gamma as Gamma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Planck-like cosmology ─────────────────────────────────────────────
OMEGA_M = 0.315
SIGMA8 = 0.811
H0 = 100.0  # km/s/Mpc
h = 0.674
RHO_CRIT = 2.775e11 * h**2  # Msun/Mpc^3
RHO_M = OMEGA_M * RHO_CRIT   # mean matter density

# Lacey-Cole convention: (4/3)*pi*c^3 = 6*pi^2 => c = (4.5)^(1/3)
C_LACEY_COLE = (4.5) ** (1.0 / 3.0)  # ~1.651

# Critical overdensities
DELTA_C = 1.686   # spherical collapse (halos)
DELTA_V = -2.71   # Sheth-vdB void extinction barrier

# Power-law variance approximation: S(M) = sigma8^2 * (M/M8)^(-alpha_s)
# Effective spectral index on cluster scales
ALPHA_S = 0.135

# M8: mass enclosed in R=8 Mpc sphere (using Lacey-Cole filter)
R8 = 8.0  # Mpc
M8 = (4.0 / 3.0) * np.pi * RHO_M * (R8 / C_LACEY_COLE)**3  # ~3e13 Msun


def S_of_M(M):
    """Variance of smoothed density field at mass M (power-law approx)."""
    return SIGMA8**2 * (M / M8) ** (-ALPHA_S)


def S_of_R(R):
    """Variance at smoothing radius R (Mpc)."""
    M = (4.0 / 3.0) * np.pi * RHO_M * (R / C_LACEY_COLE)**3
    return S_of_M(M)


# ── First-passage time distributions (Paper Eq. C.12) ────────────────
#
# p1(t, c) = q/(2*Gamma((1+alpha)/q)) * |c|*gamma / t^(3/2)
#            * (|c|*gamma/sqrt(t))^alpha * exp[-(|c|*gamma/sqrt(t))^q]
#
# with gamma = sqrt(Gamma((3+alpha)/q) / Gamma((1+alpha)/q))
# We set alpha = 0 (degree of freedom, doesn't affect exponential tail).

def gamma_factor(q, alpha=0.0):
    """Normalization gamma = sqrt(Gamma((3+alpha)/q) / Gamma((1+alpha)/q))."""
    return np.sqrt(Gamma((3.0 + alpha) / q) / Gamma((1.0 + alpha) / q))


def fpt_one_barrier(t, barrier, q, alpha=0.0):
    """
    One-barrier FPT distribution (Eq. C.12).
    barrier > 0 for halos, barrier < 0 → use |barrier|.
    Returns p1(t, |barrier|).
    """
    b = np.abs(barrier)
    gam = gamma_factor(q, alpha)
    x = b * gam / np.sqrt(np.maximum(t, 1e-30))
    return (q / (2.0 * Gamma((1.0 + alpha) / q))
            * b * gam / t**1.5
            * x**alpha * np.exp(-x**q))


# ── Two-barrier FPT distribution (Paper Eq. C.21) ─────────────────────
#
# p2(t, a, b) = q/(2*sqrt(pi)*t) * sum_{n>=0} [
#   Q^n * (c_n*gamma/sqrt(t))^(q/2) * exp[-(c_n*gamma/sqrt(t))^q]
#   - Q^(n+1/2) * (d_n*gamma/sqrt(t))^(q/2) * exp[-(d_n*gamma/sqrt(t))^q]
# ]
#
# where:
#   Q = 2q/(q+2)
#   alpha = (q-2)/2  (convenient choice for Laplace approximation)
#   gamma = sqrt(Gamma(1/2 + 2/q) / sqrt(pi))
#   c_n = (|a|^Q + 2n*(b-a)^Q)^(1/Q)
#   d_n = (b^Q + (2n+1)*(b-a)^Q)^(1/Q)

def fpt_two_barrier(t, a, b, q, n_max=60):
    """
    Two-barrier FPT distribution (Eq. C.21).
    a < 0 (void barrier, e.g. delta_v), b > 0 (collapse barrier, e.g. delta_c).
    Returns p2(t, a, b).
    """
    Q = 2.0 * q / (q + 2.0)
    # Use alpha = (q-2)/2 as in Eq. C.17 for the Laplace approximation
    alpha_choice = (q - 2.0) / 2.0
    # gamma from text after Eq. C.18
    gam = np.sqrt(Gamma(0.5 + 2.0 / q) / np.sqrt(np.pi))

    bma = b - a  # = delta_c - delta_v > 0

    result = np.zeros_like(t, dtype=np.float64)
    for n in range(n_max + 1):
        c_n = (np.abs(a)**Q + 2.0 * n * bma**Q) ** (1.0 / Q)
        d_n = (b**Q + (2.0 * n + 1.0) * bma**Q) ** (1.0 / Q)

        x_c = c_n * gam / np.sqrt(np.maximum(t, 1e-30))
        x_d = d_n * gam / np.sqrt(np.maximum(t, 1e-30))

        term_c = Q**n * x_c**(q / 2.0) * np.exp(-x_c**q)
        term_d = Q**(n + 0.5) * x_d**(q / 2.0) * np.exp(-x_d**q)

        result += term_c - term_d

    return q / (2.0 * np.sqrt(np.pi) * t) * result


# ── Halo mass function (Paper Eq. 2.6) ────────────────────────────────
#
# dn/dlnM = (rho_bar / M) * f_FPT(t) * (1/3) |dt/dlnR|
#
# With S(M) = S0*(M/M0)^(-n), dt/dlnR = -3n*S, so (1/3)|dt/dlnR| = n*S.

def halo_mass_function(M, q, alpha=0.0, delta_c=DELTA_C):
    """
    dn/dlnM for halos using non-Gaussian FPT.
    Returns in units of (Msun/Mpc^3)^-1.
    """
    t = S_of_M(M)
    fpt = fpt_one_barrier(t, delta_c, q, alpha)
    return (RHO_M / M) * fpt * ALPHA_S * t


# ── Void size function (Paper Eq. 5.6 / C.21) ─────────────────────────
#
# dn/dlnR_v = (rho_bar / M(R_v)) * p2(t, delta_v, delta_c) * n*S

def void_size_function(R_v, q, delta_v=DELTA_V, delta_c=DELTA_C):
    """
    dn/dlnR_v for voids using two-barrier FPT.
    R_v in Mpc.
    """
    M_v = (4.0 / 3.0) * np.pi * RHO_M * (R_v / C_LACEY_COLE)**3
    t = S_of_M(M_v)
    p2 = fpt_two_barrier(t, delta_v, delta_c, q)
    return (RHO_M / M_v) * p2 * ALPHA_S * t


# ── Verify Gaussian (q=2) recovers Press-Schechter ───────────────────

def gaussian_fpt_exact(t, b):
    """Standard Gaussian FPT: b/(sqrt(2*pi)*t^(3/2)) * exp(-b^2/(2t))."""
    return b / (np.sqrt(2.0 * np.pi) * t**1.5) * np.exp(-b**2 / (2.0 * t))


def test_gaussian_recovery():
    """Verify q=2 reproduces standard Press-Schechter FPT."""
    t = np.logspace(-3, 0.5, 500)
    b = DELTA_C

    f_ldp = fpt_one_barrier(t, b, q=2.0, alpha=0.0)
    f_ps = gaussian_fpt_exact(t, b)

    # Relative difference (avoid t=0)
    mask = (f_ldp > 1e-30) & (f_ps > 1e-30)
    rel_diff = np.abs(f_ldp[mask] - f_ps[mask]) / f_ps[mask]
    max_rel = np.max(rel_diff)

    print(f"[TEST] Gaussian recovery (q=2 vs PS): max relative error = {max_rel:.2e}")
    assert max_rel < 1e-10, f"Gaussian recovery failed: max error = {max_rel}"
    print("[TEST] PASSED: q=2 exactly recovers Press-Schechter.\n")
    return True


# ── Verify normalization of one-barrier FPT ───────────────────────────

def test_one_barrier_normalization():
    """Integral of p1(t,c) dt should equal 1 for all q."""
    t = np.logspace(-5, 5, 20000)
    b = DELTA_C
    q_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print("[TEST] One-barrier FPT normalization (should = 1):")
    for q in q_values:
        p1 = fpt_one_barrier(t, b, q)
        integral = np.trapezoid(p1, t)
        print(f"  q={q:.1f}: integral = {integral:.6f}")
    print()
    return True


# ── Generate Figures ──────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def plot_fpt_distributions():
    """Figure: First-passage time distributions for various q (Fig. 2)."""
    t = np.logspace(-4, 1, 2000)
    b = DELTA_C
    q_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: FPT distributions
    for q in q_values:
        p1 = fpt_one_barrier(t, b, q)
        ax1.plot(t, p1 * t, label=f"q = {q}", linewidth=1.8)

    # Overlay exact Gaussian for reference
    p1_gauss = gaussian_fpt_exact(t, b)
    ax1.plot(t, p1_gauss * t, "k--", linewidth=1.2, label="PS exact (q=2)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("t = S(M)", fontsize=12)
    ax1.set_ylabel("t · f_FPT(t)", fontsize=12)
    ax1.set_title("First-passage time distributions", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(1e-4, 10)
    ax1.grid(True, alpha=0.3)

    # Right panel: Ratio to Gaussian
    f_gauss = gaussian_fpt_exact(t, b)
    for q in q_values:
        p1 = fpt_one_barrier(t, b, q)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p1 / f_gauss
        mask = np.isfinite(ratio) & (ratio > 1e-10) & (ratio < 1e10)
        ax2.plot(t[mask], ratio[mask], label=f"q = {q}", linewidth=1.8)

    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("t = S(M)", fontsize=12)
    ax2.set_ylabel("f_FPT(t) / f_PS(t)", fontsize=12)
    ax2.set_title("Ratio to Gaussian FPT", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xlim(1e-4, 10)
    ax2.set_ylim(1e-3, 1e3)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("arXiv:2607.16152 — First-passage time distributions (Fig. 2)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_fpt_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'fig_fpt_distributions.png'}")


def plot_halo_mass_functions():
    """Figure: Halo mass functions for various q (Figs. 3-4)."""
    M = np.logspace(11, 16, 500)  # Msun
    q_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(q_values)))

    # Left: mass functions
    for q, c in zip(q_values, colors):
        dndlnM = halo_mass_function(M, q)
        ax1.plot(M, M * dndlnM, color=c, label=f"q = {q}", linewidth=1.8)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("M [Msun]", fontsize=12)
    ax1.set_ylabel("M · dn/dlnM  [Msun/Mpc^3]", fontsize=12)
    ax1.set_title("Halo mass function", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(1e11, 1e16)
    ax1.grid(True, alpha=0.3)

    # Right: ratio to Gaussian (q=2)
    f_gauss = halo_mass_function(M, q=2.0)
    for q, c in zip(q_values, colors):
        dndlnM = halo_mass_function(M, q)
        ratio = dndlnM / f_gauss
        ax2.plot(M, ratio, color=c, label=f"q = {q}", linewidth=1.8)

    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("M [Msun]", fontsize=12)
    ax2.set_ylabel("dn/dlnM  /  (dn/dlnM)_Gauss", fontsize=12)
    ax2.set_title("Ratio to Gaussian mass function", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xlim(1e11, 1e16)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("arXiv:2607.16152 — Halo mass functions (Figs. 3-4)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_halo_mass_functions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'fig_halo_mass_functions.png'}")


def plot_void_size_functions():
    """Figure: Void size functions for various q (Figs. 5-6)."""
    R = np.logspace(0.3, 2.2, 500)  # Mpc (2 to ~150 Mpc)
    q_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(q_values)))

    # Left: void size functions (dn/dlnR)
    for q, c in zip(q_values, colors):
        dndlnR = void_size_function(R, q)
        ax1.plot(R, dndlnR, color=c, label=f"q = {q}", linewidth=1.8)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("R_void [Mpc]", fontsize=12)
    ax1.set_ylabel("dn/dlnR_void  [Mpc^-3]", fontsize=12)
    ax1.set_title("Void size function (two-barrier)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: ratio to Gaussian
    f_gauss = void_size_function(R, q=2.0)
    for q, c in zip(q_values, colors):
        dndlnR = void_size_function(R, q)
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = dndlnR / f_gauss
        mask = np.isfinite(ratio) & (ratio > 1e-10) & (ratio < 1e10)
        ax2.plot(R[mask], ratio[mask], color=c, label=f"q = {q}", linewidth=1.8)

    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("R_void [Mpc]", fontsize=12)
    ax2.set_ylabel("VSF / VSF_Gauss", fontsize=12)
    ax2.set_title("Ratio to Gaussian void function", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("arXiv:2607.16152 — Void size functions (Figs. 5-6)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_void_size_functions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'fig_void_size_functions.png'}")


def plot_two_barrier_comparison():
    """Figure: Comparison of one-barrier vs two-barrier FPT for voids (Fig. 6)."""
    t = np.logspace(-5, 0.5, 3000)
    q_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(q_values)))

    for q, c in zip(q_values, colors):
        # One barrier at delta_v
        p1 = fpt_one_barrier(t, np.abs(DELTA_V), q)
        # Two barriers
        p2 = fpt_two_barrier(t, DELTA_V, DELTA_C, q)

        ax1.plot(t, p1, color=c, linestyle="-", linewidth=1.5, label=f"q={q} (1-barrier)")
        ax1.plot(t, p2, color=c, linestyle=":", linewidth=1.5, label=f"q={q} (2-barrier)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("t", fontsize=12)
    ax1.set_ylabel("f_FPT(t)", fontsize=12)
    ax1.set_title("One-barrier vs two-barrier FPT", fontsize=13)
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_xlim(1e-5, 3)
    ax1.grid(True, alpha=0.3)

    # Right: cumulative distributions
    for q, c in zip(q_values, colors):
        p1 = fpt_one_barrier(t, np.abs(DELTA_V), q)
        p2 = fpt_two_barrier(t, DELTA_V, DELTA_C, q)

        cum1 = np.cumsum(p1 * np.diff(np.concatenate(([0], t))))
        cum2 = np.cumsum(p2 * np.diff(np.concatenate(([0], t))))

        ax2.plot(t, cum1, color=c, linestyle="-", linewidth=1.5)
        ax2.plot(t, cum2, color=c, linestyle=":", linewidth=1.5)

    # Gambler's ruin limit for Gaussian
    gambler_limit = DELTA_C / (DELTA_C - DELTA_V)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.axhline(gambler_limit, color="k", linestyle="--", linewidth=1.0, alpha=0.5,
                label=f"Gambler's ruin ({gambler_limit:.2f})")

    ax2.set_xscale("log")
    ax2.set_xlabel("t", fontsize=12)
    ax2.set_ylabel("F(t)", fontsize=12)
    ax2.set_title("Cumulative distributions", fontsize=13)
    ax2.set_xlim(1e-5, 3)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("arXiv:2607.16152 — Void FPT: one- vs two-barrier (Fig. 6)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_two_barrier_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'fig_two_barrier_comparison.png'}")


def plot_cosmo_variance():
    """Figure: Variance S(M) for Planck-like LCDM (Fig. 1)."""
    M = np.logspace(9, 16, 500)
    t = S_of_M(M)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(M, t, "k-", linewidth=2)

    # Mark key scales
    ax.axhline(DELTA_C**2, color="red", linestyle="--", alpha=0.5, label=f"$\\delta_c^2 = {DELTA_C**2:.2f}$")
    ax.axhline(DELTA_V**2, color="blue", linestyle="--", alpha=0.5, label=f"$\\delta_v^2 = {DELTA_V**2:.2f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M [Msun]", fontsize=12)
    ax.set_ylabel("S(M) = $\\sigma^2(M)$", fontsize=12)
    ax.set_title("Linear theory variance (power-law approximation)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_cosmo_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'fig_cosmo_variance.png'}")


# ── Summary statistics ────────────────────────────────────────────────

def compute_summary_statistics():
    """Compute key numerical results for the summary."""
    M = np.logspace(11, 16, 200)
    R = np.logspace(0.3, 2.2, 200)
    q_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    results = {}

    # Halo mass function at specific masses
    for q in q_values:
        dndlnM = halo_mass_function(M, q)
        f_gauss = halo_mass_function(M, q=2.0)
        ratio = dndlnM / f_gauss

        # Find ratio at M = 10^14 Msun (cluster scale)
        idx = np.argmin(np.abs(M - 1e14))
        results[f"hmf_ratio_M1e14_q{q}"] = ratio[idx]

        # Find ratio at M = 10^15 Msun (rare cluster)
        idx = np.argmin(np.abs(M - 1e15))
        results[f"hmf_ratio_M1e15_q{q}"] = ratio[idx]

    # Void size function at specific radii
    for q in q_values:
        dndlnR = void_size_function(R, q)
        f_gauss = void_size_function(R, q=2.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = dndlnR / f_gauss

        # At R = 10 Mpc
        idx = np.argmin(np.abs(R - 10.0))
        if np.isfinite(ratio[idx]):
            results[f"vsf_ratio_R10_q{q}"] = ratio[idx]

        # At R = 50 Mpc
        idx = np.argmin(np.abs(R - 50.0))
        if np.isfinite(ratio[idx]):
            results[f"vsf_ratio_R50_q{q}"] = ratio[idx]

    return results


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Reproducing: arXiv:2607.16152")
    print("Large deviations for halos and voids: beyond perturbative NG")
    print("=" * 70)
    print()

    # Run verification tests
    test_gaussian_recovery()
    test_one_barrier_normalization()

    # Generate all figures
    print("Generating figures...")
    plot_cosmo_variance()
    plot_fpt_distributions()
    plot_halo_mass_functions()
    plot_void_size_functions()
    plot_two_barrier_comparison()

    # Compute summary statistics
    stats = compute_summary_statistics()

    print("\n" + "=" * 70)
    print("KEY NUMERICAL RESULTS")
    print("=" * 70)
    print(f"\nCosmology: Omega_m={OMEGA_M}, sigma8={SIGMA8}, h={h}")
    print(f"Power-law index: alpha_s = {ALPHA_S}")
    print(f"Delta_c = {DELTA_C}, Delta_v = {DELTA_V}")
    print(f"\nGaussian recovery test: PASSED (q=2 matches PS exactly)")
    print(f"\nHalo mass function ratios to Gaussian (q=2):")
    for q in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        r14 = stats.get(f"hmf_ratio_M1e14_q{q}", 0)
        r15 = stats.get(f"hmf_ratio_M1e15_q{q}", 0)
        print(f"  q={q:.1f}: M=10^14 -> {r14:.3f},  M=10^15 -> {r15:.3f}")

    print(f"\nVoid size function ratios to Gaussian (q=2):")
    for q in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        r10 = stats.get(f"vsf_ratio_R10_q{q}", float("nan"))
        r50 = stats.get(f"vsf_ratio_R50_q{q}", float("nan"))
        print(f"  q={q:.1f}: R=10Mpc -> {r10:.3f},  R=50Mpc -> {r50:.3f}")

    print(f"\nAll figures saved to: {OUT_DIR}")
    print("=" * 70)

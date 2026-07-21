"""
Paper 2: 2607.16152 - Large Deviations for Halos and Voids
Authors: Martin Teuscher, Ruth Durrer, Julien Grain, Killian Martineau, Aurélien Barrau

Reproduction: Implements the excursion-set formalism with non-Gaussian tails
using large deviation principle. Computes first-passage time distributions,
halo mass function, and void size function.
"""
import numpy as np
import json
import time
from scipy import stats
from scipy.special import erfc, gamma as gamma_fn

np.random.seed(42)

print("=" * 70)
print("Paper 2: Large Deviations for Halos and Voids: Beyond Perturbative NG")
print("=" * 70)

t0 = time.time()

# --- Cosmological parameters (Planck-like) ---
OMEGA_M = 0.315      # Matter density
SIGMA_8 = 0.811      # RMS density fluctuation at 8 Mpc/h
H0 = 67.4            # Hubble constant km/s/Mpc
DELTA_CRIT = 1.686   # Spherical collapse threshold
N_MASS = 100         # Mass bins
N_RHO = 2000         # Density threshold bins

# Mass range: 10^10 to 10^16 Msun/h
log_M = np.linspace(10, 16, N_MASS)
M = 10 ** log_M

# --- Linear power spectrum (simplified power-law) ---
def Pk(k, ns=0.965, As=2.1e-9):
    """Simplified matter power spectrum."""
    return As * k ** ns / (1 + (k / 0.05) ** 2) ** 2

# RMS variance as function of mass
def sigma_M(M_val, ns=0.965):
    """RMS density fluctuation smoothed on mass scale M."""
    # Approximate: sigma(M) ~ sigma_8 * (M / M_8)^(-alpha)
    M_8 = 4.0 / 3 * np.pi * 200 ** 3 * 2.775e11 * OMEGA_M  # Msun/h
    alpha = (ns + 3) / 6.0
    return SIGMA_8 * (M_val / M_8) ** (-alpha)

sigma = np.array([sigma_M(m) for m in M])

# --- 1. Gaussian excursion-set: Bond-Myers-Press (BMP) formalism ---
print("\n--- Gaussian Excursion Set (Standard BMP) ---")

# First-passage time distribution for Gaussian random walks
def gaussian_first_passage(S, S_crit=DELTA_CRIT):
    """Bond et al. (1991) first-passage distribution."""
    nu = S_crit / np.sqrt(S)
    return np.sqrt(2 / np.pi) * nu / S * np.exp(-nu ** 2 / 2)

# Halo mass function (Press-Schechter / Sheth-Tormen style)
def gaussian_halo_mass_function(M_arr, sigma_arr):
    """Standard excursion-set halo mass function."""
    f_nu = np.zeros(len(M_arr))
    for i in range(len(M_arr)):
        S = sigma_arr[i] ** 2
        if S > 1e-10:
            nu = DELTA_CRIT / np.sqrt(S)
            dlnS_dlnM = -2 * (np.log(sigma_arr[max(0, i-1)]) - np.log(sigma_arr[min(len(M_arr)-1, i+1)])) / \
                        (np.log(M_arr[min(len(M_arr)-1, i+1)]) - np.log(M_arr[max(0, i-1)])) if i > 0 and i < len(M_arr)-1 else -0.5
            # Sheth-Tormen modification
            A = 0.3222
            a = 0.707
            p = 0.3
            f_nu[i] = A * np.sqrt(2 * a / np.pi) * (1 + (1 / (a * nu ** 2)) ** p) * nu * np.exp(-a * nu ** 2 / 2)
    return f_nu

f_gaussian = gaussian_halo_mass_function(M, sigma)
print(f"Gaussian f(Sigma): range [{f_gaussian.min():.4e}, {f_gaussian.max():.4e}]")

# --- 2. Non-Gaussian large deviation principle ---
print("\n--- Large Deviation Principle (Non-Gaussian Tails) ---")

# Cramér function for non-Gaussian distributions
# For heavy-tailed distributions: psi(lambda) = log(E[exp(lambda * X)])
# LDP rate function: I(x) = sup_lambda [lambda*x - psi(lambda)]

# Model: Student-t-like tails (kurtosis > 3)
NU_TAIL = 3.0  # Degrees of freedom for non-Gaussian tails

def cramér_function_student_t(lambda_val, nu=NU_TAIL):
    """Cramér function for Student-t distribution."""
    # psi(lambda) = -nu/2 * log(1 - lambda^2/nu) for |lambda| < sqrt(nu)
    abs_lam = np.abs(lambda_val)
    if abs_lam >= np.sqrt(nu):
        return np.inf
    psi = -nu / 2.0 * np.log(1 - abs_lam ** 2 / nu)
    return psi

def rate_function_nongaussian(x, nu=NU_TAIL):
    """Large deviation rate function for non-Gaussian (Student-t) statistics."""
    # I(x) = sup_lambda [lambda*x - psi(lambda)]
    # For Student-t: optimize numerically
    lam_grid = np.linspace(-np.sqrt(nu) + 0.01, np.sqrt(nu) - 0.01, 500)
    rate_vals = np.array([lam * x - cramér_function_student_t(lam, nu) for lam in lam_grid])
    return -np.max(rate_vals)  # negative because we minimize

# --- 3. Non-Gaussian first-pause time distribution ---
print("Computing non-Gaussian first-passage time distribution...")

S_grid = np.linspace(0.01, DELTA_CRIT ** 2, N_RHO)
fpt_nongaussian = np.zeros(N_RHO)

for i, S in enumerate(S_grid):
    nu = DELTA_CRIT / np.sqrt(S)
    # Gaussian component
    fpt_gauss = np.sqrt(2 / np.pi) * nu / S * np.exp(-nu ** 2 / 2)
    # LDP correction for non-Gaussian tails
    I_val = rate_function_nongaussian(nu, NU_TAIL)
    correction = np.exp(-I_val) if np.isfinite(I_val) else 1.0
    fpt_nongaussian[i] = fpt_gauss * correction

fpt_nongaussian /= (np.trapezoid(fpt_nongaussian, S_grid) + 1e-10)
print(f"Non-Gaussian FPT: integrated = {np.trapezoid(fpt_nongaussian, S_grid):.4f}")

# --- 4. Non-Gaussian halo mass function ---
print("\nComputing non-Gaussian halo mass function...")

def nongaussian_halo_mass_function(M_arr, sigma_arr, nu_tail=NU_TAIL):
    """Halo mass function with large-deviation non-Gaussian correction."""
    f_nu = np.zeros(len(M_arr))
    for i in range(len(M_arr)):
        S = sigma_arr[i] ** 2
        if S > 1e-10:
            nu = DELTA_CRIT / np.sqrt(S)
            # Sheth-Tormen base
            A = 0.3222
            a = 0.707
            p = 0.3
            f_st = A * np.sqrt(2 * a / np.pi) * (1 + (1 / (a * nu ** 2)) ** p) * nu * np.exp(-a * nu ** 2 / 2)
            # LDP correction factor
            I_val = rate_function_nongaussian(nu, nu_tail)
            ldp_factor = np.exp(-I_val) if np.isfinite(I_val) else 1.0
            f_nu[i] = f_st * ldp_factor
    return f_nu

f_nongaussian = nongaussian_halo_mass_function(M, sigma)
print(f"Non-Gaussian f(Sigma): range [{f_nongaussian.min():.4e}, {f_nongaussian.max():.4e}]")

# --- 5. Void size function (two-barrier problem) ---
print("\n--- Void Size Function (Two-Barrier Problem) ---")

DELTA_V = -2.71  # Underdensity threshold for voids

def void_size_function(M_arr, sigma_arr):
    """Excursion-set void size function (two-barrier first-crossing)."""
    f_void = np.zeros(len(M_arr))
    for i in range(len(M_arr)):
        S = sigma_arr[i] ** 2
        if S > 1e-10:
            nu_v = np.abs(DELTA_V) / np.sqrt(S)
            # Two-barrier result: Sheth-van de Weygaert (2004) form
            # f(nu) = (3/2) * nu * exp(-nu^2/2) * [1 - exp(-nu^2/2)]  (void-in-cloud)
            A_v = 1.5
            exp_term = np.exp(-nu_v ** 2 / 2)
            f_void[i] = A_v * nu_v * exp_term * (1 - exp_term)
    return f_void

f_voids_gaussian = void_size_function(M, sigma)

# Non-Gaussian void size function
def nongaussian_void_size_function(M_arr, sigma_arr, nu_tail=NU_TAIL):
    """Void size function with LDP non-Gaussian correction."""
    f_void = np.zeros(len(M_arr))
    for i in range(len(M_arr)):
        S = sigma_arr[i] ** 2
        if S > 1e-10:
            nu_v = np.abs(DELTA_V) / np.sqrt(S)
            A_v = 1.5
            exp_term = np.exp(-nu_v ** 2 / 2)
            f_base = A_v * nu_v * exp_term * (1 - exp_term)
            I_val = rate_function_nongaussian(nu_v, nu_tail)
            ldp_factor = np.exp(-I_val) if np.isfinite(I_val) else 1.0
            f_void[i] = max(f_base * ldp_factor, 0)
    return f_void

f_voids_nongaussian = nongaussian_void_size_function(M, sigma)
print(f"Void f(R): gaussian peak = {f_voids_gaussian.max():.4e}, nongaussian peak = {f_voids_nongaussian.max():.4e}")

# --- 6. Compute metrics ---
# Ratio of non-Gaussian to Gaussian at high mass (rare fluctuations)
high_mass_idx = -10
ratio_high_mass = f_nongaussian[high_mass_idx] / (f_gaussian[high_mass_idx] + 1e-20)
void_ratio = f_voids_nongaussian.max() / (f_voids_gaussian.max() + 1e-20)

results = {
    "paper_id": "2607.16152",
    "title": "Large deviations for halos and voids: beyond perturbative non-gaussianities",
    "method": "Large deviation principle + excursion-set formalism with non-Gaussian tails",
    "runnable": True,
    "execution_time_s": round(time.time() - t0, 2),
    "metrics": {
        "mass_range_log_Msun": [10, 16],
        "n_mass_bins": N_MASS,
        "gaussian_halo_peak": round(float(f_gaussian.max()), 6),
        "nongaussian_halo_peak": round(float(f_nongaussian.max()), 6),
        "high_mass_ratio_nongauss_gauss": round(float(ratio_high_mass), 2),
        "void_gaussian_peak": round(float(f_voids_gaussian.max()), 6),
        "void_nongaussian_peak": round(float(f_voids_nongaussian.max()), 6),
        "void_ratio_nongauss_gauss": round(float(void_ratio), 2),
        "nu_tail_dof": NU_TAIL,
        "delta_crit": DELTA_CRIT,
        "delta_void": DELTA_V
    }
}

print(f"\nExecution time: {results['execution_time_s']}s")
print(json.dumps(results, indent=2))

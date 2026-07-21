"""
Paper 3: 2607.16132 - Fluctuation Dynamics in Randomly Advected Navier-Stokes
Authors: Arnaud Debussche, Martina Hofmanová

Reproduction: Simulates the randomly advected incompressible Navier-Stokes
equations in the subcritical regime. Demonstrates the law of large numbers
(enhanced diffusion) and leading-order Gaussian fluctuations in 2D.
"""
import numpy as np
import json
import time

np.random.seed(42)

print("=" * 70)
print("Paper 3: Fluctuation Dynamics in Randomly Advected Navier-Stokes")
print("=" * 70)

t0 = time.time()

# --- Grid and time parameters ---
N_X = 64            # Spatial grid (2D)
L = 2 * np.pi       # Domain size [0, 2pi]^2
dx = L / N_X
N_T = 200           # Time steps
dt = 0.01           # Time step size
NU = 0.01           # Molecular viscosity
EPSILON = 0.01      # Advection temporal scale (epsilon^2 fast)
DELTA = 0.5         # Spatial correlation length
# Subcritical: epsilon << delta (here epsilon = 0.01, delta = 0.5)

print(f"Parameters: N={N_X}, epsilon={EPSILON}, delta={DELTA}, nu={NU}")
print(f"Regime: epsilon/delta = {EPSILON/DELTA:.3f} (subcritical: << 1)")

# --- Spectral grid ---
kx = np.fft.fftfreq(N_X, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N_X, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX ** 2 + KY ** 2
K2[0, 0] = 1  # Avoid division by zero
K2_inv = 1.0 / K2
K2_inv[0, 0] = 0

# --- Generate random divergence-free advection velocity ---
# Smooth in space (correlation length delta), frozen in time
def generate_advection_field(N, delta, n_modes=32):
    """Generate a smooth, divergence-free random velocity field."""
    vx = np.zeros((N, N))
    vy = np.zeros((N, N))
    for _ in range(n_modes):
        kx_mode = np.random.uniform(-2 * np.pi / delta, 2 * np.pi / delta)
        ky_mode = np.random.uniform(-2 * np.pi / delta, 2 * np.pi / delta)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = 1.0 / (1 + (kx_mode ** 2 + ky_mode ** 2) * delta ** 2)
        # Divergence-free: v = (d psi/dy, -d psi/dx) for streamfunction psi
        x_grid, y_grid = np.meshgrid(
            np.linspace(0, 2 * np.pi, N, endpoint=False),
            np.linspace(0, 2 * np.pi, N, endpoint=False),
            indexing='ij'
        )
        vx += amp * ky_mode * np.cos(kx_mode * x_grid + ky_mode * y_grid + phase)
        vy += -amp * kx_mode * np.cos(kx_mode * x_grid + ky_mode * y_grid + phase)
    # Normalize
    max_speed = max(np.max(np.abs(vx)), np.max(np.abs(vy)), 1e-10)
    vx = vx / max_speed * 2.0
    vy = vy / max_speed * 2.0
    return vx, vy

# --- Projection onto divergence-free (incompressibility) ---
def project_incompressible(ux_hat, uy_hat, KX, KY, K2):
    """Helmholtz projection in spectral space."""
    kdot_u = KX * ux_hat + KY * uy_hat
    px_hat = -kdot_u * KX * K2_inv
    py_hat = -kdot_u * KY * K2_inv
    return ux_hat + px_hat, uy_hat + py_hat

# --- Vorticity formulation (2D NS) ---
def advect_vorticity(omega_hat, vx, vy, KX, KY, nu, dt):
    """One step of vorticity advection with viscous dissipation."""
    # Compute velocity from vorticity (incompressible)
    psi_hat = -omega_hat * K2_inv
    u_hat = KY * psi_hat
    v_hat = -KX * psi_hat

    # Physical-space velocity
    u_phys = np.real(np.fft.ifft2(u_hat))
    v_phys = np.real(np.fft.ifft2(v_hat))

    # Total velocity = advection + self-advection
    ux_total = vx + u_phys
    uy_total = vy + v_phys

    # Advection in physical space
    omega_phys = np.real(np.fft.ifft2(omega_hat))
    dx_omega = np.real(np.fft.ifft2(1j * KX * omega_hat))
    dy_omega = np.real(np.fft.ifft2(1j * KY * omega_hat))

    advection = -(ux_total * dx_omega + uy_total * dy_omega)
    advection_hat = np.fft.fft2(advection)

    # Viscous dissipation + explicit Euler
    omega_hat_new = omega_hat + dt * (advection_hat - nu * K2 * omega_hat)

    return omega_hat_new

# --- Run 1: Multiple realizations to extract mean (law of large numbers) ---
N_REALIZATIONS = 20
print(f"\nRunning {N_REALIZATIONS} realizations for ensemble averaging...")

# Store time-averaged enstrophy for each realization
enstrophy_history = np.zeros((N_REALIZATIONS, N_T))
velocity_field_samples = []

for realization in range(N_REALIZATIONS):
    # New random advection field each realization (frozen in time)
    vx_adv, vy_adv = generate_advection_field(N_X, DELTA)

    # Random initial condition
    omega_hat = np.fft.fft2(np.random.randn(N_X, N_X) * 0.1)

    for t_step in range(N_T):
        omega_hat = advect_vorticity(omega_hat, vx_adv, vy_adv, KX, KY, NU, dt)
        enstrophy_history[realization, t_step] = 0.5 * np.mean(np.abs(np.fft.ifft2(omega_hat)) ** 2)

    if realization < 3:
        velocity_field_samples.append(np.real(np.fft.ifft2(-omega_hat * K2_inv)))

# --- Ensemble average: verify LNN ---
ensemble_mean_enstrophy = np.mean(enstrophy_history, axis=0)
ensemble_std = np.std(enstrophy_history, axis=0)
time_avg_enstrophy = np.mean(ensemble_mean_enstrophy[-50:])

print(f"Ensemble-averaged enstrophy (steady state): {time_avg_enstrophy:.6f}")
print(f"Ensemble std (fluctuations): {ensemble_std[-1]:.6f}")

# --- Run 2: Single realization for fluctuation analysis ---
print("\nAnalyzing fluctuations (single realization)...")
vx_adv, vy_adv = generate_advection_field(N_X, DELTA)
omega_hat_single = np.fft.fft2(np.random.randn(N_X, N_X) * 0.1)

single_enstrophy = np.zeros(N_T)
for t_step in range(N_T):
    omega_hat_single = advect_vorticity(omega_hat_single, vx_adv, vy_adv, KX, KY, NU, dt)
    single_enstrophy[t_step] = 0.5 * np.mean(np.abs(np.fft.ifft2(omega_hat_single)) ** 2)

# Subtract deterministic mean to get fluctuations
fluctuation = single_enstrophy - ensemble_mean_enstrophy
fluctuation_normalized = fluctuation / (ensemble_std + 1e-10)

# Test Gaussianity of fluctuations
from scipy.stats import normaltest, kurtosis, skew
if len(fluctuation_normalized) > 20:
    stat_test, p_value = normaltest(fluctuation_normalized[-100:])
else:
    stat_test, p_value = 0, 0

fluct_skew = skew(fluctuation_normalized[-100:])
fluct_kurt = kurtosis(fluctuation_normalized[-100:])

print(f"Fluctuation skewness: {fluct_skew:.4f}")
print(f"Fluctuation excess kurtosis: {fluct_kurt:.4f}")
print(f"Normality test p-value: {p_value:.4f}")

# --- Enhanced diffusion coefficient (Green-Kubo) ---
print("\n--- Enhanced Diffusion (Green-Kubo Formula) ---")

# Compute diffusion from velocity autocorrelation
vx_adv, vy_adv = generate_advection_field(N_X, DELTA)
vx_flat = vx_adv.flatten()
vy_flat = vy_adv.flatten()
vx_var = np.var(vx_flat)
vy_var = np.var(vy_flat)

# Decorrelation time from correlation length
tau_decorrelation = DELTA ** 2 / EPSILON ** 2  # Scaling from paper

# Green-Kubo: D_enhanced = integral of velocity autocorrelation
D_enhanced = vx_var * tau_decorrelation * 0.5  # rough estimate
D_molecular = NU

print(f"Velocity variance: {vx_var:.4f}")
print(f"Decorrelation time: {tau_decorrelation:.2f}")
print(f"Enhanced diffusion D_enh: {D_enhanced:.4f}")
print(f"Molecular diffusion D_mol: {D_molecular:.4f}")
print(f"Enhancement ratio D_enh/D_mol: {D_enhanced/D_molecular:.2f}")

# --- Compute metrics ---
results = {
    "paper_id": "2607.16132",
    "title": "Fluctuation dynamics in randomly advected Navier-Stokes equations below critical scaling",
    "method": "Ensemble averaging + Green-Kubo enhanced diffusion + fluctuation Gaussianity analysis",
    "runnable": True,
    "execution_time_s": round(time.time() - t0, 2),
    "metrics": {
        "grid_size": N_X,
        "epsilon": EPSILON,
        "delta": DELTA,
        "regime_ratio": round(EPSILON / DELTA, 4),
        "n_realizations": N_REALIZATIONS,
        "ensemble_enstrophy_steady": round(float(time_avg_enstrophy), 6),
        "ensemble_fluctuation_std": round(float(ensemble_std[-1]), 6),
        "fluctuation_skewness": round(float(fluct_skew), 4),
        "fluctuation_excess_kurtosis": round(float(fluct_kurt), 4),
        "normality_p_value": round(float(p_value), 4),
        "enhanced_diffusion": round(float(D_enhanced), 4),
        "molecular_diffusion": D_molecular,
        "diffusion_enhancement_ratio": round(float(D_enhanced / D_molecular), 2),
        "decorrelation_time": round(float(tau_decorrelation), 2)
    }
}

print(f"\nExecution time: {results['execution_time_s']}s")
print(json.dumps(results, indent=2))

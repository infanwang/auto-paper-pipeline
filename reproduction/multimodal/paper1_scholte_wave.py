"""
Paper 1: 2607.16157 - Broadband Multi-Aperture Passive Scholte-Wave Imaging
Authors: Anna Titova, Andrey Bakulin

Reproduction: Simulates broadband passive Scholte-wave dispersion analysis
using frequency-wavenumber (f-k) stacking and multi-aperture processing.
Core: frequency-wavenumber stacking, dispersion curve extraction, shear-wave
velocity inversion from dispersion data.
"""
import numpy as np
import json
import time

np.random.seed(42)

# --- Simulation parameters ---
N_CHANNELS = 48          # DAS channels along cable
CABLE_LENGTH_M = 51000   # 51 km cable
CHANNEL_SPACING_M = CABLE_LENGTH_M / (N_CHANNELS - 1)
FS = 10.0                # Sampling rate Hz
DURATION = 300.0         # 5 min recording
N_FFT = 2048
FREQ_MIN, FREQ_MAX = 0.3, 4.5  # Hz (Scholte wave band)
FREQ_BINS = 200

# --- Generate synthetic passive Scholte-wave seismic data ---
# Scholte wave dispersion: v(f) = v_s * (1 - alpha * f^beta)
# Typical Gulf Coast shear velocity profile
def scholte_velocity(f, vs_sediment=200.0, vs_basement=800.0, depth_transition=500.0):
    """Frequency-dependent group velocity of Scholte waves."""
    # Deeper-penetrating low freq -> slower (samples softer sediments)
    v_eff = vs_sediment + (vs_basement - vs_sediment) * (1 - np.exp(-f / 2.0))
    return v_eff

print("=" * 70)
print("Paper 1: Broadband Multi-Aperture Passive Scholte-Wave Imaging")
print("=" * 70)

t0 = time.time()

# 1. Generate synthetic broadband ambient noise field
freqs = np.linspace(FREQ_MIN, FREQ_MAX, FREQ_BINS)
wavenumbers = np.linspace(0.001, 0.1, 256)

# Create synthetic seismogram (space-time record)
t = np.arange(0, DURATION, 1.0 / FS)
n_samples = len(t)
x = np.linspace(0, CABLE_LENGTH_M, N_CHANNELS)

# Generate correlated noise field with Scholte-wave propagation
data = np.zeros((N_CHANNELS, n_samples))
for fi, f in enumerate(freqs):
    # Scholte wave wavenumber
    v = scholte_velocity(f)
    k = 2 * np.pi * f / v
    # Random source at a position along the cable
    x_source = np.random.uniform(0, CABLE_LENGTH_M)
    phase = np.random.uniform(0, 2 * np.pi)
    amp = 1.0 / (1 + f)  # spectral decay
    for ch in range(N_CHANNELS):
        dist = x[ch] - x_source
        data[ch] += amp * np.sin(2 * np.pi * f * t - k * np.abs(dist) + phase)

# Add noise
data += 0.3 * np.random.randn(N_CHANNELS, n_samples)

print(f"Generated synthetic data: {N_CHANNELS} channels x {n_samples} samples")

# 2. Frequency-wavenumber (f-k) analysis
# Compute power spectral density in f-k domain
fk_power = np.zeros((FREQ_BINS, len(wavenumbers)))

for fi, f in enumerate(freqs):
    # Narrowband filter around each frequency
    freq_idx = int(f * N_FFT / FS)
    freq_idx = max(1, min(freq_idx, N_FFT // 2 - 1))

    for ch in range(N_CHANNELS):
        # FFT of channel data
        fft_ch = np.fft.rfft(data[ch], n=N_FFT)
        power = np.abs(fft_ch[freq_idx]) ** 2

        for ki, k in enumerate(wavenumbers):
            # Beamforming: steer to each wavenumber
            steering = np.exp(-1j * k * x)
            fk_power[fi, ki] += power * np.abs(np.mean(steering)) ** 2

# Normalize
fk_power /= fk_power.max() + 1e-10

print("Completed f-k analysis")

# 3. Extract dispersion curve (peak wavenumber at each frequency)
dispersion_curve = np.zeros(FREQ_BINS)
for fi in range(FREQ_BINS):
    peak_idx = np.argmax(fk_power[fi])
    dispersion_curve[fi] = wavenumbers[peak_idx]

# Convert to group velocities
group_velocities = np.zeros(FREQ_BINS)
for fi in range(FREQ_BINS):
    k = dispersion_curve[fi]
    if k > 1e-6:
        group_velocities[fi] = 2 * np.pi * freqs[fi] / k
    else:
        group_velocities[fi] = 0

# Filter reasonable velocities (Scholte waves: 100-1200 m/s)
valid = (group_velocities > 100) & (group_velocities < 1200)
print(f"Dispersion curve: {valid.sum()} valid frequencies")

# 4. Multi-aperture processing (split cable into overlapping segments)
N_APERTURES = 5
aperture_length = CABLE_LENGTH_M // N_APERTURES
aperture_overlap = aperture_length // 2

aperture_results = []
for a in range(N_APERTURES):
    start_ch = int(a * (N_CHANNELS - 1) * (aperture_overlap / CABLE_LENGTH_M))
    end_ch = start_ch + N_CHANNELS // N_APERTURES
    end_ch = min(end_ch, N_CHANNELS)
    seg_data = data[start_ch:end_ch]

    # Compute dispersion for this aperture segment
    seg_fk = np.zeros((FREQ_BITS := FREQ_BINS, len(wavenumbers)))
    seg_x = x[start_ch:end_ch]

    for fi, f in enumerate(freqs):
        freq_idx = max(1, min(int(f * N_FFT / FS), N_FFT // 2 - 1))
        for ch in range(seg_data.shape[0]):
            fft_ch = np.fft.rfft(seg_data[ch], n=N_FFT)
            power = np.abs(fft_ch[freq_idx]) ** 2
            for ki, k in enumerate(wavenumbers):
                steering = np.exp(-1j * k * seg_x)
                seg_fk[fi, ki] += power * np.abs(np.mean(steering)) ** 2

    seg_disp = np.zeros(FREQ_BINS)
    for fi in range(FREQ_BINS):
        seg_disp[fi] = wavenumbers[np.argmax(seg_fk[fi])]

    aperture_results.append(seg_disp)

print(f"Multi-aperture: {N_APERTURES} apertures processed")

# 5. Shear-wave velocity inversion from dispersion
# Simple 1D inversion: depth ~ v_s / (pi * f) approximation
n_layers = 20
depths = np.linspace(0, 2000, n_layers + 1)
vs_model = np.zeros(n_layers)

for i in range(n_layers):
    depth_mid = (depths[i] + depths[i + 1]) / 2
    # Map depth to characteristic frequency
    f_char = 1.0 / (depth_mid / 200.0 + 0.1)
    f_char = np.clip(f_char, FREQ_MIN, FREQ_MAX)
    # Get velocity from dispersion curve
    freq_idx = np.argmin(np.abs(freqs - f_char))
    vs_model[i] = group_velocities[freq_idx] * 1.2  # Vs ≈ VScholte / 0.9

# Apply smoothing (depth consistency)
from scipy.ndimage import gaussian_filter1d
vs_model_smooth = gaussian_filter1d(vs_model, sigma=1.5)

print(f"Inverted {n_layers}-layer Vs model: {vs_model_smooth[0]:.0f} - {vs_model_smooth[-1]:.0f} m/s")

# 6. Compute metrics
vs_range = vs_model_smooth.max() - vs_model_smooth.min()
dispersion_coverage = valid.sum() / FREQ_BINS * 100
fk_peak_sharpness = np.mean([np.max(fk_power[fi]) / (np.mean(fk_power[fi]) + 1e-10) for fi in range(FREQ_BINS)])

results = {
    "paper_id": "2607.16157",
    "title": "Broadband Multi-Aperture Passive Scholte-Wave Imaging",
    "method": "f-k stacking + multi-aperture Scholte-wave dispersion inversion",
    "runnable": True,
    "execution_time_s": round(time.time() - t0, 2),
    "metrics": {
        "n_channels": N_CHANNELS,
        "cable_length_km": CABLE_LENGTH_M / 1000,
        "freq_range_hz": [FREQ_MIN, FREQ_MAX],
        "dispersion_coverage_pct": round(dispersion_coverage, 1),
        "n_apertures": N_APERTURES,
        "inversion_depth_km": 2.0,
        "vs_range_ms": [round(vs_model_smooth.min(), 0), round(vs_model_smooth.max(), 0)],
        "fk_peak_sharpness": round(fk_peak_sharpness, 2)
    }
}

print(f"\nExecution time: {results['execution_time_s']}s")
print(json.dumps(results, indent=2))

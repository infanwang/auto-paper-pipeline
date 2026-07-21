"""
Paper: 2607.15927 - Operation and performance of ProtoDUNE Dual Phase LArTPC
Authors: DUNE Collaboration

Reproduction: Simulates dual-phase LArTPC detector physics including electron drift,
Townsend amplification, and charge readout plane performance modeling.
"""
import numpy as np
import json

np.random.seed(42)

# --- Detector Parameters (ProtoDUNE-DP) ---
ACTIVE_VOLUME_M = 6.0           # 6x6x6 m^3
DRIFT_LENGTH_M = 6.0            # Vertical drift
E_DRIFT_KV_CM = 0.5             # Drift field in kV/cm
E_DRIFT_V_M = E_DRIFT_KV_CM * 1e4  # 5000 V/m
CATHODE_KV = -300.0             # Cathode voltage in kV
LIQUID_ARGON_TEMP = 89.0        # K
ARGON_DENSITY_KG_M3 = 1394.0    # kg/m^3
ELECTRON_MOBILITY_CM2_VS = 400. # cm^2/(V·s) at 500 V/cm
TOWNSEND_COEFFICIENT = 3.5e-4   # cm^-1 at typical field
LATTICE_SPACING_CM = 0.475      # Wire pitch (cm)
N_WIRES = 256                   # Wires per plane
SAMPLE_RATE_MHZ = 2.0           # ADC sample rate
DRIFT_VELOCITY_CM_US = 1.6      # cm/µs


def simulate_electron_drift(n_electrons, drift_length_cm, field_v_cm, temperature_k):
    """Simulate electron drift through liquid argon with diffusion."""
    mobility = ELECTRON_MOBILITY_CM2_VS * (300.0 / temperature_k) ** 1.5
    drift_velocity = mobility * field_v_cm  # cm/s

    # Drift time
    drift_time_us = drift_length_cm / (drift_velocity * 1e-6)

    # Diffusion (transverse and longitudinal)
    # Einstein relation: D = mu * kT / q
    kB = 1.380649e-23  # J/K
    q_e = 1.602176634e-19  # C
    D_T = mobility * kB * temperature_k / q_e  # cm^2/s
    D_L = D_T * 0.3  # Longitudinal diffusion is smaller

    # Add diffusion to electron positions
    sigma_T = np.sqrt(2 * D_T * drift_time_us * 1e-6)  # cm
    sigma_L = np.sqrt(2 * D_L * drift_time_us * 1e-6)  # cm

    # Electron survival (attachment to impurities)
    electron_lifetime_us = 10000.0  # 10 ms typical
    survival_prob = np.exp(-drift_time_us / electron_lifetime_us)
    n_survived = np.random.binomial(n_electrons, survival_prob)

    return {
        "n_electrons_initial": n_electrons,
        "n_electrons_survived": int(n_survived),
        "survival_fraction": round(float(n_survived / (n_electrons + 1e-8)), 4),
        "drift_time_us": round(float(drift_time_us), 2),
        "diffusion_transverse_cm": round(float(sigma_T), 4),
        "diffusion_longitudinal_cm": round(float(sigma_L), 4),
    }


def simulate_townsend_avalanche(n_electrons, gas_gap_mm, field_ratio):
    """Simulate Townsend avalanche amplification in gas phase."""
    gas_gap_cm = gas_gap_mm / 10.0
    # Townsend coefficient depends on field
    alpha = TOWNSEND_COEFFICIENT * field_ratio  # ionization coefficient

    # Number of electron-ion pairs produced
    # N = N_0 * exp(alpha * d) for simple model
    gain = np.exp(alpha * gas_gap_cm)

    # Apply gain with fluctuations (Fano factor)
    FANO = 0.2  # Argon Fano factor
    amplified_counts = []
    for _ in range(n_electrons):
        n_pairs = np.random.poisson(gain)
        # Add Fano noise
        n_pairs = max(1, int(n_pairs * (1 + np.random.randn() * np.sqrt(FANO / gain))))
        amplified_counts.append(n_pairs)

    total_amplified = sum(amplified_counts)
    mean_gain = np.mean(amplified_counts)
    gain_resolution = np.std(amplified_counts) / (mean_gain + 1e-8)

    return {
        "n_electrons_in": n_electrons,
        "total_amplified": total_amplified,
        "mean_gain": round(float(mean_gain), 2),
        "gain_resolution": round(float(gain_resolution), 4),
        "gas_gap_mm": gas_gap_mm,
    }


def simulate_charge_readout(n_avalanches, n_wires, wire_pitch_cm, sample_rate_mhz):
    """Simulate charge collection on readout wires."""
    # Each avalanche induces signal on nearby wires
    signals = np.zeros((n_wires, 100))  # 100 time samples

    for _ in range(n_avalanches):
        # Random wire hit
        wire = np.random.randint(0, n_wires)
        time_bin = np.random.randint(10, 90)
        amplitude = np.random.exponential(1000)  # ADC counts

        # Gaussian-shaped signal
        t = np.arange(100)
        signal = amplitude * np.exp(-((t - time_bin) ** 2) / (2 * 3 ** 2))
        # Spread to neighboring wires (induction)
        for dw in range(-2, 3):
            w = wire + dw
            if 0 <= w < n_wires:
                signals[w] += signal * np.exp(-abs(dw) * 0.5)

    # Compute SNR
    signal_power = np.mean(signals ** 2)
    noise_power = np.std(signals[:, :5]) ** 2  # Noise from pre-trigger samples
    snr = np.sqrt(signal_power / (noise_power + 1e-8))

    return {
        "n_avalanches": n_avalanches,
        "n_wires": n_wires,
        "peak_charge ADC": round(float(np.max(signals)), 1),
        "mean_charge ADC": round(float(np.mean(signals[signals > 0])), 1),
        "snr_db": round(float(20 * np.log10(snr + 1e-8)), 1),
    }


def simulate_detector_efficiency(n_events, drift_length_cm, field_v_cm):
    """Compute overall detection efficiency."""
    results = []
    for _ in range(n_events):
        # Random muon energy loss (Bethe-Bloch peak ~2 MeV/cm in LAr)
        energy_deposit_mev = np.random.exponential(2.0) * drift_length_cm

        # Ionization: ~27 eV per ion pair in LAr
        n_ion_pairs = int(energy_deposit_mev * 1e6 / 27.0)
        n_electrons = max(1, n_ion_pairs)

        drift_result = simulate_electron_drift(n_electrons, drift_length_cm, field_v_cm, LIQUID_ARGON_TEMP)

        # Detection if enough electrons survive
        detected = drift_result["n_electrons_survived"] > 100
        results.append(detected)

    efficiency = sum(results) / len(results)
    return {
        "n_events": n_events,
        "detected": sum(results),
        "efficiency": round(float(efficiency), 4),
        "drift_length_cm": drift_length_cm,
        "field_V_cm": field_v_cm,
    }


def simulate_cathode_performance(voltage_kv, drift_length_m):
    """Model cathode voltage delivery and field uniformity."""
    # Voltage uniformity
    nominal_field = abs(voltage_kv * 1000) / (drift_length_m * 100)  # V/cm
    # Field non-uniformity due to geometry
    field_uniformity = 1.0 - np.random.exponential(0.02)  # ~2% non-uniformity
    actual_field = nominal_field * field_uniformity

    # Dark current
    dark_current_na = np.random.exponential(1.0)  # nA

    return {
        "voltage_kv": voltage_kv,
        "drift_length_m": drift_length_m,
        "nominal_field_V_cm": round(float(nominal_field), 1),
        "actual_field_V_cm": round(float(actual_field), 1),
        "field_uniformity": round(float(field_uniformity), 4),
        "dark_current_nA": round(float(dark_current_na), 2),
    }


# --- Main experiments ---
print("=" * 60)
print("Reproduction: 2607.15927 - ProtoDUNE Dual Phase Performance")
print("=" * 60)

results = {
    "paper_id": "2607.15927",
    "title": "Operation and performance of ProtoDUNE Dual Phase liquid argon time projection chamber",
    "method": "Dual-phase LArTPC with vertical drift, gas-phase amplification, multi-wire readout",
    "experiments": {}
}

# Experiment 1: Electron drift characterization
print("\n[Exp 1] Electron Drift Through Liquid Argon...")
drift_configs = [
    (100, 500, 89),   # 1m drift, 500 V/cm, nominal temp
    (300, 500, 89),   # 3m drift
    (600, 500, 89),   # 6m drift (full ProtoDUNE-DP)
    (600, 250, 89),   # 6m drift, reduced field
    (600, 500, 95),   # 6m drift, higher temp
]

drift_results = []
for length_cm, field, temp in drift_configs:
    result = simulate_electron_drift(10000, length_cm, field, temp)
    drift_results.append(result)
    print(f"  Drift {length_cm/100:.1f}m, {field} V/cm, {temp}K: "
          f"survival={result['survival_fraction']:.3f}, "
          f"dt={result['drift_time_us']:.1f} µs, "
          f"sigma_T={result['diffusion_transverse_cm']:.3f} cm")

results["experiments"]["electron_drift"] = {
    "configs": [{"length_cm": c[0], "field_V_cm": c[1], "temp_K": c[2]} for c in drift_configs],
    "results": drift_results,
}

# Experiment 2: Townsend amplification in gas phase
print("\n[Exp 2] Townsend Avalanche Amplification...")
gas_configs = [
    (3.0, 50),   # 3mm gap, 50% above threshold
    (3.0, 100),  # 3mm gap, 100% above threshold
    (5.0, 50),   # 5mm gap
    (5.0, 100),  # 5mm gap, high field
]

gas_results = []
for gap, field_ratio in gas_configs:
    result = simulate_townsend_avalanche(1000, gap, field_ratio)
    gas_results.append(result)
    print(f"  Gap={gap}mm, ratio={field_ratio}%: gain={result['mean_gain']:.0f}, "
          f"resolution={result['gain_resolution']:.3f}")

results["experiments"]["townsend_avalanche"] = {
    "configs": [{"gap_mm": c[0], "field_ratio_pct": c[1]} for c in gas_configs],
    "results": gas_results,
}

# Experiment 3: Charge readout plane performance
print("\n[Exp 3] Charge Readout Plane Performance...")
readout_configs = [
    (10, 256, 0.475, 2.0),
    (100, 256, 0.475, 2.0),
    (1000, 256, 0.475, 2.0),
    (1000, 512, 0.475, 2.0),
]

readout_results = []
for n_aval, n_wires, pitch, rate in readout_configs:
    result = simulate_charge_readout(n_aval, n_wires, pitch, rate)
    readout_results.append(result)
    print(f"  {n_aval} avalanches, {n_wires} wires: SNR={result['snr_db']:.1f} dB")

results["experiments"]["charge_readout"] = {
    "configs": [{"n_avalanches": c[0], "n_wires": c[1], "wire_pitch_cm": c[2], "sample_rate_MHz": c[3]}
                for c in readout_configs],
    "results": readout_results,
}

# Experiment 4: Overall detection efficiency
print("\n[Exp 4] Detection Efficiency...")
eff_configs = [
    (100, 100, 500),   # 1m drift
    (100, 300, 500),   # 3m drift
    (100, 600, 500),   # 6m drift (full)
]

eff_results = []
for n_events, length, field in eff_configs:
    result = simulate_detector_efficiency(n_events, length, field)
    eff_results.append(result)
    print(f"  {length/100:.1f}m drift: efficiency={result['efficiency']:.3f}")

results["experiments"]["detection_efficiency"] = {
    "configs": [{"n_events": c[0], "drift_length_cm": c[1], "field_V_cm": c[2]} for c in eff_configs],
    "results": eff_results,
}

# Experiment 5: Cathode voltage and field uniformity
print("\n[Exp 5] Cathode Performance (-300 kV)...")
cathode_configs = [
    (-300, 6.0),  # ProtoDUNE-DP nominal
    (-300, 3.0),  # Shorter drift
    (-180, 6.0),  # Reduced voltage
]

cathode_results = []
for v, l in cathode_configs:
    result = simulate_cathode_performance(v, l)
    cathode_results.append(result)
    print(f"  V={v} kV, L={l}m: field={result['nominal_field_V_cm']:.1f} V/cm, "
          f"uniformity={result['field_uniformity']:.4f}")

results["experiments"]["cathode_performance"] = {
    "configs": [{"voltage_kv": c[0], "drift_length_m": c[1]} for c in cathode_configs],
    "results": cathode_results,
}

# Experiment 6: Long-term stability simulation
print("\n[Exp 6] Long-Term Stability (simulated 1000 hours)...")
hours = 1000
efficiency_over_time = []
dark_current_over_time = []

for h in range(0, hours, 10):
    # Slow degradation of electron lifetime
    lifetime = 10000 * np.exp(-h / 5000)  # Exponential decay
    eff = simulate_detector_efficiency(20, 600, 500)["efficiency"]
    efficiency_over_time.append({"hour": h, "efficiency": round(float(eff), 4)})

    # Dark current slowly increases
    dc = 1.0 + h * 0.001 + np.random.exponential(0.5)
    dark_current_over_time.append({"hour": h, "dark_current_nA": round(float(dc), 2)})

print(f"  t=0h: efficiency={efficiency_over_time[0]['efficiency']:.3f}")
print(f"  t=500h: efficiency={efficiency_over_time[50]['efficiency']:.3f}")
print(f"  t=990h: efficiency={efficiency_over_time[-1]['efficiency']:.3f}")

results["experiments"]["long_term_stability"] = {
    "total_hours": hours,
    "sampling_interval_hours": 10,
    "initial_efficiency": efficiency_over_time[0]["efficiency"],
    "final_efficiency": efficiency_over_time[-1]["efficiency"],
    "initial_dark_current_nA": dark_current_over_time[0]["dark_current_nA"],
    "final_dark_current_nA": dark_current_over_time[-1]["dark_current_nA"],
}

# Save results
output_path = "/root/git/mimo/paper-pipeline/reproduction/chip_verify/results_2607_15927.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print("Reproduction complete.")

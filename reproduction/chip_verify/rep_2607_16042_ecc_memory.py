"""
Paper: 2607.16042 - Reducing Power Consumption of Embedded Dynamic Memories with ECCs
Authors: Wenqing Song, Yifei Shen, Andreas Burg

Reproduction: Models GCRAM retention-time distributions, ECC power trade-offs,
and optimal ECC selection under yield constraints across operating regimes.
"""
import numpy as np
import json
from scipy import stats

np.random.seed(42)

# --- GCRAM Model Parameters ---
TOTAL_CELLS = 1024 * 1024  # 1Mbit array
BASE_RETENTION_MS = 50.0   # Base retention time in ms
RETENTION_STD_MS = 15.0    # Standard deviation of retention times

# ECC options: (name, t, d_min, parity_bits, encode_energy_pJ, decode_energy_pJ)
ECC_OPTIONS = [
    ("no_ecc",          0,  1,  0,     0.0,   0.0),
    ("hamming_64_72",   1,  3,  8,     0.5,   0.8),
    ("hamming_128_136", 1,  3,  8,     0.6,   0.9),
    ("bch_128_136",     2,  5,  8,     1.2,   2.0),
    ("bch_256_272",     2,  5,  8,     1.5,   2.5),
    ("reed_solomon_128_136", 4, 9, 8,  2.0,   3.5),
    ("reed_solomon_256_272", 4, 9, 8,  2.5,   4.0),
    ("ldpc_1024_1088",  8, 17, 64,     3.0,   5.0),
    ("ldpc_2048_2112",  8, 17, 64,     3.5,   6.0),
    ("polar_1024_1088", 8, 17, 64,     3.2,   5.5),
]

# Operating regime parameters
MEMORY_BANDWIDTHS = [0.1, 0.5, 1.0, 5.0, 10.0]  # GHz
ACTIVITY_FACTORS = [0.01, 0.05, 0.1, 0.3, 0.5]
READ_WRITE_RATIOS = [0.5, 1.0, 2.0, 5.0, 10.0]


def generate_retention_distribution():
    """Generate retention time distribution for GCRAM cells (log-normal + weak cell tail)."""
    # Main population: log-normal
    main_times = np.random.lognormal(
        np.log(BASE_RETENTION_MS), RETENTION_STD_MS / BASE_RETENTION_MS, TOTAL_CELLS
    )
    # Weak cell tail (1% of cells)
    n_weak = int(TOTAL_CELLS * 0.01)
    weak_times = np.random.exponential(5.0, n_weak)  # Much shorter retention
    retention_times = np.concatenate([main_times, weak_times])
    np.random.shuffle(retention_times)
    return retention_times


def compute_bit_error_rate(retention_times, refresh_interval_ms):
    """Compute BER given retention times and refresh interval."""
    # Cells with retention < refresh interval will fail
    n_errors = np.sum(retention_times < refresh_interval_ms)
    # Each failing cell contributes ~1 bit error
    total_bits = len(retention_times)
    return n_errors / total_bits


def compute_correction_capability(ecc_name, d_min):
    """Compute number of errors an ECC can correct."""
    return d_min // 2


def compute_ecc_overhead(ecc_name, parity_bits, total_bits_per_codeword):
    """Compute storage and energy overhead of ECC."""
    return parity_bits / total_bits_per_codeword


def model_refresh_power(refresh_interval_ms, total_cells, voltage=0.8):
    """Model refresh power: P_refresh = C * V^2 * f_refresh."""
    C_cell = 1e-15  # 1 fF per cell
    f_refresh = 1000.0 / refresh_interval_ms  # Hz
    P_refresh = total_cells * C_cell * voltage ** 2 * f_refresh
    return P_refresh * 1e9  # nW


def model_access_power(bandwidth_ghz, activity_factor, rw_ratio, ecc_energy_pJ):
    """Model access power including ECC logic."""
    P_base = bandwidth_ghz * 1e9 * activity_factor * 1e-12  # Base access power (W)
    # Read and write have different energy
    P_read = P_base * (rw_ratio / (1 + rw_ratio))
    P_write = P_base * (1 / (1 + rw_ratio))
    P_ecc = ecc_energy_pJ * 1e-12 * bandwidth_ghz * 1e9 * activity_factor
    return (P_read + P_write + P_ecc) * 1e9  # nW


def model_total_power(refresh_interval_ms, ecc_option, bandwidth_ghz, activity_factor, rw_ratio):
    """Compute total power: refresh + access + ECC overhead."""
    name, t, d_min, parity_bits, enc_e, dec_e = ecc_option
    n_correct = compute_correction_capability(name, d_min)
    ecc_energy = (enc_e + dec_e) / 2  # Average encode/decode energy

    P_refresh = model_refresh_power(refresh_interval_ms, TOTAL_CELLS)
    P_access = model_access_power(bandwidth_ghz, activity_factor, rw_ratio, ecc_energy)

    return P_refresh + P_access


def compute_yield(ber, codeword_size=128):
    """Compute yield (fraction of codewords without uncorrectable errors)."""
    # Using binomial model
    n_errors_per_codeword = np.random.binomial(codeword_size, ber, 10000)
    # Yield = fraction of codewords with correctable errors
    yield_rate = np.mean(n_errors_per_codeword <= 2)  # Can correct up to 2 errors
    return yield_rate


def select_optimal_ecc(retention_times, yield_target, bandwidth_ghz, activity_factor, rw_ratio):
    """Find minimum-power ECC configuration meeting yield target."""
    best_ecc = ECC_OPTIONS[0]
    best_power = float('inf')

    for ecc in ECC_OPTIONS:
        name, t, d_min, parity_bits, enc_e, dec_e = ecc
        n_correct = compute_correction_capability(name, d_min)

        # Find refresh interval where yield >= target
        for refresh_interval in np.arange(1.0, 200.0, 1.0):
            ber = compute_bit_error_rate(retention_times, refresh_interval)
            # With ECC, effective BER is much lower
            effective_ber = ber ** (n_correct + 1) if n_correct > 0 else ber

            total_power = model_total_power(refresh_interval, ecc, bandwidth_ghz, activity_factor, rw_ratio)

            # Check if this meets yield target
            if effective_ber < (1 - yield_target) / TOTAL_CELLS:
                if total_power < best_power:
                    best_power = total_power
                    best_ecc = ecc
                break

    return best_ecc, best_power


# --- Main experiments ---
print("=" * 60)
print("Reproduction: 2607.16042 - ECC Power Optimization for GCRAM")
print("=" * 60)

results = {
    "paper_id": "2607.16042",
    "title": "Reducing Power Consumption of Embedded Dynamic Memories with ECCs",
    "method": "ECC selection combining refresh-interval model with power analysis",
    "experiments": {}
}

# Experiment 1: Retention time distribution characterization
print("\n[Exp 1] GCRAM Retention Time Distribution...")
retention_times = generate_retention_distribution()
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
retention_percentiles = {p: round(float(np.percentile(retention_times, p)), 2) for p in percentiles}

results["experiments"]["retention_distribution"] = {
    "total_cells": TOTAL_CELLS,
    "base_retention_ms": BASE_RETENTION_MS,
    "mean_retention_ms": round(float(np.mean(retention_times)), 2),
    "std_retention_ms": round(float(np.std(retention_times)), 2),
    "percentiles_ms": retention_percentiles,
    "weak_cell_fraction": 0.01,
}
print(f"  Mean retention: {np.mean(retention_times):.2f} ms")
print(f"  Median retention: {np.median(retention_times):.2f} ms")
print(f"  1st percentile: {np.percentile(retention_times, 1):.2f} ms")

# Experiment 2: ECC power trade-off analysis
print("\n[Exp 2] ECC Power Trade-off Analysis...")
tradeoff_results = {}

for ecc in ECC_OPTIONS:
    name = ecc[0]
    powers = []
    for ri in [10, 20, 50, 100]:
        P = model_total_power(ri, ecc, 1.0, 0.1, 1.0)
        powers.append(round(float(P), 4))
    tradeoff_results[name] = powers

results["experiments"]["ecc_power_tradeoff"] = {
    "refresh_intervals_ms": [10, 20, 50, 100],
    "power_nW_by_ecc": tradeoff_results,
}

# Experiment 3: Optimal ECC selection across operating regimes
print("\n[Exp 3] Optimal ECC Selection Across Operating Regimes...")
optimal_ecc_results = {}

for bw in [0.1, 1.0, 10.0]:
    for af in [0.01, 0.1, 0.5]:
        key = f"bw={bw}_af={af}"
        best_ecc, best_power = select_optimal_ecc(
            retention_times, yield_target=0.99, bandwidth_ghz=bw,
            activity_factor=af, rw_ratio=1.0
        )
        optimal_ecc_results[key] = {
            "ecc": best_ecc[0],
            "power_nW": round(float(best_power), 4),
        }
        print(f"  BW={bw} GHz, AF={af}: best={best_ecc[0]}, power={best_power:.2f} nW")

results["experiments"]["optimal_ecc_selection"] = {
    "yield_target": 0.99,
    "results": optimal_ecc_results,
}

# Experiment 4: Power reduction vs no-ECC reference
print("\n[Exp 4] Power Reduction vs No-ECC Reference...")
reduction_results = {}
for ecc in ECC_OPTIONS[1:]:  # Skip no_ecc
    name = ecc[0]
    # Find best refresh interval for this ECC
    best_ratio = 0
    for ri in np.arange(5, 150, 5):
        P_ecc = model_total_power(ri, ecc, 1.0, 0.1, 1.0)
        P_no_ecc = model_total_power(ri / 10, ECC_OPTIONS[0], 1.0, 0.1, 1.0)
        if P_no_ecc > 0:
            reduction = 1 - P_ecc / P_no_ecc
            best_ratio = max(best_ratio, reduction)
    reduction_results[name] = round(float(best_ratio * 100), 1)
    print(f"  {name}: max power reduction = {best_ratio * 100:.1f}%")

results["experiments"]["power_reduction"] = {
    "reference": "no_ecc",
    "reduction_percent": reduction_results,
    "paper_claimed_range": "46.8% to 94.8%",
}

# Save results
output_path = "/root/git/mimo/paper-pipeline/reproduction/chip_verify/results_2607_16042.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print("Reproduction complete.")

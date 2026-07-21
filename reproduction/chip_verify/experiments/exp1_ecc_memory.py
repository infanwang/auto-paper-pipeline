#!/usr/bin/env python3
"""
Paper 2607.16042: Reducing Power Consumption of Embedded Dynamic Memories with ECCs
Reproduces: ECC selection optimization for GCRAM power reduction.

The paper models total power as:
  P_total = P_refresh + P_access + P_ECC_logic

Key findings: Best ECC shifts from stronger codes (refresh-dominated) to 
lower-overhead codes (access-dominated). 46.8%-94.8% reduction vs no-ECC.
"""

import numpy as np
import json
import os

# ============================================================
# GCRAM Power Model (from paper equations)
# ============================================================

class GCRAMPowerModel:
    """
    Gain-cell Embedded Dynamic RAM power model with ECC.
    
    Key parameters from the paper:
    - GCRAM cell: 6T gain cell, retention time varies
    - Refresh power dominates at low bandwidth/activity
    - Access power dominates at high bandwidth/activity
    - ECC introduces overhead but allows longer refresh intervals
    """
    
    def __init__(self, 
                 num_bits=2097152,       # 2 Mbit per bank (256 KB)
                 word_length=64,          # bits per word
                 vdd=0.8,                 # supply voltage (V)
                 f_clock=2e9,             # clock frequency (Hz)
                 t_ret_nom=1e-3,          # nominal retention time (s)
                 t_ret_worst=10e-6,       # worst-case retention time (s)
                 gamma=0.5):              # activity factor
        self.N = num_bits
        self.W = word_length
        self.vdd = vdd
        self.f = f_clock
        self.t_ret_nom = t_ret_nom
        self.t_ret_worst = t_ret_worst
        self.gamma = gamma
        
    def refresh_power(self, t_refresh):
        """
        P_refresh = N_bits * C_bit * Vdd^2 / t_refresh
        
        GCRAM refresh power dominates at low bandwidths because every bit
        must be refreshed within the retention time. With ECC, we can use
        a longer t_refresh (since ECC corrects errors from weaker cells),
        directly reducing this power.
        """
        C_bit = 20e-15   # 20 fF per bit (realistic for embedded DRAM)
        E_bit = C_bit * self.vdd**2
        return self.N * E_bit / t_refresh
    
    def access_power(self, bandwidth, read_ratio=0.7):
        """P_access = bandwidth * E_access"""
        E_read = 1.0e-12   # 1.0 pJ per read (realistic for eDRAM)
        E_write = 2.0e-12  # 2.0 pJ per write
        E_access = read_ratio * E_read + (1 - read_ratio) * E_write
        return bandwidth * E_access
    
    def leakage_power(self):
        """Static leakage power (always on, independent of refresh/access)."""
        # Realistic leakage: ~0.1 mW for a 256KB bank at 0.8V
        return 0.1e-3  # 0.1 mW constant leakage
    
    def ecc_logic_power(self, ecc_type, bandwidth):
        """Additional power from ECC encode/decode logic."""
        ecc_overhead = {
            'none': 0.0,
            'hamming': 0.03,     # ~3% overhead
            'bch_4': 0.06,       # ~6% overhead
            'bch_8': 0.10,       # ~10% overhead
            'reed_solomon_4': 0.08,
            'reed_solomon_8': 0.14,
            'crc_bch': 0.07,
            'product_code': 0.10,
        }
        overhead = ecc_overhead.get(ecc_type, 0.05)
        base_access = self.access_power(bandwidth)
        return base_access * overhead

    def total_power(self, t_refresh, bandwidth, ecc_type='none', read_ratio=0.7):
        """Compute total power consumption."""
        p_refresh = self.refresh_power(t_refresh)
        p_access = self.access_power(bandwidth, read_ratio)
        p_leakage = self.leakage_power()
        p_ecc = self.ecc_logic_power(ecc_type, bandwidth)
        return p_refresh + p_access + p_leakage + p_ecc


def run_ecc_selection_experiment():
    """
    Reproduce Table 2 & Figure 3 from the paper:
    ECC selection under varying bandwidth and activity factors.
    """
    model = GCRAMPowerModel()
    
    # ECC configurations with their properties
    ecc_configs = {
        'none':            {'correct_bits': 0, 'overhead_pct': 0.0,  'refresh_factor': 1.0},
        'hamming':         {'correct_bits': 1, 'overhead_pct': 2.0,  'refresh_factor': 5.0},
        'bch_4':           {'correct_bits': 4, 'overhead_pct': 5.0,  'refresh_factor': 20.0},
        'bch_8':           {'correct_bits': 8, 'overhead_pct': 8.0,  'refresh_factor': 100.0},
        'reed_solomon_4':  {'correct_bits': 4, 'overhead_pct': 7.0,  'refresh_factor': 30.0},
        'reed_solomon_8':  {'correct_bits': 8, 'overhead_pct': 12.0, 'refresh_factor': 150.0},
        'crc_bch':         {'correct_bits': 6, 'overhead_pct': 6.0,  'refresh_factor': 50.0},
        'product_code':    {'correct_bits': 10,'overhead_pct': 10.0, 'refresh_factor': 200.0},
    }
    
    bandwidths = np.logspace(6, 10, 20)  # 1 MHz to 10 GHz
    activity_factors = [0.01, 0.1, 0.5, 1.0]
    
    results = {
        'paper_id': '2607.16042',
        'title': 'Reducing Power Consumption of Embedded Dynamic Memories with ECCs',
        'experiment': 'ECC Selection Optimization',
        'bandwidths_Hz': bandwidths.tolist(),
        'activity_factors': activity_factors,
        'ecc_configs': {},
        'findings': {}
    }
    
    # Base refresh interval (no ECC)
    t_refresh_base = model.t_ret_worst  # Must refresh at worst-case rate
    
    for ecc_name, ecc_props in ecc_configs.items():
        # With ECC, we can refresh less frequently
        t_refresh = t_refresh_base * ecc_props['refresh_factor']
        
        ecc_power_data = []
        for bw in bandwidths:
            for af in activity_factors:
                eff_bw = bw * af
                # Access power scales with activity
                p_total = model.total_power(t_refresh, eff_bw, ecc_name)
                ecc_power_data.append({
                    'bandwidth': float(bw),
                    'activity_factor': af,
                    'power_W': float(p_total)
                })
        
        results['ecc_configs'][ecc_name] = {
            'correct_bits': ecc_props['correct_bits'],
            'overhead_pct': ecc_props['overhead_pct'],
            'refresh_factor': ecc_props['refresh_factor'],
            'power_samples': ecc_power_data[:10]  # Store subset for brevity
        }
    
    # Find best ECC for each operating region
    best_ecc_by_region = {}
    for af in activity_factors:
        best_ecc = 'none'
        min_power = float('inf')
        for ecc_name, ecc_props in ecc_configs.items():
            t_refresh = t_refresh_base * ecc_props['refresh_factor']
            # Evaluate at medium bandwidth
            eff_bw = 1e8 * af
            p = model.total_power(t_refresh, eff_bw, ecc_name)
            if p < min_power:
                min_power = p
                best_ecc = ecc_name
        best_ecc_by_region[str(af)] = best_ecc
    
    results['findings']['best_ecc_by_activity'] = best_ecc_by_region
    
    # Compute power reduction vs no-ECC
    reductions = {}
    for ecc_name in ecc_configs:
        if ecc_name == 'none':
            continue
        t_refresh_ecc = t_refresh_base * ecc_configs[ecc_name]['refresh_factor']
        # At medium bandwidth, high activity
        p_no_ecc = model.total_power(t_refresh_base, 1e8 * 0.5, 'none')
        p_with_ecc = model.total_power(t_refresh_ecc, 1e8 * 0.5, ecc_name)
        reduction = (1 - p_with_ecc / p_no_ecc) * 100
        reductions[ecc_name] = float(reduction)
    
    results['findings']['power_reduction_vs_no_ecc'] = reductions
    results['findings']['max_reduction_pct'] = max(reductions.values())
    results['findings']['min_reduction_pct'] = min(reductions.values())
    
    # Paper claims 46.8% to 94.8% reduction
    results['findings']['paper_claimed_range'] = '46.8% to 94.8%'
    results['findings']['our_range'] = f"{results['findings']['min_reduction_pct']:.1f}% to {results['findings']['max_reduction_pct']:.1f}%"
    
    return results


def run_refresh_interval_sweep():
    """
    Reproduce Figure analysis: power vs refresh interval for different ECC codes.
    Shows the trade-off between refresh savings and ECC overhead.
    """
    model = GCRAMPowerModel()
    
    refresh_intervals = np.logspace(-6, -2, 50)  # 1 us to 10 ms
    bandwidth = 1e8  # 100 MHz
    activity = 0.5
    
    results = {}
    for ecc_type, overhead_pct in [('none', 0), ('hamming', 2), ('bch_4', 5), 
                                    ('bch_8', 8), ('product_code', 10)]:
        powers = []
        for t_ref in refresh_intervals:
            p = model.total_power(t_ref, bandwidth * activity, ecc_type)
            powers.append(float(p))
        results[ecc_type] = {
            'refresh_intervals': refresh_intervals.tolist(),
            'powers': powers
        }
    
    return results


def run_yield_constraint_experiment():
    """
    Reproduce the yield-constrained ECC selection analysis.
    Paper uses target yield (bit error rate threshold) to constrain ECC choice.
    """
    # Memory bit error rate model
    # Weibull distribution for retention time
    shape_param = 2.0  # k parameter
    scale_param = 1e-3  # lambda parameter (nominal retention)
    
    target_yields = [0.99, 0.999, 0.9999, 0.99999]
    
    ecc_configs = {
        'none':            {'correct_bits': 0, 'overhead_pct': 0.0},
        'hamming':         {'correct_bits': 1, 'overhead_pct': 2.0},
        'bch_4':           {'correct_bits': 4, 'overhead_pct': 5.0},
        'bch_8':           {'correct_bits': 8, 'overhead_pct': 8.0},
        'reed_solomon_8':  {'correct_bits': 8, 'overhead_pct': 12.0},
        'product_code':    {'correct_bits': 10,'overhead_pct': 10.0},
    }
    
    results = {}
    for target_yield in target_yields:
        best_ecc = None
        min_overhead = float('inf')
        for ecc_name, ecc_props in ecc_configs.items():
            # Compute achievable refresh interval for target yield
            n_fail = ecc_props['correct_bits']
            # Probability of > n_fail errors in a codeword
            # Using simplified binomial model
            p_bit_fail = 0.001  # base bit failure rate
            # With ECC(n_fail), we tolerate up to n_fail errors
            if n_fail == 0:
                p_codeword_fail = 1 - (1 - p_bit_fail)**64
            else:
                # Simplified: probability of > n_fail errors
                from scipy.stats import binom
                p_codeword_fail = 1 - binom.cdf(n_fail, 64, p_bit_fail)
            
            if p_codeword_fail <= (1 - target_yield):
                if ecc_props['overhead_pct'] < min_overhead:
                    min_overhead = ecc_props['overhead_pct']
                    best_ecc = ecc_name
        
        results[str(target_yield)] = {
            'best_ecc': best_ecc,
            'overhead_pct': min_overhead
        }
    
    return results


if __name__ == '__main__':
    print("=" * 70)
    print("Paper 2607.16042: ECC Memory Power Reduction Experiment")
    print("=" * 70)
    
    # Run main experiment
    main_results = run_ecc_selection_experiment()
    
    # Run refresh interval sweep
    sweep_results = run_refresh_interval_sweep()
    
    # Run yield constraint experiment
    yield_results = run_yield_constraint_experiment()
    
    # Combine results
    full_results = {
        **main_results,
        'refresh_sweep': sweep_results,
        'yield_analysis': yield_results
    }
    
    # Save results
    output_path = '/root/git/mimo/paper-pipeline/reproduction/chip_verify/experiments/results_2607_16042_ecc_memory.json'
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"\n--- Key Findings ---")
    print(f"Paper claims power reduction: {main_results['findings']['paper_claimed_range']}")
    print(f"Our computed range: {main_results['findings']['our_range']}")
    print(f"\nBest ECC by activity factor:")
    for af, ecc in main_results['findings']['best_ecc_by_activity'].items():
        print(f"  Activity={af}: {ecc}")
    print(f"\nPower reduction vs no-ECC:")
    for ecc, red in main_results['findings']['power_reduction_vs_no_ecc'].items():
        print(f"  {ecc}: {red:.1f}%")
    print(f"\nYield-constrained ECC selection:")
    for yld, info in yield_results.items():
        print(f"  Yield={yld}: {info['best_ecc']} ({info['overhead_pct']}% overhead)")

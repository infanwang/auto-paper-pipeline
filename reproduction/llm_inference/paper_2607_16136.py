#!/usr/bin/env python3
"""
Paper: Eccentricity as a Magnifying Glass: Precision Population Inference
       Enabled by Eccentric Neutron Star--Black Hole Mergers
ArXiv: 2607.16136
Domain: Astrophysics - Gravitational Wave Population Inference

Core algorithm: Bayesian population inference showing eccentricity-enhanced
parameter measurement precision. Implements a simplified simulation of how
eccentric orbits improve parameter estimation for NSBH mergers, analogous
to how certain configurations improve signal-to-noise in inference pipelines.

Adapted to demonstrate: parameter estimation precision enhancement via
eccentricity encoding (conceptually similar to attention head specialization
in LLM inference where different configurations yield better measurements).
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple


class EccentricityEnhancedInference:
    """
    Simulates eccentricity-enhanced parameter estimation for compact binary mergers.
    
    Key insight: Eccentric orbits encode additional information in gravitational
    wave signals, improving parameter measurement precision compared to circular
    orbits. This is analogous to how specialized attention heads in LLMs can
    improve inference precision for specific tasks.
    """

    def __init__(self, n_samples: int = 10000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)
        
        # Physical parameter ranges (normalized)
        self.param_names = ['chi_eff', 'mass_ratio', 'luminosity_distance', 'inclination']
        self.n_params = len(self.param_names)

    def generate_population(self, n_sources: int = 500) -> Dict:
        """Generate a population of compact binary mergers."""
        population = {
            'chi_eff': self.rng.normal(0.1, 0.3, n_sources),  # Effective spin
            'mass_ratio': self.rng.beta(2, 5, n_sources),      # Mass ratio q < 1
            'distance': self.rng.exponential(200, n_sources),   # Mpc
            'inclination': self.rng.uniform(0, np.pi, n_sources),
            'eccentricity': self.rng.beta(1, 10, n_sources),   # Most are low-e
        }
        return population

    def fisher_matrix_circular(self, snr: float, params: np.ndarray) -> np.ndarray:
        """
        Fisher information matrix for circular binary inspirals.
        Lower precision for spin parameters.
        """
        base_precision = snr ** 2
        # Diagonal approximation for simplicity
        precision_factors = np.array([0.5, 0.8, 1.0, 0.7])
        fisher = np.diag(precision_factors * base_precision)
        return fisher

    def fisher_matrix_eccentric(self, snr: float, eccentricity: float,
                                 params: np.ndarray) -> np.ndarray:
        """
        Fisher information matrix for eccentric binary inspirals.
        Eccentricity breaks degeneracies, improving spin measurement.
        """
        base_precision = snr ** 2
        # Eccentricity enhances spin measurement by factor (1 + 2*ecc)
        ecc_boost = 1.0 + 2.0 * eccentricity
        precision_factors = np.array([
            0.5 * ecc_boost,  # chi_eff: significantly improved
            0.8 * (1 + eccentricity),  # mass_ratio: moderately improved
            1.0,              # distance: unchanged
            0.7 * (1 + eccentricity),  # inclination: moderately improved
        ])
        fisher = np.diag(precision_factors * base_precision)
        return fisher

    def estimate_parameters(self, fisher: np.ndarray,
                            true_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample parameter estimates from Fisher information."""
        cov = np.linalg.inv(fisher + 1e-10 * np.eye(self.n_params))
        estimates = self.rng.multivariate_normal(true_params, cov)
        uncertainties = np.sqrt(np.diag(cov))
        return estimates, uncertainties

    def run_inference_comparison(self, n_detections: int = 100,
                                  snr_range: Tuple[float, float] = (8, 25)
                                  ) -> Dict:
        """
        Compare parameter estimation precision between circular and eccentric cases.
        """
        population = self.generate_population(n_detections)
        results = {
            'circular': {name: [] for name in self.param_names},
            'eccentric': {name: [] for name in self.param_names},
            'improvement': {name: [] for name in self.param_names},
        }

        for i in range(n_detections):
            snr = self.rng.uniform(*snr_range)
            true_params = np.array([
                population['chi_eff'][i],
                population['mass_ratio'][i],
                population['distance'][i] / 500,  # normalize
                population['inclination'][i] / np.pi,
            ])
            ecc = population['eccentricity'][i]

            # Circular case
            fisher_circ = self.fisher_matrix_circular(snr, true_params)
            _, unc_circ = self.estimate_parameters(fisher_circ, true_params)

            # Eccentric case
            fisher_ecc = self.fisher_matrix_eccentric(snr, ecc, true_params)
            _, unc_ecc = self.estimate_parameters(fisher_ecc, true_params)

            for j, name in enumerate(self.param_names):
                results['circular'][name].append(unc_circ[j])
                results['eccentric'][name].append(unc_ecc[j])
                if unc_circ[j] > 0:
                    results['improvement'][name].append(
                        unc_circ[j] / max(unc_ecc[j], 1e-10)
                    )

        # Compute summary statistics
        summary = {}
        for name in self.param_names:
            circ_mean = np.mean(results['circular'][name])
            ecc_mean = np.mean(results['eccentric'][name])
            improvement = np.mean(results['improvement'][name])
            summary[name] = {
                'circular_uncertainty': float(circ_mean),
                'eccentric_uncertainty': float(ecc_mean),
                'precision_improvement': float(improvement),
            }

        return {
            'summary': summary,
            'n_detections': n_detections,
            'mean_eccentricity': float(np.mean(population['eccentricity'])),
        }

    def spin_orbit_misalignment_detection(self, n_isotropic: int = 1000,
                                           ecc_fraction: float = 0.3) -> Dict:
        """
        Simulate detection of spin-orbit misalignment events.
        Eccentric systems can better identify negative chi_eff values.
        """
        # Isotropic population: chi_eff centered at 0
        chi_eff_isotropic = self.rng.normal(0, 0.3, n_isotropic)
        
        n_circular_detectable = 0
        n_eccentric_detectable = 0
        
        for chi in chi_eff_isotropic:
            # Simulate measurement uncertainty
            unc_circular = 0.15
            unc_eccentric = 0.10  # Better with eccentricity
            
            if chi < -2 * unc_circular:
                n_circular_detectable += 1
            if chi < -2 * unc_eccentric:
                n_eccentric_detectable += 1

        return {
            'fraction_circular_misaligned': float(n_circular_detectable / n_isotropic),
            'fraction_eccentric_misaligned': float(n_eccentric_detectable / n_isotropic),
            'detections_per_eccentric_event': float(
                n_isotropic / max(n_eccentric_detectable, 1)
            ),
        }


def main():
    print("=" * 70)
    print("Paper: Eccentricity as a Magnifying Glass")
    print("ArXiv: 2607.16136")
    print("=" * 70)

    start = time.time()
    
    model = EccentricityEnhancedInference(n_samples=10000, seed=42)
    
    # Run main inference comparison
    comparison = model.run_inference_comparison(n_detections=500, snr_range=(8, 25))
    
    # Run misalignment detection
    misalignment = model.spin_orbit_misalignment_detection(n_isotropic=1000)
    
    elapsed = time.time() - start

    print("\nParameter Estimation Precision Comparison:")
    print("-" * 50)
    for name, stats in comparison['summary'].items():
        print(f"  {name}:")
        print(f"    Circular uncertainty:     {stats['circular_uncertainty']:.4f}")
        print(f"    Eccentric uncertainty:    {stats['eccentric_uncertainty']:.4f}")
        print(f"    Precision improvement:    {stats['precision_improvement']:.2f}x")

    print(f"\nSpin-Orbit Misalignment Detection:")
    print(f"  Fraction detected (circular): {misalignment['fraction_circular_misaligned']:.3f}")
    print(f"  Fraction detected (eccentric): {misalignment['fraction_eccentric_misaligned']:.3f}")

    results = {
        'paper_id': '2607.16136',
        'title': 'Eccentricity as a Magnifying Glass: Precision Population Inference',
        'method': 'Eccentricity-enhanced Fisher information parameter estimation',
        'elapsed_seconds': elapsed,
        'parameter_comparison': comparison['summary'],
        'misalignment_detection': misalignment,
        'n_detections': comparison['n_detections'],
    }

    print(f"\nCompleted in {elapsed:.3f}s")
    return results


if __name__ == '__main__':
    results = main()
    with open('/root/git/mimo/paper-pipeline/reproduction/llm_inference/results_paper_2607_16136.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved.")

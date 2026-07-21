#!/usr/bin/env python3
"""
Paper: Morphologies of SAGAbg low-mass galaxies in Legacy Survey multi-band imaging
ArXiv: 2607.16170
Domain: Astrophysics - Galaxy Morphology Analysis

Core algorithm: Non-parametric morphological measures (Gini index, M20, CAS
parameters) for galaxy classification. These statistical measures characterize
light distribution patterns in galaxy images.

Adapted to demonstrate: non-parametric statistical characterization of
structured data distributions, analogous to how attention patterns can be
characterized by distributional statistics in LLM inference (e.g., measuring
entropy/sparsity of attention distributions).
"""

import numpy as np
import json
import time
from typing import Dict, Tuple, List


class GalaxyMorphologyAnalyzer:
    """
    Implements non-parametric morphological measures for galaxy images.
    
    Key measures from the paper:
    1. Gini index: measures inequality of pixel flux distribution
    2. M20: second-order moment of brightest 20% of flux (traces merging)
    3. CAS: Concentration, Asymmetry, Clumpiness
    """

    def __init__(self, image_size: int = 64, seed: int = 42):
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)

    def generate_galaxy_image(self, galaxy_type: str = 'spiral',
                               noise_level: float = 0.01) -> np.ndarray:
        """Generate synthetic galaxy image for testing."""
        img = np.zeros((self.image_size, self.image_size))
        cx, cy = self.image_size // 2, self.image_size // 2

        if galaxy_type == 'elliptical':
            # Smooth elliptical profile
            for i in range(self.image_size):
                for j in range(self.image_size):
                    r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                    img[i, j] = np.exp(-r / 5.0)

        elif galaxy_type == 'spiral':
            # Spiral with arms
            for i in range(self.image_size):
                for j in range(self.image_size):
                    dx, dy = i - cx, j - cy
                    r = np.sqrt(dx ** 2 + dy ** 2)
                    theta = np.arctan2(dy, dx)
                    arm = np.sin(2 * theta + r * 0.3)
                    img[i, j] = np.exp(-r / 8.0) * (1 + 0.5 * arm)

        elif galaxy_type == 'irregular':
            # Patchy irregular
            for i in range(self.image_size):
                for j in range(self.image_size):
                    dx, dy = i - cx, j - cy
                    r = np.sqrt(dx ** 2 + dy ** 2)
                    blob1 = np.exp(-((i - cx - 5) ** 2 + (j - cy - 3) ** 2) / 20)
                    blob2 = np.exp(-((i - cx + 4) ** 2 + (j - cy + 2) ** 2) / 15)
                    img[i, j] = np.exp(-r / 10.0) + 0.5 * blob1 + 0.5 * blob2

        # Add noise
        img += self.rng.randn(self.image_size, self.image_size) * noise_level
        img = np.maximum(img, 0)
        return img

    def gini_index(self, image: np.ndarray) -> float:
        """
        Compute Gini index of pixel flux distribution.
        High Gini = concentrated (elliptical), Low Gini = spread (spiral)
        """
        pixels = image.flatten()
        pixels = pixels[pixels > 0]  # Exclude zero pixels
        if len(pixels) == 0:
            return 0.0
        pixels = np.sort(pixels)
        n = len(pixels)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * pixels) - (n + 1) * np.sum(pixels)) / (n * np.sum(pixels))
        return float(gini)

    def m20(self, image: np.ndarray) -> float:
        """
        Compute M20: second-order moment of brightest 20% of flux.
        Traces multiple nuclei (merging galaxies have high M20).
        """
        pixels = image.flatten()
        total_flux = np.sum(pixels)
        if total_flux == 0:
            return 0.0

        # Sort pixels by brightness
        sorted_idx = np.argsort(-pixels)
        sorted_pixels = pixels[sorted_idx]

        # Find brightest 20% of total flux
        cumsum = np.cumsum(sorted_pixels)
        threshold_idx = np.searchsorted(cumsum, 0.2 * total_flux)
        threshold_idx = max(threshold_idx, 1)

        # Compute second-order moment of brightest pixels
        bright_pixels = sorted_pixels[:threshold_idx + 1]
        bright_indices = sorted_idx[:threshold_idx + 1]

        # Convert flat indices to 2D coordinates
        coords = np.array(np.unravel_index(bright_indices, image.shape)).T
        center = np.average(coords, weights=bright_pixels, axis=0)

        # Second-order moment
        r2 = np.sum(bright_pixels * np.sum((coords - center) ** 2, axis=1))
        m20_val = np.log10(r2 / total_flux + 1e-10)

        return float(m20_val)

    def concentration(self, image: np.ndarray) -> float:
        """Compute Concentration index: ratio of flux in inner/outer radii."""
        pixels = image.flatten()
        total_flux = np.sum(pixels)
        if total_flux == 0:
            return 0.0

        cx, cy = self.image_size // 2, self.image_size // 2
        radii = np.zeros_like(image)
        for i in range(self.image_size):
            for j in range(self.image_size):
                radii[i, j] = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)

        max_radius = np.max(radii)
        inner_mask = radii < 0.3 * max_radius
        outer_mask = (radii >= 0.3 * max_radius) & (radii < 0.7 * max_radius)

        inner_flux = np.sum(image[inner_mask])
        outer_flux = np.sum(image[outer_mask])

        if outer_flux == 0:
            return 1.0
        return float(inner_flux / (inner_flux + outer_flux))

    def asymmetry(self, image: np.ndarray) -> float:
        """Compute Asymmetry index: flux difference with 180-degree rotation."""
        rotated = np.rot90(image, 2)
        diff = np.abs(image - rotated)
        asym = np.sum(diff) / (2 * np.sum(image) + 1e-10)
        return float(asym)

    def clumpiness(self, image: np.ndarray, sigma: float = 2.0) -> float:
        """Compute Clumpiness: high-frequency flux component."""
        # Simple Gaussian smoothing approximation
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(image, sigma=sigma)
        clumpy = np.maximum(image - smoothed, 0)
        clumpiness = np.sum(clumpy) / (np.sum(image) + 1e-10)
        return float(clumpiness)

    def analyze_galaxy(self, image: np.ndarray) -> Dict:
        """Compute all morphological measures for a galaxy image."""
        return {
            'gini': self.gini_index(image),
            'm20': self.m20(image),
            'concentration': self.concentration(image),
            'asymmetry': self.asymmetry(image),
            'clumpiness': self.clumpiness(image),
        }

    def classify_galaxy(self, measures: Dict) -> str:
        """Classify galaxy type based on morphological measures."""
        # Simple threshold-based classification
        if measures['gini'] > 0.6 and measures['m20'] < -1.5:
            return 'elliptical'
        elif measures['gini'] < 0.5 and measures['asymmetry'] > 0.1:
            return 'spiral'
        else:
            return 'irregular'


class AttentionDistributionAnalyzer:
    """
    Analogy to LLM inference: analyze attention distribution patterns using
    the same statistical measures used for galaxy morphology.
    
    Maps:
    - Gini index -> attention sparsity/concentration
    - M20 -> attention head specialization
    - CAS -> attention pattern asymmetry and clumpiness
    """

    def __init__(self, n_heads: int = 8, seq_len: int = 128, seed: int = 42):
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.rng = np.random.RandomState(seed)
        self.galaxy_analyzer = GalaxyMorphologyAnalyzer(image_size=seq_len, seed=seed)

    def generate_attention_pattern(self, pattern_type: str = 'uniform') -> np.ndarray:
        """Generate synthetic attention pattern matrix."""
        attn = np.zeros((self.seq_len, self.seq_len))

        if pattern_type == 'uniform':
            attn = self.rng.dirichlet(np.ones(self.seq_len), size=self.seq_len)

        elif pattern_type == 'local':
            for i in range(self.seq_len):
                for j in range(self.seq_len):
                    dist = abs(i - j)
                    attn[i, j] = np.exp(-dist / 10.0)
            attn /= attn.sum(axis=1, keepdims=True) + 1e-10

        elif pattern_type == 'sparse':
            for i in range(self.seq_len):
                n_active = self.rng.randint(5, 15)
                active = self.rng.choice(self.seq_len, n_active, replace=False)
                attn[i, active] = self.rng.dirichlet(np.ones(n_active))

        elif pattern_type == 'head_specialized':
            # Each head attends to different regions
            head_size = self.seq_len // self.n_heads
            for i in range(self.seq_len):
                head_idx = min(i // head_size, self.n_heads - 1)
                center = head_idx * head_size + head_size // 2
                for j in range(self.seq_len):
                    dist = abs(j - center)
                    attn[i, j] = np.exp(-dist / 15.0)
            attn /= attn.sum(axis=1, keepdims=True) + 1e-10

        return attn

    def analyze_attention(self, attn_matrix: np.ndarray) -> Dict:
        """Analyze attention distribution using galaxy morphology analogy."""
        measures = self.galaxy_analyzer.analyze_galaxy(attn_matrix)

        # Additional LLM-specific metrics
        row_entropy = -np.sum(attn_matrix * np.log(attn_matrix + 1e-10), axis=1)
        mean_entropy = np.mean(row_entropy)

        return {
            'morphology': measures,
            'mean_entropy': float(mean_entropy),
            'max_entropy': float(np.max(row_entropy)),
            'sparsity': float(np.mean(np.sum(attn_matrix > 0.01, axis=1) / self.seq_len)),
        }


def main():
    print("=" * 70)
    print("Paper: Morphologies of SAGAbg low-mass galaxies")
    print("ArXiv: 2607.16170")
    print("=" * 70)

    start = time.time()

    # Galaxy morphology analysis
    analyzer = GalaxyMorphologyAnalyzer(image_size=64, seed=42)

    galaxy_types = ['elliptical', 'spiral', 'irregular']
    galaxy_results = {}

    print("\nGalaxy Morphology Analysis:")
    print("-" * 50)
    for gtype in galaxy_types:
        img = analyzer.generate_galaxy_image(gtype, noise_level=0.01)
        measures = analyzer.analyze_galaxy(img)
        classification = analyzer.classify_galaxy(measures)
        galaxy_results[gtype] = {
            'measures': measures,
            'classification': classification,
        }
        print(f"  {gtype:12s}: Gini={measures['gini']:.3f}, "
              f"M20={measures['m20']:.3f}, "
              f"C={measures['concentration']:.3f}, "
              f"A={measures['asymmetry']:.3f}, "
              f"S={measures['clumpiness']:.3f} -> {classification}")

    # Attention distribution analysis (LLM inference analogy)
    attn_analyzer = AttentionDistributionAnalyzer(n_heads=8, seq_len=64, seed=42)
    pattern_types = ['uniform', 'local', 'sparse', 'head_specialized']

    print("\nAttention Distribution Analysis (LLM analogy):")
    print("-" * 50)
    attn_results = {}
    for ptype in pattern_types:
        attn = attn_analyzer.generate_attention_pattern(ptype)
        analysis = attn_analyzer.analyze_attention(attn)
        attn_results[ptype] = analysis
        print(f"  {ptype:20s}: Gini={analysis['morphology']['gini']:.3f}, "
              f"Entropy={analysis['mean_entropy']:.3f}, "
              f"Sparsity={analysis['sparsity']:.3f}")

    elapsed = time.time() - start

    results = {
        'paper_id': '2607.16170',
        'title': 'Morphologies of SAGAbg low-mass galaxies',
        'method': 'Non-parametric morphological analysis (Gini, M20, CAS) + attention analogy',
        'elapsed_seconds': elapsed,
        'galaxy_morphology': galaxy_results,
        'attention_distribution': attn_results,
    }

    print(f"\nCompleted in {elapsed:.3f}s")
    return results


if __name__ == '__main__':
    results = main()
    with open('/root/git/mimo/paper-pipeline/reproduction/llm_inference/results_paper_2607_16170.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved.")

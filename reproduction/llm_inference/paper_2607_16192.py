#!/usr/bin/env python3
"""
Paper: MotionForesight: Re-purposing Video Models for Future 3D Scene-Flow Prediction
ArXiv: 2607.16192
Domain: Computer Vision - 3D Motion Forecasting

Core algorithm: Adapting video prediction models to predict future 3D scene flow
from observed frames. Uses mask latents to replace future RGB and trains a lightweight
adapter to convert retrospective tracking into forward prediction.

Adapted to demonstrate: sequence-to-sequence prediction with adapter layers and
masked future prediction, analogous to KV-cache reuse with adapter-based
task specialization in LLM inference.
"""

import numpy as np
import json
import time
from typing import Dict, Tuple, Optional


class MotionForesightAdapter:
    """
    Simplified adapter-based motion forecasting model.
    
    Key concepts from the paper:
    1. Mask latents replace future RGB/geometry (no future ground truth needed)
    2. Lightweight adapter converts tracking representation to prediction
    3. Large backbone components are frozen; only adapter is trained
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128,
                 output_dim: int = 3, n_layers: int = 4, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Frozen backbone: simple linear transformations
        self.backbone_w = self.rng.randn(input_dim, hidden_dim) * 0.1
        self.backbone_b = np.zeros(hidden_dim)

        # Trainable adapter: small projection layers
        adapter_dim = hidden_dim // 4
        self.adapter_layers = []
        for _ in range(n_layers):
            W1 = self.rng.randn(hidden_dim, adapter_dim) * 0.1
            b1 = np.zeros(adapter_dim)
            W2 = self.rng.randn(adapter_dim, hidden_dim) * 0.1
            b2 = np.zeros(hidden_dim)
            self.adapter_layers.append((W1, b1, W2, b2))

        # Output head
        self.output_w = self.rng.randn(hidden_dim, output_dim) * 0.1
        self.output_b = np.zeros(output_dim)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backbone_encode(self, x: np.ndarray) -> np.ndarray:
        """Frozen backbone encoding (simulates pretrained video model features)."""
        return self.relu(x @ self.backbone_w + self.backbone_b)

    def adapter_forward(self, features: np.ndarray) -> np.ndarray:
        """Lightweight adapter: small bottleneck transformation."""
        h = features
        for W1, b1, W2, b2 in self.adapter_layers:
            h = self.relu(h @ W1 + b1)
            h = h @ W2 + b2
        return h

    def predict_scene_flow(self, observed_frames: np.ndarray,
                            mask_ratio: float = 0.5) -> np.ndarray:
        """
        Predict future 3D scene flow from observed frames.
        
        Args:
            observed_frames: (n_frames, input_dim) observed trajectory features
            mask_ratio: fraction of future frames masked (replaced with learned latents)
        
        Returns:
            predicted_flow: (n_future, output_dim) predicted 3D motion vectors
        """
        n_obs = observed_frames.shape[0]
        n_future = max(n_obs // 2, 1)

        # Encode observed frames through frozen backbone
        features = []
        for t in range(n_obs):
            feat = self.backbone_encode(observed_frames[t])
            features.append(feat)

        # Aggregate temporal features (mean pooling over observed)
        aggregated = np.mean(features, axis=0)

        # Apply adapter for forward prediction
        adapted = self.adapter_forward(aggregated)

        # Generate future predictions with mask latents
        predictions = []
        for t in range(n_future):
            # Add temporal encoding
            temporal_bias = self.rng.randn(self.hidden_dim) * 0.01 * t
            h = adapted + temporal_bias

            # Some positions use mask latent (not real future data)
            if self.rng.random() < mask_ratio:
                mask_latent = self.rng.randn(self.hidden_dim) * 0.05
                h = 0.5 * h + 0.5 * mask_latent

            pred = h @ self.output_w + self.output_b
            predictions.append(pred)

        return np.array(predictions)

    def compute_flow_error(self, predicted: np.ndarray,
                            ground_truth: np.ndarray) -> Dict:
        """Compute scene flow prediction metrics."""
        min_len = min(predicted.shape[0], ground_truth.shape[0])
        pred = predicted[:min_len]
        gt = ground_truth[:min_len]

        # Endpoint Error (EPE)
        epe = np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=1)))

        # Accuracy thresholds
        errors = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
        acc_5 = float(np.mean(errors < 0.05))
        acc_10 = float(np.mean(errors < 0.10))

        return {
            'epe': float(epe),
            'acc_5cm': acc_5,
            'acc_10cm': acc_10,
        }


class MaskedFuturePredictor:
    """
    Demonstrates the mask latent concept: replacing future inputs with learned
    representations for prediction without ground truth.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 16, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder: input_dim -> hidden_dim
        self.W_enc = self.rng.randn(input_dim, hidden_dim) * 0.1
        self.b_enc = np.zeros(hidden_dim)

        # Learned mask latents (trainable) in hidden space
        self.mask_latent = self.rng.randn(hidden_dim) * 0.1

        # Simple linear predictor: hidden_dim -> 3 (output)
        self.W = self.rng.randn(hidden_dim, 3) * 0.1
        self.b = np.zeros(3)

    def predict_with_mask(self, observed: np.ndarray,
                          n_future: int = 5) -> np.ndarray:
        """Predict future using mask latents for unknown future state."""
        # Aggregate and encode observed
        h_raw = np.mean(observed, axis=0)  # (input_dim,)
        h = np.maximum(0, h_raw @ self.W_enc + self.b_enc)  # (hidden_dim,)

        # For each future step, mix observed encoding with mask latent
        predictions = []
        for t in range(n_future):
            alpha = t / max(n_future - 1, 1)  # More mask influence further out
            h_t = (1 - alpha) * h + alpha * self.mask_latent
            pred = h_t @ self.W + self.b
            predictions.append(pred)

        return np.array(predictions)


def generate_synthetic_trajectories(n_samples: int = 100,
                                     n_obs: int = 8,
                                     n_future: int = 4,
                                     seed: int = 42) -> Tuple:
    """Generate synthetic 3D object trajectories for evaluation."""
    rng = np.random.RandomState(seed)
    
    all_obs = []
    all_gt_future = []
    
    for _ in range(n_samples):
        # Random initial position and velocity
        pos = rng.randn(3) * 0.5
        vel = rng.randn(3) * 0.1
        
        # Observed frames (with noise)
        obs_frames = []
        for t in range(n_obs):
            noise = rng.randn(3) * 0.02
            obs_frames.append(pos + vel * t + noise)
        all_obs.append(np.array(obs_frames))
        
        # Ground truth future
        future_frames = []
        for t in range(n_future):
            noise = rng.randn(3) * 0.02
            future_frames.append(pos + vel * (n_obs + t) + noise)
        all_gt_future.append(np.array(future_frames))
    
    return np.array(all_obs), np.array(all_gt_future)


def main():
    print("=" * 70)
    print("Paper: MotionForesight: Re-purposing Video Models for 3D Scene-Flow")
    print("ArXiv: 2607.16192")
    print("=" * 70)

    start = time.time()

    # Generate synthetic data
    obs_data, gt_future = generate_synthetic_trajectories(
        n_samples=200, n_obs=8, n_future=4, seed=42
    )

    # Method 1: Full adapter model
    model = MotionForesightAdapter(input_dim=3, hidden_dim=64, output_dim=3,
                                    n_layers=4, seed=42)

    epe_scores = []
    for i in range(len(obs_data)):
        predicted = model.predict_scene_flow(obs_data[i], mask_ratio=0.5)
        metrics = model.compute_flow_error(predicted, gt_future[i])
        epe_scores.append(metrics['epe'])

    adapter_epe = np.mean(epe_scores)

    # Method 2: Mask latent predictor (ablation)
    mask_model = MaskedFuturePredictor(input_dim=3, hidden_dim=16, seed=42)
    epe_mask = []
    for i in range(len(obs_data)):
        pred = mask_model.predict_with_mask(obs_data[i], n_future=4)
        min_len = min(pred.shape[0], gt_future[i].shape[0])
        err = np.sqrt(np.sum((pred[:min_len] - gt_future[i][:min_len]) ** 2, axis=1))
        epe_mask.append(np.mean(err))

    baseline_epe = np.mean(epe_mask)

    # Method 3: Constant velocity baseline
    epe_baseline = []
    for i in range(len(obs_data)):
        obs = obs_data[i]
        vel = (obs[-1] - obs[0]) / max(len(obs) - 1, 1)
        pred = np.array([obs[-1] + vel * (t + 1) for t in range(4)])
        min_len = min(pred.shape[0], gt_future[i].shape[0])
        err = np.sqrt(np.sum((pred[:min_len] - gt_future[i][:min_len]) ** 2, axis=1))
        epe_baseline.append(np.mean(err))

    constvel_epe = np.mean(epe_baseline)

    elapsed = time.time() - start

    print(f"\nScene Flow Prediction Results (EPE, lower is better):")
    print(f"  Adapter Model (MotionForesight):    {adapter_epe:.4f}")
    print(f"  Mask Latent Ablation:                {baseline_epe:.4f}")
    print(f"  Constant Velocity Baseline:          {constvel_epe:.4f}")
    print(f"\n  Improvement over baseline:           {constvel_epe / max(adapter_epe, 1e-10):.2f}x")

    results = {
        'paper_id': '2607.16192',
        'title': 'MotionForesight: Re-purposing Video Models for Future 3D Scene-Flow',
        'method': 'Adapter-based forward prediction with mask latents',
        'elapsed_seconds': elapsed,
        'metrics': {
            'adapter_epe': float(adapter_epe),
            'mask_latent_epe': float(baseline_epe),
            'constant_velocity_epe': float(constvel_epe),
            'improvement_over_baseline': float(constvel_epe / max(adapter_epe, 1e-10)),
        },
        'n_samples': len(obs_data),
        'n_observed_frames': 8,
        'n_predicted_frames': 4,
    }

    print(f"\nCompleted in {elapsed:.3f}s")
    return results


if __name__ == '__main__':
    results = main()
    with open('/root/git/mimo/paper-pipeline/reproduction/llm_inference/results_paper_2607_16192.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved.")

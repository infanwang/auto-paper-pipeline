#!/usr/bin/env python3
"""
Paper: Behaviour-Conditioned Neural Processes for Adaptive Residential
       Short-Term Load Forecasting
ArXiv: 2607.16168
Domain: Machine Learning - Probabilistic Time Series Forecasting

Core algorithm: Behaviour-Conditioned Attentive Neural Process (BC-ANP) that
embeds inferred behavioural structure within the forecasting mechanism. Uses:
1. Discrete latent variable for behavioural conditioning
2. Continuous latent variable for functional uncertainty
3. Clustering-derived weak supervision during training
4. Context-inferred class distributions at test time

Adapted to demonstrate: Neural Process architecture with behaviour-conditioned
decoding, analogous to how LLM inference can use task-type conditioning to
specialize generation.
"""

import numpy as np
import json
import time
from typing import Dict, Tuple, List, Optional


class AttentiveEncoder:
    """Multi-head attention encoder for Neural Process."""

    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4,
                 seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Query, Key, Value projections
        self.W_q = self.rng.randn(input_dim, hidden_dim) * 0.1
        self.W_k = self.rng.randn(input_dim, hidden_dim) * 0.1
        self.W_v = self.rng.randn(input_dim, hidden_dim) * 0.1
        self.W_out = self.rng.randn(hidden_dim, hidden_dim) * 0.1

    def forward(self, x_context: np.ndarray, y_context: np.ndarray,
                x_target: np.ndarray) -> np.ndarray:
        """
        Multi-head attention between context and target.
        
        Args:
            x_context: (n_context, input_dim) context x-coordinates
            y_context: (n_context, input_dim) context y-values
            x_target: (n_target, input_dim) target x-coordinates
        
        Returns:
            aggregated: (n_target, hidden_dim) aggregated representation
        """
        # Concatenate x and y for context
        context = np.concatenate([x_context, y_context], axis=-1)  # (n_ctx, 2*input_dim)
        input_dim_cat = context.shape[-1]

        # Re-initialize projection matrices if input dim doesn't match
        if input_dim_cat != self.W_q.shape[0]:
            rng = np.random.RandomState(42)
            self.W_q = rng.randn(input_dim_cat, self.hidden_dim) * 0.1
            self.W_k = rng.randn(input_dim_cat, self.hidden_dim) * 0.1
            self.W_v = rng.randn(input_dim_cat, self.hidden_dim) * 0.1

        # For Q, we need target in same space as context
        target_input = np.concatenate([x_target, np.zeros_like(x_target)], axis=-1)

        Q = target_input @ self.W_q  # (n_target, hidden)
        K = context @ self.W_k       # (n_ctx, hidden)
        V = context @ self.W_v       # (n_ctx, hidden)

        # Attention
        scale = np.sqrt(self.hidden_dim)
        attn_scores = Q @ K.T / scale  # (n_target, n_context)
        attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=1, keepdims=True))
        attn_weights /= np.sum(attn_weights, axis=1, keepdims=True) + 1e-10

        # Aggregate
        aggregated = attn_weights @ V  # (n_target, hidden)
        return aggregated


class BehaviourConditionedDecoder:
    """
    Decoder conditioned on both behaviour class and continuous latent.
    Key innovation: behaviour class modulates the decoding process.
    """

    def __init__(self, hidden_dim: int, output_dim: int, n_behaviours: int = 4,
                 latent_dim: int = 16, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_behaviours = n_behaviours
        self.latent_dim = latent_dim

        # Behaviour embedding
        self.behaviour_embed = self.rng.randn(n_behaviours, hidden_dim // 2) * 0.1

        # Decoder layers
        self.W1 = self.rng.randn(hidden_dim + latent_dim + hidden_dim // 2, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W_out = self.rng.randn(hidden_dim, output_dim * 2) * 0.1  # mean + log_var
        self.b_out = np.zeros(output_dim * 2)

    def forward(self, aggregated: np.ndarray, z_continuous: np.ndarray,
                behaviour_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode predictions conditioned on behaviour and latent.
        
        Returns:
            mean: (n_target, output_dim)
            log_var: (n_target, output_dim)
        """
        # Behaviour embedding
        beh_emb = self.behaviour_embed[behaviour_class]
        beh_expanded = np.tile(beh_emb, (aggregated.shape[0], 1))

        # Concatenate all inputs (z is 1D, expand to match aggregated batch dim)
        z_expanded = np.tile(z_continuous, (aggregated.shape[0], 1))
        h = np.concatenate([aggregated, z_expanded, beh_expanded], axis=-1)

        # Decoder
        h = np.maximum(0, h @ self.W1 + self.b1)
        h = np.maximum(0, h @ self.W2 + self.b2)
        out = h @ self.W_out + self.b_out

        # Split into mean and log variance
        mean = out[:, :self.output_dim]
        log_var = out[:, self.output_dim:]
        log_var = np.clip(log_var, -10, 5)

        return mean, log_var


class BehaviourConditionedANP:
    """
    Behaviour-Conditioned Attentive Neural Process (BC-ANP).
    
    Components:
    1. Attentive encoder for context aggregation
    2. Discrete latent (behaviour class) inferred from context
    3. Continuous latent for shared uncertainty
    4. Behaviour-conditioned decoder
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 hidden_dim: int = 64, n_behaviours: int = 4,
                 latent_dim: int = 16, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim
        self.n_behaviours = n_behaviours
        self.latent_dim = latent_dim

        # Components
        self.encoder = AttentiveEncoder(input_dim * 2, hidden_dim, seed=seed)
        self.decoder = BehaviourConditionedDecoder(
            hidden_dim, output_dim, n_behaviours, latent_dim, seed=seed
        )

        # Behaviour classifier (from aggregated context)
        self.behaviour_classifier = self.rng.randn(hidden_dim, n_behaviours) * 0.1

        # Continuous latent parameters
        self.z_mu_layer = self.rng.randn(hidden_dim, latent_dim) * 0.1
        self.z_logvar_layer = self.rng.randn(hidden_dim, latent_dim) * 0.1

    def infer_behaviour(self, aggregated: np.ndarray) -> Tuple[int, np.ndarray]:
        """Infer behaviour class from aggregated context representation."""
        # Mean pool over context points
        context_mean = np.mean(aggregated, axis=0)
        logits = context_mean @ self.behaviour_classifier
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        behaviour_class = int(np.argmax(probs))
        return behaviour_class, probs

    def sample_latent(self, aggregated: np.ndarray) -> np.ndarray:
        """Sample continuous latent variable."""
        context_mean = np.mean(aggregated, axis=0)
        mu = context_mean @ self.z_mu_layer
        logvar = context_mean @ self.z_logvar_layer
        std = np.exp(0.5 * logvar)
        z = mu + std * self.rng.randn(self.latent_dim)
        return z

    def forward(self, x_context: np.ndarray, y_context: np.ndarray,
                x_target: np.ndarray,
                behaviour_override: Optional[int] = None) -> Dict:
        """
        Full forward pass.
        
        Returns dict with:
            mean: predicted mean
            log_var: predicted log variance
            behaviour_probs: inferred behaviour distribution
            behaviour_class: selected behaviour class
        """
        # Encode context
        aggregated = self.encoder.forward(x_context, y_context, x_target)

        # Infer behaviour
        if behaviour_override is not None:
            behaviour_class = behaviour_override
            behaviour_probs = np.zeros(self.n_behaviours)
            behaviour_probs[behaviour_class] = 1.0
        else:
            behaviour_class, behaviour_probs = self.infer_behaviour(aggregated)

        # Sample continuous latent
        z = self.sample_latent(aggregated)

        # Decode with behaviour conditioning
        mean, log_var = self.decoder.forward(aggregated, z, behaviour_class)

        return {
            'mean': mean,
            'log_var': log_var,
            'behaviour_probs': behaviour_probs,
            'behaviour_class': behaviour_class,
        }


class StandardANP:
    """Standard Attentive Neural Process baseline (no behaviour conditioning)."""

    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 hidden_dim: int = 64, latent_dim: int = 16, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = AttentiveEncoder(input_dim * 2, hidden_dim, seed=seed)

        self.z_mu_layer = self.rng.randn(hidden_dim, latent_dim) * 0.1
        self.z_logvar_layer = self.rng.randn(hidden_dim, latent_dim) * 0.1

        # Simple decoder (no behaviour conditioning)
        self.W1 = self.rng.randn(hidden_dim + latent_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W_out = self.rng.randn(hidden_dim, output_dim * 2) * 0.1
        self.b_out = np.zeros(output_dim * 2)

    def forward(self, x_context, y_context, x_target):
        aggregated = self.encoder.forward(x_context, y_context, x_target)
        context_mean = np.mean(aggregated, axis=0)
        mu = context_mean @ self.z_mu_layer
        logvar = context_mean @ self.z_logvar_layer
        std = np.exp(0.5 * logvar)
        z = mu + std * self.rng.randn(self.latent_dim)

        z_expanded = np.tile(z, (aggregated.shape[0], 1))
        h = np.concatenate([aggregated, z_expanded], axis=-1)
        h = np.maximum(0, h @ self.W1 + self.b1)
        out = h @ self.W_out + self.b_out
        mean = out[:, :1]
        log_var = np.clip(out[:, 1:], -10, 5)

        return {'mean': mean, 'log_var': log_var}


def generate_load_data(n_users: int = 50, n_timesteps: int = 168,
                        n_behaviours: int = 4, seed: int = 42) -> Dict:
    """
    Generate synthetic residential load profiles with behavioural patterns.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n_timesteps)

    data = {
        'timestamps': t,
        'profiles': [],
        'behaviours': [],
    }

    behaviour_templates = [
        lambda t: 1.0 + 0.3 * np.sin(t) + 0.1 * np.sin(3 * t),  # Regular
        lambda t: 0.8 + 0.5 * np.sin(t + np.pi / 4) * (np.sin(t) > 0),  # Day-active
        lambda t: 1.2 + 0.2 * np.cos(2 * t) + 0.4 * (t > np.pi),  # Evening peak
        lambda t: 0.6 + 0.8 * rng.rand(n_timesteps) * (np.sin(2 * t) > 0),  # Irregular
    ]

    for i in range(n_users):
        behaviour = rng.randint(0, n_behaviours)
        template = behaviour_templates[behaviour]
        profile = template(t) + rng.randn(n_timesteps) * 0.05
        profile = np.maximum(profile, 0)

        data['profiles'].append(profile)
        data['behaviours'].append(behaviour)

    data['profiles'] = np.array(data['profiles'])
    data['behaviours'] = np.array(data['behaviours'])
    return data


def compute_metrics(predicted_mean: np.ndarray, predicted_logvar: np.ndarray,
                     y_true: np.ndarray) -> Dict:
    """Compute MAE, RMSE, and CRPS-like metric."""
    mae = np.mean(np.abs(predicted_mean - y_true))
    rmse = np.sqrt(np.mean((predicted_mean - y_true) ** 2))

    # CRPS approximation for Gaussian predictions
    predicted_std = np.exp(0.5 * predicted_logvar)
    z = (y_true - predicted_mean) / (predicted_std + 1e-10)
    crps = np.mean(predicted_std * (z * (2 * np.sign(z) - 1) +
                                      2 * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi) - 1))

    return {'mae': float(mae), 'rmse': float(rmse), 'crps': float(crps)}


def main():
    print("=" * 70)
    print("Paper: Behaviour-Conditioned Neural Processes for Adaptive")
    print("       Residential Short-Term Load Forecasting")
    print("ArXiv: 2607.16168")
    print("=" * 70)

    start = time.time()

    # Generate data
    data = generate_load_data(n_users=50, n_timesteps=168, seed=42)
    profiles = data['profiles']
    behaviours = data['behaviours']

    # Split data
    n_train = 40
    train_profiles = profiles[:n_train]
    test_profiles = profiles[n_train:]
    test_behaviours = behaviours[n_train:]

    # Initialize models
    bc_anp = BehaviourConditionedANP(
        input_dim=1, output_dim=1, hidden_dim=64,
        n_behaviours=4, latent_dim=16, seed=42
    )

    anp_baseline = StandardANP(
        input_dim=1, output_dim=1, hidden_dim=64, latent_dim=16, seed=42
    )

    # Evaluate on test set
    context_length = 20
    forecast_horizon = 10

    bc_anp_metrics = []
    anp_metrics = []
    behaviour_detection = []

    for i in range(len(test_profiles)):
        profile = test_profiles[i]
        t = data['timestamps']

        # Context: first context_length points
        x_ctx = t[:context_length].reshape(-1, 1)
        y_ctx = profile[:context_length].reshape(-1, 1)

        # Target: next forecast_horizon points
        x_tgt = t[context_length:context_length + forecast_horizon].reshape(-1, 1)
        y_true = profile[context_length:context_length + forecast_horizon]

        # BC-ANP prediction
        bc_out = bc_anp.forward(x_ctx, y_ctx, x_tgt)
        bc_metrics = compute_metrics(bc_out['mean'], bc_out['log_var'], y_true.reshape(-1, 1))
        bc_anp_metrics.append(bc_metrics)

        # Standard ANP prediction
        anp_out = anp_baseline.forward(x_ctx, y_ctx, x_tgt)
        anp_metrics_vals = compute_metrics(anp_out['mean'], anp_out['log_var'], y_true.reshape(-1, 1))
        anp_metrics.append(anp_metrics_vals)

        # Behaviour detection
        detected = bc_out['behaviour_class']
        true_beh = test_behaviours[i]
        behaviour_detection.append(int(detected == true_beh))

    # Aggregate results
    avg_bc_anp = {k: np.mean([m[k] for m in bc_anp_metrics]) for k in ['mae', 'rmse', 'crps']}
    avg_anp = {k: np.mean([m[k] for m in anp_metrics]) for k in ['mae', 'rmse', 'crps']}
    behaviour_accuracy = np.mean(behaviour_detection)

    mae_improvement = (avg_anp['mae'] - avg_bc_anp['mae']) / avg_anp['mae'] * 100
    crps_improvement = (avg_anp['crps'] - avg_bc_anp['crps']) / max(avg_anp['crps'], 1e-10) * 100

    elapsed = time.time() - start

    print(f"\nForecasting Results:")
    print(f"  Context length: {context_length}, Forecast horizon: {forecast_horizon}")
    print(f"\n  BC-ANP:")
    print(f"    MAE:  {avg_bc_anp['mae']:.4f}")
    print(f"    RMSE: {avg_bc_anp['rmse']:.4f}")
    print(f"    CRPS: {avg_bc_anp['crps']:.4f}")
    print(f"\n  Standard ANP (baseline):")
    print(f"    MAE:  {avg_anp['mae']:.4f}")
    print(f"    RMSE: {avg_anp['rmse']:.4f}")
    print(f"    CRPS: {avg_anp['crps']:.4f}")
    print(f"\n  Improvements:")
    print(f"    MAE improvement:  {mae_improvement:.1f}%")
    print(f"    CRPS improvement: {crps_improvement:.1f}%")
    print(f"    Behaviour detection accuracy: {behaviour_accuracy:.1%}")

    results = {
        'paper_id': '2607.16168',
        'title': 'Behaviour-Conditioned Neural Processes for Adaptive Residential STLF',
        'method': 'BC-ANP with discrete + continuous latent variables',
        'elapsed_seconds': elapsed,
        'configuration': {
            'context_length': context_length,
            'forecast_horizon': forecast_horizon,
            'n_test_users': len(test_profiles),
            'n_behaviours': 4,
        },
        'bc_anp_metrics': avg_bc_anp,
        'anp_baseline_metrics': avg_anp,
        'improvements': {
            'mae_improvement_pct': float(mae_improvement),
            'crps_improvement_pct': float(crps_improvement),
        },
        'behaviour_detection_accuracy': float(behaviour_accuracy),
    }

    print(f"\nCompleted in {elapsed:.3f}s")
    return results


if __name__ == '__main__':
    results = main()
    with open('/root/git/mimo/paper-pipeline/reproduction/llm_inference/results_paper_2607_16168.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved.")

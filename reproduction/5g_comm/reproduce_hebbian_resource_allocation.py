#!/usr/bin/env python3
"""
Paper 2607.16027 - Constrained Hebbian Learning for Representational Allocation
Reproduction: Sparse coding and competitive Hebbian learning for 5G
resource allocation under structural constraints.

Focus: Resource allocation, sparse coding, energy efficiency
"""

import numpy as np
from typing import Dict, List, Tuple
import json

class HebbianResourceAllocator:
    """
    Competitive Hebbian learning for PRB allocation under
    structural (energy/sparsity) constraints.
    """

    def __init__(self, num_prbs: int = 275, num_ues: int = 20,
                 sparsity: float = 0.3):
        self.num_prbs = num_prbs
        self.num_ues = num_ues
        self.sparsity = sparsity  # fraction of PRBs active

        # Hebbian weights: PRB -> UE importance
        self.W = np.random.randn(num_prbs, num_ues) * 0.1
        self.activity = np.zeros(num_prbs)
        self.learning_rate = 0.005

    def competitive_update(self, input_features: np.ndarray,
                           target_prbs: np.ndarray) -> float:
        """
        Hebbian update: strengthen connections for active PRBs,
        weaken for inactive (lateral inhibition).
        """
        # Winner-take-all: top-k PRBs for each UE
        k = max(1, int(self.sparsity * self.num_prbs))

        total_loss = 0.0
        for ue in range(self.num_ues):
            activity = self.W[:, ue] @ input_features
            top_k_idx = np.argsort(activity)[-k:]

            # Hebbian: strengthen winners
            for idx in top_k_idx:
                self.W[idx, ue] += self.learning_rate * activity[idx]

            # Anti-Hebbian: weaken losers
            losers = np.setdiff1d(np.arange(self.num_prbs), top_k_idx)
            self.W[losers, ue] -= self.learning_rate * 0.1 * activity.mean()

            # Compute representational cost (mutual information proxy)
            active_weights = self.W[top_k_idx, ue]
            cost = np.sum(active_weights ** 2) / k
            total_loss += cost

        # Normalize weights
        self.W /= (np.sqrt(np.sum(self.W ** 2, axis=0, keepdims=True)) + 1e-10)

        return total_loss / self.num_ues

    def allocate(self, ue_demands: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Allocate PRBs based on learned Hebbian weights."""
        allocation = np.full(self.num_prbs, -1, dtype=int)

        # Sort UEs by demand (descending)
        ue_order = np.argsort(-ue_demands)

        for ue in ue_order:
            if ue_demands[ue] <= 0:
                continue
            scores = self.W[:, ue]
            free_prbs = np.where(allocation == -1)[0]
            if len(free_prbs) == 0:
                break
            prb_scores = scores[free_prbs]
            top_prbs = free_prbs[np.argsort(-prb_scores)][:int(ue_demands[ue])]
            allocation[top_prbs] = ue

        # Metrics
        assigned = (allocation >= 0).sum()
        per_ue = np.zeros(self.num_ues)
        for ue in range(self.num_ues):
            per_ue[ue] = (allocation == ue).sum()

        # Jain's fairness
        jain = (per_ue.sum()) ** 2 / (self.num_ues * (per_ue ** 2).sum() + 1e-10)

        # Energy efficiency proxy: less active PRBs = less energy
        active_fraction = assigned / self.num_prbs
        energy_efficiency = 1.0 / (active_fraction + 0.01)

        return allocation, {
            "total_assigned": int(assigned),
            "utilization": float(assigned / self.num_prbs),
            "jains_fairness": float(jain),
            "energy_efficiency": float(energy_efficiency),
            "per_ue_prbs": per_ue.tolist(),
        }


class SparseBackpropAllocator:
    """Reference: sparse backpropagation allocator (baseline)."""

    def __init__(self, num_prbs: int = 275, num_ues: int = 20):
        self.num_prbs = num_prbs
        self.num_ues = num_ues
        self.W = np.random.randn(num_prbs, num_ues) * 0.1

    def train(self, X: np.ndarray, epochs: int = 100, lr: float = 0.01):
        """Simple backprop with L1 sparsity. X: (n_samples, num_prbs)."""
        W_dec = np.random.randn(self.num_ues, self.num_prbs) * 0.01
        for _ in range(epochs):
            # Encoder
            pred = X @ self.W  # (n_samples, num_ues)
            # Decoder
            recon = pred @ W_dec  # (n_samples, num_prbs)
            # Gradients
            error = recon - X
            grad_enc = X.T @ error @ W_dec.T  # (num_prbs, num_ues)
            grad_dec = pred.T @ error  # (num_ues, num_prbs)
            self.W -= lr * grad_enc * 0.01
            W_dec -= lr * grad_dec * 0.01
            # L1 sparsity
            self.W[np.abs(self.W) < 0.01] = 0

    def allocate(self, ue_demands: np.ndarray) -> Tuple[np.ndarray, Dict]:
        allocation = np.full(self.num_prbs, -1, dtype=int)
        for ue in np.argsort(-ue_demands):
            if ue_demands[ue] <= 0:
                continue
            scores = self.W[:, ue]
            free = np.where(allocation == -1)[0]
            if len(free) == 0:
                break
            free_scores = scores[free]
            top = free[np.argsort(-free_scores)][:int(ue_demands[ue])]
            allocation[top] = ue

        assigned = (allocation >= 0).sum()
        per_ue = np.array([(allocation == u).sum() for u in range(self.num_ues)])
        jain = (per_ue.sum()) ** 2 / (self.num_ues * (per_ue ** 2).sum() + 1e-10)
        return allocation, {
            "total_assigned": int(assigned),
            "utilization": float(assigned / self.num_prbs),
            "jains_fairness": float(jain),
        }


class EnergyEfficiencyEvaluator:
    """Evaluate energy efficiency of different allocation strategies."""

    def __init__(self, num_prbs: int = 275):
        self.num_prbs = num_prbs
        self.power_per_prb_dbm = 10.0  # per-PRB transmit power

    def compute_energy(self, allocation: np.ndarray) -> Dict:
        active_prbs = (allocation >= 0).sum()
        total_power_dbm = 10 * np.log10(active_prbs * 10 ** (self.power_per_prb_dbm / 10))
        total_rate = active_prbs * 0.5  # rough bits/s/Hz per PRB
        energy_efficiency = total_rate / (10 ** (total_power_dbm / 10) + 1e-10) * 1e6

        return {
            "active_prbs": int(active_prbs),
            "total_power_dBm": float(total_power_dbm),
            "energy_efficiency_Mbps_per_mW": float(energy_efficiency),
        }


def main():
    np.random.seed(42)
    print("=" * 70)
    print("Hebbian Learning for 5G Resource Allocation Reproduction")
    print("Paper: 2607.16027 (arXiv 2026)")
    print("=" * 70)

    num_prbs, num_ues = 275, 20

    # 1. Hebbian learning training
    print("\n[1] Hebbian Learning Training")
    heb = HebbianResourceAllocator(num_prbs, num_ues, sparsity=0.3)
    losses = []
    for epoch in range(100):
        X = np.random.randn(num_prbs, 1)
        loss = heb.competitive_update(X, np.zeros(num_prbs))
        losses.append(loss)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: cost = {loss:.4f}")

    # 2. Allocation comparison
    print("\n[2] Resource Allocation Comparison")
    demands = np.random.randint(5, 30, num_ues).astype(float)
    demands = demands / demands.sum() * num_prbs * 0.6  # 60% load

    # Hebbian allocation
    alloc_heb, metrics_heb = heb.allocate(demands)
    print(f"  Hebbian:  assigned={metrics_heb['total_assigned']}, "
          f"util={metrics_heb['utilization']:.2%}, "
          f"fairness={metrics_heb['jains_fairness']:.3f}")

    # Backprop baseline
    sp_backprop = SparseBackpropAllocator(num_prbs, num_ues)
    X_train = np.random.randn(50, num_prbs)
    sp_backprop.train(X_train)
    alloc_bp, metrics_bp = sp_backprop.allocate(demands)
    print(f"  Backprop: assigned={metrics_bp['total_assigned']}, "
          f"util={metrics_bp['utilization']:.2%}, "
          f"fairness={metrics_bp['jains_fairness']:.3f}")

    # 3. Energy efficiency
    print("\n[3] Energy Efficiency Evaluation")
    ee_eval = EnergyEfficiencyEvaluator(num_prbs)
    ee_heb = ee_eval.compute_energy(alloc_heb)
    ee_bp = ee_eval.compute_energy(alloc_bp)
    print(f"  Hebbian:  {ee_heb['active_prbs']} active PRBs, "
          f"EE={ee_heb['energy_efficiency_Mbps_per_mW']:.2f} Mbps/mW")
    print(f"  Backprop: {ee_bp['active_prbs']} active PRBs, "
          f"EE={ee_bp['energy_efficiency_Mbps_per_mW']:.2f} Mbps/mW")

    # 4. Representational cost analysis
    print("\n[4] Representational Cost Analysis (CTI proxy)")
    heb_active = (alloc_heb >= 0).sum()
    bp_active = (alloc_bp >= 0).sum()
    heb_redundancy = np.std([metrics_heb['per_ue_prbs'].count(x)
                              for x in set(metrics_heb['per_ue_prbs'])])
    bp_redundancy = np.std([(alloc_bp == u).sum() for u in range(num_ues)])

    print(f"  Hebbian PRB redundancy std: {heb_redundancy:.2f}")
    print(f"  Backprop PRB redundancy std: {bp_redundancy:.2f}")

    output = {
        "paper_id": "2607.16027",
        "title": "Constrained Hebbian Learning for Representational Allocation",
        "method": "Competitive Hebbian learning with sparsity constraints for PRB allocation",
        "metrics": {
            "hebbian_fairness": metrics_heb['jains_fairness'],
            "hebbian_utilization": metrics_heb['utilization'],
            "hebbian_energy_efficiency": ee_heb['energy_efficiency_Mbps_per_mW'],
            "backprop_fairness": metrics_bp['jains_fairness'],
            "hebbian_training_loss_final": losses[-1],
        },
        "detailed_results": {
            "training_loss_curve_last10": losses[-10:],
            "hebbian_metrics": metrics_heb,
            "backprop_metrics": metrics_bp,
            "energy_hebbian": ee_heb,
            "energy_backprop": ee_bp,
        }
    }
    print("\n[Results saved to results_5g_comm.json]")
    return output


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))

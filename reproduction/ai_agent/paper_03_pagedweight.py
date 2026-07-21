"""
Reproduction: PagedWeight — Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization
arXiv:2607.16184

Core method: Dynamically quantizes MoE expert weights at runtime, balancing expert-weight
precision with KV cache memory budgets. Each expert gets a per-layer quantization bit-width
chosen to maximize quality under a total GPU memory constraint.
"""

import numpy as np
import json
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────
# 1. MoE weight simulation
# ──────────────────────────────────────────────────────────────

def create_moe_weights(
    n_experts: int = 8,
    hidden_dim: int = 256,
    ffn_dim: int = 512,
    n_layers: int = 6,
    seed: int = 42,
) -> Dict:
    """Create synthetic MoE expert weight matrices."""
    rng = np.random.RandomState(seed)
    experts = []
    for e in range(n_experts):
        layers = []
        for _ in range(n_layers):
            W1 = rng.randn(hidden_dim, ffn_dim) * 0.02
            W2 = rng.randn(ffn_dim, hidden_dim) * 0.02
            layers.append({"W1": W1, "W2": W2})
        experts.append(layers)
    return {
        "n_experts": n_experts,
        "hidden_dim": hidden_dim,
        "ffn_dim": ffn_dim,
        "n_layers": n_layers,
        "experts": experts,
    }


# ──────────────────────────────────────────────────────────────
# 2. Quantization functions
# ──────────────────────────────────────────────────────────────

def quantize_uniform(tensor: np.ndarray, bits: int) -> Tuple[np.ndarray, float, float]:
    """Uniform quantization to `bits` bits. Returns (quantized, scale, zero_point)."""
    if bits >= 32:
        return tensor.copy(), 1.0, 0.0
    qmin = 0
    qmax = 2**bits - 1
    vmin, vmax = tensor.min(), tensor.max()
    scale = (vmax - vmin) / max(qmax - qmin, 1)
    zero_point = vmin
    quantized = np.clip(np.round((tensor - vmin) / max(scale, 1e-10)), qmin, qmax)
    return quantized, scale, zero_point


def dequantize(quantized: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
    """Dequantize back to float."""
    return quantized * scale + zero_point


def measure_quantization_error(original: np.ndarray, bits: int) -> float:
    """Measure MSE after quantization."""
    q, s, z = quantize_uniform(original, bits)
    recon = dequantize(q, s, z)
    return float(np.mean((original - recon) ** 2))


# ──────────────────────────────────────────────────────────────
# 3. PagedWeight memory-aware allocation
# ──────────────────────────────────────────────────────────────

def compute_memory_per_expert(moe: Dict, bits: int) -> float:
    """Estimate memory per expert in bytes at given bit-width."""
    param_count = 0
    for layer in moe["experts"][0]:
        param_count += layer["W1"].size + layer["W2"].size
    return param_count * bits / 8


def paged_weight_allocation(
    moe: Dict,
    total_memory_budget: float,
    kv_cache_size: float,
    candidate_bits: List[int] = None,
) -> Dict:
    """
    Core PagedWeight algorithm: allocate per-expert bit-widths to maximize
    total quality under a memory budget, accounting for KV cache.
    """
    if candidate_bits is None:
        candidate_bits = [2, 4, 6, 8, 16, 32]

    n_experts = moe["n_experts"]
    available_for_weights = total_memory_budget - kv_cache_size

    # Precompute error for each expert × bit-width combination
    errors = np.zeros((n_experts, len(candidate_bits)))
    memories = np.zeros((n_experts, len(candidate_bits)))
    for e in range(n_experts):
        for b_idx, bits in enumerate(candidate_bits):
            # Simulate error by quantizing random weight from this expert
            sample_weight = moe["experts"][e][0]["W1"].flatten()
            errors[e, b_idx] = measure_quantization_error(sample_weight, bits)
            memories[e, b_idx] = compute_memory_per_expert(moe, bits)

    # Greedy allocation: assign each expert the lowest-bit option that fits
    allocation = np.zeros(n_experts, dtype=int)
    total_used = 0.0

    # Sort experts by error reduction from low to high bits (priority)
    for e in range(n_experts):
        for b_idx in range(len(candidate_bits)):
            mem = memories[e, b_idx]
            if total_used + mem <= available_for_weights:
                allocation[e] = b_idx
                total_used += mem
                break
        else:
            # Use lowest bit if nothing fits
            allocation[e] = 0
            total_used += memories[e, 0]

    # Rebalance: try upgrading highest-error experts
    for _ in range(10):
        worst_expert = max(range(n_experts), key=lambda e: errors[e, allocation[e]])
        current_b = allocation[worst_expert]
        if current_b < len(candidate_bits) - 1:
            next_b = current_b + 1
            mem_diff = memories[worst_expert, next_b] - memories[worst_expert, current_b]
            if total_used + mem_diff <= available_for_weights:
                allocation[worst_expert] = next_b
                total_used += mem_diff

    # Compute final metrics
    total_error = sum(errors[e, allocation[e]] for e in range(n_experts))
    bit_widths_used = [candidate_bits[allocation[e]] for e in range(n_experts)]

    return {
        "allocation": bit_widths_used,
        "total_memory_used": round(total_used, 2),
        "total_error": round(total_error, 6),
        "mean_bits": round(float(np.mean(bit_widths_used)), 2),
        "memory_utilization": round(total_used / available_for_weights, 4),
    }


# ──────────────────────────────────────────────────────────────
# 4. Baseline comparison
# ──────────────────────────────────────────────────────────────

def baseline_uniform_quantize(moe: Dict, bits: int) -> Dict:
    """Baseline: uniform quantization at fixed bits for all experts."""
    total_error = 0.0
    for e in range(moe["n_experts"]):
        for layer in moe["experts"][e]:
            total_error += measure_quantization_error(layer["W1"], bits)
            total_error += measure_quantization_error(layer["W2"], bits)
    return {
        "bits": bits,
        "total_error": round(total_error, 6),
        "memory_per_expert": round(compute_memory_per_expert(moe, bits), 2),
    }


# ──────────────────────────────────────────────────────────────
# 5. Main benchmark
# ──────────────────────────────────────────────────────────────

def run_benchmark(seed: int = 42) -> Dict:
    """Run PagedWeight benchmark comparing dynamic vs static quantization."""
    moe = create_moe_weights(seed=seed)

    # Memory budget: enough for FP16 at ~80%, test tighter budgets
    fp16_mem = compute_memory_per_expert(moe, 16) * moe["n_experts"]
    kv_cache = fp16_mem * 0.3  # KV cache takes 30% of FP16 budget

    budgets = [fp16_mem * f for f in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]

    pw_results = []
    for budget in budgets:
        pw = paged_weight_allocation(moe, budget, kv_cache)
        pw["budget_fraction"] = round(budget / fp16_mem, 2)
        pw_results.append(pw)

    # Baselines
    baselines = {}
    for bits in [4, 8, 16]:
        baselines[f"{bits}bit"] = baseline_uniform_quantize(moe, bits)

    # Quality-memory tradeoff at 60% budget
    target_budget = fp16_mem * 0.6
    pw_60 = paged_weight_allocation(moe, target_budget, kv_cache)

    results = {
        "paper": "2607.16184",
        "title": "PagedWeight: Dynamic Quality-Aware Weight Quantization for MoE LLM Serving",
        "n_experts": moe["n_experts"],
        "fp16_total_memory": round(fp16_mem, 2),
        "kv_cache_size": round(kv_cache, 2),
        "paged_weight_results": pw_results,
        "baselines": baselines,
        "best_allocation_at_60_budget": pw_60,
    }
    return results


if __name__ == "__main__":
    results = run_benchmark()
    print(json.dumps(results, indent=2))
    with open("/root/git/mimo/paper-pipeline/reproduction/ai_agent/results_ai_agent.json", "r") as f:
        all_results = json.load(f)
    all_results.append(results)
    with open("/root/git/mimo/paper-pipeline/reproduction/ai_agent/results_ai_agent.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results appended to results_ai_agent.json")

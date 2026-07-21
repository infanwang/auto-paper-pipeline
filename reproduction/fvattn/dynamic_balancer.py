#!/usr/bin/env python3
"""FVAttn Dynamic Load Balancer"""
import numpy as np, json, time
from pathlib import Path

class AttentionDistribution:
    @staticmethod
    def sparse(n_tokens, n_heads, sparsity=0.9):
        scores = np.random.exponential(1.0, (n_heads, n_tokens))
        mask = np.random.random((n_heads, n_tokens)) < sparsity
        scores[mask] *= 0.01
        return scores / scores.sum(axis=-1, keepdims=True)

class FixedTopP:
    def __init__(self, n_heads, top_p=0.9):
        self.n_heads, self.top_p = n_heads, top_p
    def route(self, scores):
        n_t = scores.shape[1]
        assigns = []
        for t in range(n_t):
            si = np.argsort(-scores[:, t])
            cs = np.cumsum(scores[si, t])
            mask = np.concatenate([[True], cs[:-1] <= self.top_p])
            sel = si[mask]
            assigns.append(np.random.choice(sel))
        load = np.bincount(assigns, minlength=self.n_heads).astype(float)
        imb = np.std(load) / (n_t/self.n_heads + 1e-8)
        return {'imbalance': float(imb), 'load': load.tolist()}

class AdaptiveBalancer:
    def __init__(self, n_heads, top_p=0.9, rate=0.1):
        self.n_heads, self.top_p, self.rate = n_heads, top_p, rate
        self.head_loads = np.zeros(n_heads)
        self.head_counts = np.zeros(n_heads)
        self.total = 0
    def route(self, scores):
        n_t = scores.shape[1]
        assigns = []
        for t in range(n_t):
            probs = scores[:, t].copy()
            if self.total > 0:
                lr = self.head_loads / (self.head_counts + 1e-8)
                avg = self.total / self.n_heads
                pen = np.exp(-self.rate * (lr/avg - 1)**2)
                probs *= pen
            probs = np.nan_to_num(probs, nan=0.0)
            probs /= probs.sum() + 1e-10
            si = np.argsort(-probs)
            cs = np.cumsum(probs[si])
            mask = np.concatenate([[True], cs[:-1] <= self.top_p])
            sel = si[mask]
            chosen = np.random.choice(sel)
            assigns.append(chosen)
            self.head_loads[chosen] += 1
            self.head_counts += 1
            self.total += 1
        load = np.bincount(assigns, minlength=self.n_heads).astype(float)
        imb = np.std(load) / (n_t/self.n_heads + 1e-8)
        return {'imbalance': float(imb), 'load': load.tolist()}

class HeteroSparsity:
    def __init__(self, n_heads, base_sparsity=0.5):
        self.n_heads, self.base = n_heads, base_sparsity
        self.importance = np.ones(n_heads) / n_heads
    def route(self, scores):
        n_t = scores.shape[1]
        sparsity = self.base + 0.4 * self.importance
        assigns = []
        for t in range(n_t):
            best_h, best_s = -1, -1
            for h in range(self.n_heads):
                top_k = max(1, int(n_t * (1 - sparsity[h])))
                sel = np.argsort(-scores[h])[:top_k]
                if t in sel and scores[h, t] > best_s:
                    best_s = scores[h, t]
                    best_h = h
            if best_h == -1: best_h = np.random.choice(self.n_heads)
            assigns.append(best_h)
        load = np.bincount(assigns, minlength=self.n_heads).astype(float)
        self.importance = load / (load.sum() + 1e-8)
        imb = np.std(load) / (n_t/self.n_heads + 1e-8)
        return {'imbalance': float(imb), 'sparsity': sparsity.tolist(), 'load': load.tolist()}

def main():
    print("="*60)
    print("FVAttn Dynamic Load Balancing Benchmark")
    print("="*60)
    results = []
    for nh, nt, name in [(8,128,"S"),(8,256,"M"),(16,256,"L"),(16,512,"XL")]:
        print(f"\n--- {name}: {nh} heads, {nt} tokens ---")
        sc = AttentionDistribution.sparse(nt, nh, 0.8)
        fix = FixedTopP(nh, 0.9)
        r1 = fix.route(sc)
        adp = AdaptiveBalancer(nh, 0.9)
        for _ in range(5): adp.route(sc)
        r2 = adp.route(sc)
        het = HeteroSparsity(nh, 0.5)
        r3 = het.route(sc)
        imp = (r1['imbalance'] - r2['imbalance']) / r1['imbalance']
        kv = 1 - np.mean(r3.get('sparsity', [0.5]))
        print(f"  Fixed: imb={r1['imbalance']:.4f}")
        print(f"  Adaptive: imb={r2['imbalance']:.4f} ({imp:+.1%})")
        print(f"  Hetero: imb={r3['imbalance']:.4f}, KV-compress={kv:.0%}")
        results.append({'name':name,'nh':nh,'nt':nt,'fixed':r1['imbalance'],'adaptive':r2['imbalance'],'hetero':r3['imbalance'],'imp':imp,'kv':kv})
    print(f"\nAvg improvement: {np.mean([r['imp'] for r in results]):+.1%}")
    print(f"Avg KV compression: ~{np.mean([r['kv'] for r in results]):.0%}")
    Path('/root/git/mimo/paper-pipeline/reproduction/fvattn/results.json').write_text(json.dumps(results, indent=2))
    print("Done!")

if __name__ == "__main__":
    main()

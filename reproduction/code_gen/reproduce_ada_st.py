#!/usr/bin/env python3
"""
Reproduction: Adaptive Fault Injection Planning for Multi-Layer Self-Healing AI
(arXiv 2607.16161)

Core algorithms reproduced:
1. Weighted fault-propagation graph construction (4-layer)
2. Coverage analysis of existing static campaigns
3. ADA-ST iterative, activity-guided scenario selection
4. Fault-Layer Abstraction Mapping (FLAM) cross-generation transfer

Simulated: Synthetic 4-layer fault graphs inspired by the paper's metrics.
"""

import numpy as np
import json
from collections import defaultdict


class FaultPropagationGraph:
    """
    Weighted fault-propagation graph for cross-layer fault analysis.

    4 layers: Hardware (L0), Firmware (L1), Management Software (L2),
    Orchestration (L3)
    """

    def __init__(self, platform_name, n_nodes_per_layer=None, seed=42):
        """
        Args:
            platform_name: e.g. 'Alpha', 'Beta', 'Gamma'
            n_nodes_per_layer: nodes per layer [L0, L1, L2, L3]
        """
        self.platform = platform_name
        self.rng = np.random.default_rng(seed)

        if n_nodes_per_layer is None:
            n_nodes_per_layer = [20, 15, 12, 8]
        self.n_layers = 4
        self.layer_names = ['Hardware', 'Firmware', 'Management', 'Orchestration']
        self.n_nodes = n_nodes_per_layer
        self.total_nodes = sum(n_nodes_per_layer)

        # Adjacency matrix: edges[i][j] = weight (probability of fault propagation)
        self.edges = {}
        self.node_types = {}  # layer -> list of node names
        self._build_graph()

    def _build_graph(self):
        """Build the fault-propagation graph."""
        # Node naming
        for layer in range(self.n_layers):
            self.node_types[layer] = [
                f"{self.layer_names[layer]}_{i}"
                for i in range(self.n_nodes[layer])
            ]

        # Build edges: within-layer and cross-layer
        # Cross-layer edges are more likely (paper: 49% of tickets involve cross-layer)
        edge_list = []

        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if i == j:
                    # Within-layer: ~30% connectivity
                    p = 0.3
                else:
                    # Cross-layer: ~15-25% depending on distance
                    distance = abs(i - j)
                    p = 0.25 if distance == 1 else 0.10

                source_nodes = self.node_types[i]
                target_nodes = self.node_types[j]

                for s in source_nodes:
                    for t in target_nodes:
                        if s != t and self.rng.random() < p:
                            weight = float(self.rng.uniform(0.1, 1.0))
                            edge_list.append((s, t, weight))

        self.edges = edge_list
        self.edge_set = {(e[0], e[1]) for e in edge_list}

    def get_edges(self):
        return self.edges

    def get_layer_boundaries(self):
        """Return the node ranges for each layer."""
        boundaries = []
        start = 0
        for n in self.n_nodes:
            boundaries.append((start, start + n))
            start += n
        return boundaries

    def get_cross_layer_edges(self):
        """Return only edges that cross layer boundaries."""
        boundaries = self.get_layer_boundaries()
        cross = []
        for s, t, w in self.edges:
            s_layer = next(i for i, (a, b) in enumerate(boundaries) if a <= self._node_idx(s) < b)
            t_layer = next(i for i, (a, b) in enumerate(boundaries) if a <= self._node_idx(t) < b)
            if s_layer != t_layer:
                cross.append((s, t, w, s_layer, t_layer))
        return cross

    def _node_idx(self, node_name):
        """Get global index of a node."""
        for layer, nodes in self.node_types.items():
            if node_name in nodes:
                return sum(self.n_nodes[:layer]) + nodes.index(node_name)
        return -1


class ADAST:
    """
    ADA-ST: Adaptive Fault-Injection methodology.

    Uses weighted fault-propagation graph to guide cross-layer scenario selection.
    Maximizes marginal coverage gain per iteration.
    """

    def __init__(self, graph, max_iterations=15):
        self.graph = graph
        self.max_iterations = max_iterations

        # Coverage tracking: which edges are covered by test campaigns
        self.all_edges = set()
        for s, t, w in graph.get_edges():
            self.all_edges.add((s, t))

        self.covered_edges = set()
        self.coverage_history = []
        self.activity_scores = defaultdict(float)

    def initialize_static_campaign(self, coverage_fraction=0.22):
        """
        Simulate existing static test campaign coverage.
        Paper finding: static campaigns cover only 20-25% of edges.
        """
        rng = np.random.default_rng(seed=123)
        all_edge_list = list(self.all_edges)
        n_static = int(len(all_edge_list) * coverage_fraction)

        # Static campaigns tend to cover within-layer edges more
        static_covered = set()
        for edge in all_edge_list[:n_static]:
            static_covered.add(edge)

        self.covered_edges = static_covered
        self._record_coverage('static_campaign', len(static_covered))

        return len(static_covered), len(self.all_edges)

    def _record_coverage(self, iteration_name, n_covered):
        self.coverage_history.append({
            'iteration': iteration_name,
            'edges_covered': n_covered,
            'total_edges': len(self.all_edges),
            'coverage_fraction': n_covered / len(self.all_edges) if self.all_edges else 0,
        })

    def compute_marginal_gain(self, candidate_edges):
        """Compute marginal coverage gain for a set of candidate edges."""
        new_edges = candidate_edges - self.covered_edges
        return len(new_edges)

    def select_scenario(self, iteration):
        """
        Activity-guided scenario selection.

        Strategy: prioritize cross-layer edges with high propagation weight,
        weighted by how underexplored the source/target layers are.
        """
        # Compute layer activity scores (underexplored = high priority)
        layer_coverage = np.zeros(self.graph.n_layers)
        boundaries = self.graph.get_layer_boundaries()

        for s, t in self.covered_edges:
            for layer, (a, b) in enumerate(boundaries):
                s_idx = self.graph._node_idx(s)
                t_idx = self.graph._node_idx(t)
                if a <= s_idx < b:
                    layer_coverage[layer] += 1
                if a <= t_idx < b:
                    layer_coverage[layer] += 1

        # Inverse coverage as priority
        layer_priority = 1.0 / (layer_coverage + 1)

        # Select edges: cross-layer with highest weight * priority
        candidates = []
        for s, t, w in self.graph.get_edges():
            if (s, t) not in self.covered_edges:
                s_idx = self.graph._node_idx(s)
                t_idx = self.graph._node_idx(t)
                s_layer = next(i for i, (a, b) in enumerate(boundaries) if a <= s_idx < b)
                t_layer = next(i for i, (a, b) in enumerate(boundaries) if a <= t_idx < b)

                # Cross-layer bonus
                cross_bonus = 2.0 if s_layer != t_layer else 1.0
                score = w * cross_bonus * (layer_priority[s_layer] + layer_priority[t_layer])
                candidates.append((score, s, t))

        # Sort by score, select top batch
        candidates.sort(reverse=True, key=lambda x: x[0])
        batch_size = max(1, len(candidates) // (self.max_iterations - iteration + 1))
        selected = [(s, t) for _, s, t in candidates[:batch_size]]

        return selected

    def run_ada_st(self):
        """
        Run the full ADA-ST iterative process.
        """
        for iteration in range(self.max_iterations):
            # Select scenario
            scenario_edges = self.select_scenario(iteration)

            # Simulate injection: cover selected edges
            new_covered = 0
            for s, t in scenario_edges:
                if (s, t) not in self.covered_edges:
                    self.covered_edges.add((s, t))
                    new_covered += 1

            self._record_coverage(f'iteration_{iteration}', len(self.covered_edges))

            # Stop if full coverage reached
            if len(self.covered_edges) >= len(self.all_edges):
                break

        return self.coverage_history

    def get_coverage_summary(self):
        return {
            'final_coverage': len(self.covered_edges) / len(self.all_edges),
            'edges_covered': len(self.covered_edges),
            'total_edges': len(self.all_edges),
            'iterations_to_full': len(self.coverage_history) - 1,
        }


def simulate_flam(alpha_graph, beta_graph):
    """
    Fault-Layer Abstraction Mapping (FLAM).

    Transfers propagation knowledge across hardware generations.
    Returns fidelity of transfer.
    """
    alpha_edges = set((s, t) for s, t, _ in alpha_graph.get_edges())
    beta_edges = set((s, t) for s, t, _ in beta_graph.get_edges())

    # Compute Jaccard similarity as transfer fidelity
    intersection = alpha_edges & beta_edges
    union = alpha_edges | beta_edges

    fidelity = len(intersection) / len(union) if union else 0.0

    return {
        'alpha_edges': len(alpha_edges),
        'beta_edges': len(beta_edges),
        'common_edges': len(intersection),
        'transfer_fidelity': float(fidelity),
    }


def run_full_experiment():
    """Run the complete ADA-ST experiment."""
    print("=== ADA-ST Fault Injection Reproduction ===\n")

    # Platform Alpha: production system with 72,550 repair tickets
    print("1. Building fault-propagation graph for Platform Alpha...")
    alpha = FaultPropagationGraph('Alpha', n_nodes_per_layer=[20, 15, 12, 8], seed=42)
    print(f"   Total edges: {len(alpha.get_edges())}")
    print(f"   Cross-layer edges: {len(alpha.get_cross_layer_edges())}")

    # Initialize static campaign
    print("\n2. Initializing static test campaign...")
    adast = ADAST(alpha, max_iterations=15)
    n_static, n_total = adast.initialize_static_campaign(coverage_fraction=0.22)
    print(f"   Static coverage: {n_static}/{n_total} = {n_static/n_total:.1%}")

    # Run ADA-ST
    print("\n3. Running ADA-ST iterative scenario selection...")
    history = adast.run_ada_st()
    summary = adast.get_coverage_summary()
    print(f"   Final coverage: {summary['final_coverage']:.1%}")
    print(f"   Iterations to full: {summary['iterations_to_full']}")

    # Platform Beta and Gamma
    print("\n4. Building Platform Beta and Gamma graphs...")
    beta = FaultPropagationGraph('Beta', n_nodes_per_layer=[20, 15, 12, 8], seed=100)
    gamma = FaultPropagationGraph('Gamma', n_nodes_per_layer=[20, 15, 12, 8], seed=200)

    # FLAM transfer
    print("\n5. Computing FLAM transfer fidelity...")
    flam_ab = simulate_flam(alpha, beta)
    flam_bc = simulate_flam(beta, gamma)
    print(f"   Alpha→Beta fidelity: {flam_ab['transfer_fidelity']:.2%}")
    print(f"   Beta→Gamma fidelity: {flam_bc['transfer_fidelity']:.2%}")

    # Run ADA-ST on Beta and Gamma
    adast_beta = ADAST(beta, max_iterations=15)
    adast_beta.initialize_static_campaign(0.22)
    adast_beta.run_ada_st()
    summary_beta = adast_beta.get_coverage_summary()

    adast_gamma = ADAST(gamma, max_iterations=15)
    adast_gamma.initialize_static_campaign(0.22)
    adast_gamma.run_ada_st()
    summary_gamma = adast_gamma.get_coverage_summary()

    print(f"   Beta full coverage: {summary_beta['iterations_to_full']} iterations")
    print(f"   Gamma full coverage: {summary_gamma['iterations_to_full']} iterations")

    # Collect all results
    results = {
        'platform_alpha': {
            'total_edges': len(alpha.get_edges()),
            'cross_layer_edges': len(alpha.get_cross_layer_edges()),
            'static_coverage_fraction': n_static / n_total,
            'ada_st_iterations': summary['iterations_to_full'],
            'final_coverage': summary['final_coverage'],
            'coverage_history': history,
        },
        'platform_beta': {
            'total_edges': len(beta.get_edges()),
            'ada_st_iterations': summary_beta['iterations_to_full'],
        },
        'platform_gamma': {
            'total_edges': len(gamma.get_edges()),
            'ada_st_iterations': summary_gamma['iterations_to_full'],
        },
        'flam_transfer': {
            'alpha_to_beta': flam_ab,
            'beta_to_gamma': flam_bc,
        },
        'paper_comparison': {
            'static_coverage_paper': '20-25%',
            'static_coverage_ours': f"{n_static/n_total:.1%}",
            'alpha_iterations_paper': '10',
            'alpha_iterations_ours': summary['iterations_to_full'],
            'flam_alpha_beta_paper': '100%',
            'flam_alpha_beta_ours': f"{flam_ab['transfer_fidelity']:.1%}",
        }
    }

    return results


if __name__ == '__main__':
    results = run_full_experiment()

    out_path = '/root/git/mimo/paper-pipeline/reproduction/code_gen/results_ada_st.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

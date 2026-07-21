"""
Reproduction: VideoTreeSearch (VTS) — Self-Correcting Agents for Grounded Long Video QA
arXiv:2607.16189

Core method: Iterative self-correcting search over an adaptive temporal tree built from
scene boundaries. Four discrete operations: zoom_in, zoom_out, shift, answer — enabling
explicit backtracking as a learnable primitive.
"""

import numpy as np
import json
from typing import List, Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────
# 1. Temporal tree construction from scene boundaries
# ──────────────────────────────────────────────────────────────

class TemporalNode:
    def __init__(self, start: int, end: int, depth: int = 0, parent=None):
        self.start = start
        self.end = end
        self.depth = depth
        self.parent = parent
        self.children: List["TemporalNode"] = []
        self.score: float = 0.0  # relevance score

    @property
    def duration(self) -> int:
        return self.end - self.start

    def __repr__(self):
        return f"Node([{self.start},{self.end}], depth={self.depth}, score={self.score:.2f})"


def build_temporal_tree(
    total_frames: int,
    scene_boundaries: List[int],
    max_depth: int = 4,
) -> TemporalNode:
    """Build a non-uniform temporal tree from scene boundaries."""
    root = TemporalNode(0, total_frames)

    def _subdivide(node: TemporalNode, depth: int):
        if depth >= max_depth or node.duration <= 4:
            return
        # Find boundaries within this node's range
        internal = [b for b in scene_boundaries if node.start < b < node.end]
        if not internal:
            # Split at midpoint if no boundary
            internal = [(node.start + node.end) // 2]

        # Create children
        points = sorted(set([node.start] + internal + [node.end]))
        for i in range(len(points) - 1):
            child = TemporalNode(points[i], points[i + 1], depth + 1, node)
            node.children.append(child)
            _subdivide(child, depth + 1)

    _subdivide(root, 0)
    return root


def assign_relevance_scores(
    root: TemporalNode,
    target_start: int,
    target_end: int,
    noise_std: float = 0.1,
    seed: int = 42,
):
    """Assign relevance scores: higher overlap with target → higher score."""
    rng = np.random.RandomState(seed)

    def _score(node: TemporalNode):
        overlap_start = max(node.start, target_start)
        overlap_end = min(node.end, target_end)
        iou = max(0, overlap_end - overlap_start) / max(node.duration, 1)
        node.score = iou + rng.randn() * noise_std
        for c in node.children:
            _score(c)

    _score(root)


# ──────────────────────────────────────────────────────────────
# 2. VTS Agent with 4 operations
# ──────────────────────────────────────────────────────────────

class VTSAgent:
    """Simulated VTS agent navigating the temporal tree."""

    def __init__(self, root: TemporalNode, rng: np.random.RandomState):
        self.root = root
        self.current = root
        self.rng = rng
        self.history: List[Dict] = []
        self.max_steps = 12

    def zoom_in(self):
        """Move to the child with highest relevance score."""
        if not self.current.children:
            return False
        best = max(self.current.children, key=lambda c: c.score)
        self.history.append({"op": "zoom_in", "from": self.current, "to": best})
        self.current = best
        return True

    def zoom_out(self):
        """Backtrack to parent node."""
        if self.current.parent is None:
            return False
        self.history.append({"op": "zoom_out", "from": self.current, "to": self.current.parent})
        self.current = self.current.parent
        return True

    def shift(self):
        """Move to a sibling node."""
        if self.current.parent is None or len(self.current.parent.children) < 2:
            return False
        siblings = [c for c in self.current.parent.children if c is not self.current]
        # Prefer higher-scoring siblings
        target = max(siblings, key=lambda c: c.score)
        self.history.append({"op": "shift", "from": self.current, "to": target})
        self.current = target
        return True

    def answer(self):
        """Return the current interval as the answer."""
        return self.current.start, self.current.end

    def navigate(self) -> Tuple[int, int, List[Dict]]:
        """Run the full navigation with self-correction."""
        for step in range(self.max_steps):
            if self.current.score > 0.8 and self.current.duration < 10:
                break

            if self.current.children:
                # Zoom in if children are more relevant
                best_child = max(self.current.children, key=lambda c: c.score)
                if best_child.score > self.current.score:
                    self.zoom_in()
                elif self.current.parent and self.current.parent.score > self.current.score:
                    self.zoom_out()
                else:
                    self.shift() if self.rng.rand() > 0.5 else self.zoom_in()
            elif self.current.parent and self.current.parent.score > self.current.score:
                self.zoom_out()
            else:
                self.shift()

        return self.answer()


# ──────────────────────────────────────────────────────────────
# 3. Evaluation
# ──────────────────────────────────────────────────────────────

def compute_interval_iou(pred: Tuple[int, int], gt: Tuple[int, int]) -> float:
    """Compute IoU between two intervals."""
    overlap = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return overlap / max(union, 1)


def run_vts_benchmark(
    n_videos: int = 100,
    total_frames: int = 300,
    seed: int = 42,
) -> Dict:
    """Run VTS benchmark on synthetic video data."""
    rng = np.random.RandomState(seed)
    results_per_video = []

    for i in range(n_videos):
        # Random scene boundaries
        n_boundaries = rng.randint(8, 20)
        boundaries = sorted(rng.choice(range(10, total_frames - 10), n_boundaries, replace=False))
        boundaries = [int(b) for b in boundaries]

        # Random target interval
        gt_start = rng.randint(0, total_frames - 50)
        gt_end = gt_start + rng.randint(10, 60)
        gt_end = min(gt_end, total_frames)

        # Build tree
        root = build_temporal_tree(total_frames, boundaries, max_depth=4)
        assign_relevance_scores(root, gt_start, gt_end, noise_std=0.05, seed=seed + i)

        # Run agent
        agent = VTSAgent(root, np.random.RandomState(seed + i + 1000))
        pred_start, pred_end = agent.navigate()

        iou = compute_interval_iou((pred_start, pred_end), (gt_start, gt_end))
        results_per_video.append({
            "video_id": i,
            "gt_interval": [gt_start, gt_end],
            "pred_interval": [pred_start, pred_end],
            "iou": round(iou, 4),
            "n_steps": len(agent.history),
        })

    mean_iou = np.mean([r["iou"] for r in results_per_video])
    mean_steps = np.mean([r["n_steps"] for r in results_per_video])

    results = {
        "paper": "2607.16189",
        "title": "VideoTreeSearch: Self-Correcting Agents for Grounded Long Video QA",
        "n_videos": n_videos,
        "mean_iou": round(float(mean_iou), 4),
        "mean_navigation_steps": round(float(mean_steps), 2),
        "max_iou": round(float(max(r["iou"] for r in results_per_video)), 4),
        "min_iou": round(float(min(r["iou"] for r in results_per_video)), 4),
    }
    return results


if __name__ == "__main__":
    results = run_vts_benchmark()
    print(json.dumps(results, indent=2))
    with open("/root/git/mimo/paper-pipeline/reproduction/ai_agent/results_ai_agent.json", "r") as f:
        all_results = json.load(f)
    all_results.append(results)
    with open("/root/git/mimo/paper-pipeline/reproduction/ai_agent/results_ai_agent.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results appended to results_ai_agent.json")

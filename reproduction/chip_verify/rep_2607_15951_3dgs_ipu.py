"""
Paper: 2607.15951 - Rendering 3D Gaussians on a Graph Processor
Authors: Nicholas Fry et al. (Eurographics 2026)

Reproduction: Simulates BSP-based 3D Gaussian splatting on a tiled IPU-like architecture.
Tests tile routing (NEWS grid), workload distribution, and SRAM-only rendering constraints.
"""
import numpy as np
import json
import time

np.random.seed(42)

# --- IPU-like Architecture Parameters ---
NUM_TILES = 1472         # IPU tile count
TILE_SRAM_KB = 256       # SRAM per tile in KB
FRAMEBUFFER_W = 1024
FRAMEBUFFER_H = 1024
TILE_W = 32              # Pixels per tile width
TILE_H = 32              # Pixels per tile height
GRID_COLS = FRAMEBUFFER_W // TILE_W  # 32
GRID_ROWS = FRAMEBUFFER_H // TILE_H  # 32
BSP_STAGES = 5           # Number of BSP phases


# --- 3D Gaussian Splatting Model ---
class Gaussian3D:
    def __init__(self, n_gaussians=10000):
        # Position (x, y, z)
        self.means = np.random.randn(n_gaussians, 3) * 2.0
        # Scale (3x3 covariance)
        self.scales = np.abs(np.random.randn(n_gaussians, 3)) * 0.1
        # Opacity
        self.opacities = np.clip(np.random.randn(n_gaussians) * 0.3 + 0.5, 0.01, 0.99)
        # Color (SH coefficients, simplified to RGB)
        self.colors = np.clip(np.random.randn(n_gaussians, 3) * 0.3 + 0.5, 0, 1)
        # Rotation quaternion (w, x, y, z)
        self.rotations = np.random.randn(n_gaussians, 4)
        norms = np.linalg.norm(self.rotations, axis=-1, keepdims=True)
        self.rotations = self.rotations / (norms + 1e-8)

    def project_to_2d(self, camera_pos=np.array([0, 0, -5.0])):
        """Simple orthographic projection to screen space."""
        # Translate
        pts = self.means - camera_pos
        # Project to screen (simplified)
        screen_x = (pts[:, 0] * 100 + FRAMEBUFFER_W / 2).astype(int)
        screen_y = (pts[:, 1] * 100 + FRAMEBUFFER_H / 2).astype(int)
        screen_z = pts[:, 2]
        # Compute 2D covariance (diagonal approximation)
        cov_2d = self.scales[:, :2] ** 2
        return screen_x, screen_y, screen_z, cov_2d


def manhattan_distance(pos_a, pos_b):
    """Manhattan distance on tile grid."""
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def get_tile坐标(screen_x, screen_y):
    """Map pixel coordinates to tile coordinates."""
    tile_x = np.clip(screen_x // TILE_W, 0, GRID_COLS - 1)
    tile_y = np.clip(screen_y // TILE_H, 0, GRID_ROWS - 1)
    return tile_x, tile_y


def route_gaussians_to_tiles(gaussians):
    """Route gaussians to destination tiles via NEWS grid hops."""
    screen_x, screen_y, screen_z, cov_2d = gaussians.project_to_2d()
    tile_x, tile_y = get_tile坐标(screen_x, screen_y)

    # Each gaussian's tile assignment
    tile_assignments = np.stack([tile_x, tile_y], axis=-1)

    # Compute hop counts (Manhattan distance from center tile)
    center_tile = np.array([GRID_COLS // 2, GRID_ROWS // 2])
    hop_counts = np.abs(tile_assignments - center_tile).sum(axis=-1)

    # Compute overlap: how many neighboring tiles each gaussian affects
    # Each gaussian spreads to neighboring tiles based on its 2D covariance
    spread_radii = np.sqrt(cov_2d.sum(axis=-1)) * 2  # 2-sigma radius in pixels
    spread_tiles = np.ceil(spread_radii / TILE_W).astype(int)

    return tile_assignments, hop_counts, spread_tiles


def simulate_bsp_phase(tiles, gaussians_per_tile, phase):
    """Simulate one BSP phase with inter-tile communication."""
    communication_volume = 0
    computation_flops = 0

    max_tile_id = max(gaussians_per_tile.keys()) + 1 if gaussians_per_tile else 0
    for tile_id in range(max_tile_id):
        n_local = gaussians_per_tile.get(tile_id, 0)
        # Each phase: exchange border data with neighbors
        neighbors = []
        tx, ty = tile_id % GRID_COLS, tile_id // GRID_COLS
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS:
                neighbors.append(ny * GRID_COLS + nx)

        # Communication: send/receive border gaussians
        border_gaussians = min(n_local, 50)  # Approximate border region
        communication_volume += border_gaussians * len(neighbors) * 64  # bytes per gaussian

        # Computation: alpha-blending for local gaussians
        computation_flops += n_local * 100  # FLOPS per gaussian

    return communication_volume, computation_flops


def simulate_sram_usage(gaussians_per_tile):
    """Estimate SRAM usage per tile."""
    # Each gaussian needs: position(12B) + covariance(8B) + color(12B) + opacity(4B) + tile_info(8B)
    GAUSSIAN_SIZE_BYTES = 44
    # Framebuffer portion
    FB_SIZE = TILE_W * TILE_H * 4  # RGBA
    # Sorting buffer
    SORT_BUFFER = 1024

    sram_per_tile = {}
    for tile_id, n_g in gaussians_per_tile.items():
        usage = n_g * GAUSSIAN_SIZE_BYTES + FB_SIZE + SORT_BUFFER
        sram_per_tile[tile_id] = usage

    return sram_per_tile


def workload_imbalance_metric(gaussians_per_tile):
    """Compute workload imbalance across tiles."""
    counts = np.array(list(gaussians_per_tile.values()), dtype=float)
    if len(counts) == 0:
        return 0.0
    return float(np.std(counts) / (np.mean(counts) + 1e-8))


def render_tile(gaussians_data, tile_id):
    """Render a single tile (simplified alpha-blending)."""
    n_g = len(gaussians_data)
    if n_g == 0:
        return np.zeros((TILE_H, TILE_W, 4))

    framebuffer = np.zeros((TILE_H, TILE_W, 4))

    # Sort by depth (painter's algorithm)
    depths = gaussians_data[:, 2] if gaussians_data.ndim > 1 else gaussians_data
    sort_idx = np.argsort(depths)[::-1]

    for idx in sort_idx:
        g = gaussians_data[idx] if gaussians_data.ndim > 1 else np.array([gaussians_data[idx]])
        # Simplified splatting: add gaussian contribution
        alpha = 0.3
        color = np.random.rand(3)  # Placeholder
        framebuffer[:, :, :3] = framebuffer[:, :, :3] * (1 - alpha) + color * alpha
        framebuffer[:, :, 3] = framebuffer[:, :, 3] + alpha * (1 - framebuffer[:, :, 3])

    return framebuffer


# --- Main experiments ---
print("=" * 60)
print("Reproduction: 2607.15951 - 3D Gaussian Rendering on Graph Processor")
print("=" * 60)

results = {
    "paper_id": "2607.15951",
    "title": "Rendering 3D Gaussians on a Graph Processor",
    "method": "BSP-model tiled rendering with NEWS-grid gaussian routing on IPU",
    "experiments": {}
}

# Experiment 1: Gaussian routing and tile distribution
print("\n[Exp 1] Gaussian Routing via NEWS Grid...")
gaussians = Gaussian3D(n_gaussians=5000)
tile_assignments, hop_counts, spread_tiles = route_gaussians_to_tiles(gaussians)

# Compute tile load distribution
unique_tiles, tile_counts = np.unique(tile_assignments, axis=0, return_counts=True)
gaussians_per_tile = {}
for i, tile in enumerate(unique_tiles):
    tile_id = tile[1] * GRID_COLS + tile[0]
    gaussians_per_tile[tile_id] = int(tile_counts[i])

print(f"  Total gaussians: 5000")
print(f"  Tiles with gaussians: {len(gaussians_per_tile)}")
print(f"  Max hop count: {np.max(hop_counts)}")
print(f"  Mean hop count: {np.mean(hop_counts):.1f}")
print(f"  Workload imbalance: {workload_imbalance_metric(gaussians_per_tile):.3f}")

results["experiments"]["gaussian_routing"] = {
    "total_gaussians": 5000,
    "tiles_activated": len(gaussians_per_tile),
    "total_tiles": GRID_COLS * GRID_ROWS,
    "max_hops": int(np.max(hop_counts)),
    "mean_hops": round(float(np.mean(hop_counts)), 1),
    "workload_imbalance": round(float(workload_imbalance_metric(gaussians_per_tile)), 3),
    "max_tile_load": int(max(gaussians_per_tile.values())),
    "min_tile_load": int(min(gaussians_per_tile.values())),
}

# Experiment 2: BSP communication and computation
print("\n[Exp 2] BSP Phase Communication Analysis...")
phase_communication = []
phase_computation = []

for phase in range(BSP_STAGES):
    comm, comp = simulate_bsp_phase(None, gaussians_per_tile, phase)
    phase_communication.append(comm)
    phase_computation.append(comp)
    print(f"  Phase {phase}: comm={comm / 1024:.1f} KB, comp={comp / 1e6:.2f} MFLOPS")

total_comm = sum(phase_communication)
total_comp = sum(phase_computation)

results["experiments"]["bsp_analysis"] = {
    "num_stages": BSP_STAGES,
    "communication_bytes_per_phase": phase_communication,
    "computation_flops_per_phase": phase_computation,
    "total_communication_KB": round(total_comm / 1024, 2),
    "total_computation_MFLOPS": round(total_comp / 1e6, 2),
    "comm_to_comp_ratio": round(total_comm / (total_comp + 1e-8), 6),
}

# Experiment 3: SRAM-only rendering constraint
print("\n[Exp 3] SRAM-Only Rendering Constraints...")
sram_usage = simulate_sram_usage(gaussians_per_tile)
sram_values = list(sram_usage.values())
max_sram = max(sram_values)
mean_sram = np.mean(sram_values)
tiles_over_budget = sum(1 for v in sram_values if v > TILE_SRAM_KB * 1024)

print(f"  Max SRAM usage: {max_sram / 1024:.1f} KB")
print(f"  Mean SRAM usage: {mean_sram / 1024:.1f} KB")
print(f"  SRAM budget per tile: {TILE_SRAM_KB} KB")
print(f"  Tiles exceeding budget: {tiles_over_budget}/{len(sram_usage)}")

results["experiments"]["sram_constraints"] = {
    "sram_per_tile_KB": TILE_SRAM_KB,
    "max_usage_KB": round(float(max_sram / 1024), 1),
    "mean_usage_KB": round(float(mean_sram / 1024), 1),
    "tiles_over_budget": tiles_over_budget,
    "total_tiles_used": len(sram_usage),
    "sram_utilization": round(float(mean_sram / (TILE_SRAM_KB * 1024)), 3),
}

# Experiment 4: Rendering performance (frame time estimate)
print("\n[Exp 4] Rendering Performance Estimate...")
t0 = time.time()
total_pixels = 0
for tile_id in range(min(100, len(gaussians_per_tile))):
    n_g = gaussians_per_tile.get(tile_id, 0)
    if n_g > 0:
        fb = render_tile(np.random.randn(n_g, 3), tile_id)
        total_pixels += TILE_W * TILE_H
render_time = time.time() - t0

fps_estimate = 100 / (render_time + 1e-6) * (GRID_COLS * GRID_ROWS / 100)
print(f"  Rendered {total_pixels} pixels in {render_time:.4f}s")
print(f"  Estimated FPS (full frame): {fps_estimate:.1f}")

results["experiments"]["rendering_performance"] = {
    "tiles_rendered": min(100, len(gaussians_per_tile)),
    "total_tiles": GRID_COLS * GRID_ROWS,
    "render_time_s": round(float(render_time), 4),
    "estimated_fps": round(float(fps_estimate), 1),
    "framebuffer_resolution": f"{FRAMEBUFFER_W}x{FRAMEBUFFER_H}",
}

# Experiment 5: Scalability analysis (varying gaussian count)
print("\n[Exp 5] Scalability: Gaussian Count vs Performance...")
scalability = []
for n_g in [1000, 5000, 10000, 30000]:
    g = Gaussian3D(n_gaussians=n_g)
    t_assign, t_hops, t_spread = route_gaussians_to_tiles(g)
    unique_rows, t_counts = np.unique(t_assign, axis=0, return_counts=True)
    g_per_t = {int(unique_rows[i][1] * GRID_COLS + unique_rows[i][0]): int(t_counts[i])
               for i in range(len(unique_rows))}

    imbalance = workload_imbalance_metric(g_per_t)
    max_load = max(g_per_t.values()) if g_per_t else 0
    sram_us = simulate_sram_usage(g_per_t)
    max_sram_usage = max(sram_us.values()) if sram_us else 0

    scalability.append({
        "n_gaussians": n_g,
        "tiles_activated": len(g_per_t),
        "workload_imbalance": round(float(imbalance), 3),
        "max_tile_load": int(max_load),
        "max_sram_KB": round(float(max_sram_usage / 1024), 1),
    })
    print(f"  N={n_g:6d}: tiles={len(g_per_t)}, imbalance={imbalance:.3f}, max_load={max_load}")

results["experiments"]["scalability"] = scalability

# Save results
output_path = "/root/git/mimo/paper-pipeline/reproduction/chip_verify/results_2607_15951.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print("Reproduction complete.")

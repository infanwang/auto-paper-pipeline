#!/usr/bin/env python3
"""
VTLoc Reproduction Experiment
Simulation-based verification of "VTLoc: Learning-based Tactile Contact Localization
in Visual Point Clouds" (arXiv:2607.16146)

Since we lack the ObjectFolder Real dataset, this experiment uses:
- Synthetic 3D objects (sphere, cube, cylinder, ellipsoid)
- Synthetic GelSight-like tactile images
- Simplified VTLoc architecture
- Baseline comparisons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, List, Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """Experiment configuration matching paper parameters"""
    # Data
    num_objects: int = 20
    points_per_object: int = 512
    tactile_size: Tuple[int, int] = (64, 64)
    num_contacts_per_object: int = 10
    
    # Training
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    
    # Model
    latent_dim: int = 128
    num_refine_iterations: int = 5  # N=5 as in paper
    
    # Evaluation
    success_threshold: float = 0.02  # 2cm threshold
    eval_refine_iterations: List[int] = (1, 3, 5, 10)
    
    # Experiment
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

# ============================================================
# Synthetic Data Generation
# ============================================================

class SyntheticObjectGenerator:
    """Generate synthetic 3D point clouds for simple shapes"""
    
    def __init__(self, num_points: int = 1024):
        self.num_points = num_points
    
    def generate_sphere(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Generate point cloud on sphere surface"""
        # Uniform sampling on sphere
        phi = np.random.uniform(0, 2 * np.pi, self.num_points)
        cos_theta = np.random.uniform(-1, 1, self.num_points)
        theta = np.arccos(cos_theta)
        
        r = radius + np.random.normal(0, 0.001, self.num_points)  # Add noise
        
        x = r * np.sin(theta) * np.cos(phi) + center[0]
        y = r * np.sin(theta) * np.sin(phi) + center[1]
        z = r * np.cos(theta) + center[2]
        
        return np.stack([x, y, z], axis=1)
    
    def generate_cube(self, center: np.ndarray, size: float) -> np.ndarray:
        """Generate point cloud on cube surface"""
        points = []
        half = size / 2
        
        for _ in range(self.num_points // 6):
            # Choose a random face
            face = np.random.randint(6)
            u, v = np.random.uniform(-half, half, 2)
            
            if face == 0:  # +x
                pt = [half, u, v]
            elif face == 1:  # -x
                pt = [-half, u, v]
            elif face == 2:  # +y
                pt = [u, half, v]
            elif face == 3:  # -y
                pt = [u, -half, v]
            elif face == 4:  # +z
                pt = [u, v, half]
            else:  # -z
                pt = [u, v, -half]
            
            # Add noise
            pt = np.array(pt) + np.random.normal(0, 0.001, 3)
            points.append(pt + center)
        
        # Pad to exact count
        points = np.array(points)
        if len(points) < self.num_points:
            idx = np.random.choice(len(points), self.num_points - len(points))
            points = np.vstack([points, points[idx]])
        elif len(points) > self.num_points:
            points = points[:self.num_points]
        
        return points
    
    def generate_cylinder(self, center: np.ndarray, radius: float, height: float) -> np.ndarray:
        """Generate point cloud on cylinder surface"""
        points = []
        half_h = height / 2
        num_side = self.num_points * 4 // 5
        num_ends = self.num_points - num_side
        
        # Side
        for _ in range(num_side):
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(-half_h, half_h)
            r = radius + np.random.normal(0, 0.001)
            
            x = r * np.cos(theta) + center[0]
            y = r * np.sin(theta) + center[1]
            z = z + center[2]
            points.append([x, y, z])
        
        # Ends
        for _ in range(num_ends):
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            z = half_h if np.random.random() > 0.5 else -half_h
            
            x = r * np.cos(theta) + center[0]
            y = r * np.sin(theta) + center[1]
            z = z + center[2]
            points.append([x, y, z])
        
        return np.array(points[:self.num_points])
    
    def generate_ellipsoid(self, center: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Generate point cloud on ellipsoid surface"""
        phi = np.random.uniform(0, 2 * np.pi, self.num_points)
        cos_theta = np.random.uniform(-1, 1, self.num_points)
        theta = np.arccos(cos_theta)
        
        x = radii[0] * np.sin(theta) * np.cos(phi) + center[0]
        y = radii[1] * np.sin(theta) * np.sin(phi) + center[1]
        z = radii[2] * np.cos(theta) + center[2]
        
        # Add noise
        noise = np.random.normal(0, 0.001, (self.num_points, 3))
        return np.stack([x, y, z], axis=1) + noise
    
    def generate_object(self, obj_type: str, **kwargs) -> np.ndarray:
        """Generate object of specified type"""
        center = kwargs.get('center', np.zeros(3))
        
        if obj_type == 'sphere':
            radius = kwargs.get('radius', 0.05)
            return self.generate_sphere(center, radius)
        elif obj_type == 'cube':
            size = kwargs.get('size', 0.1)
            return self.generate_cube(center, size)
        elif obj_type == 'cylinder':
            radius = kwargs.get('radius', 0.03)
            height = kwargs.get('height', 0.1)
            return self.generate_cylinder(center, radius, height)
        elif obj_type == 'ellipsoid':
            radii = kwargs.get('radii', np.array([0.06, 0.04, 0.03]))
            return self.generate_ellipsoid(center, radii)
        else:
            raise ValueError(f"Unknown object type: {obj_type}")


class SyntheticTactileGenerator:
    """Generate synthetic GelSight-like tactile images"""
    
    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.size = size
    
    def generate_tactile(self, contact_point: np.ndarray, contact_normal: np.ndarray,
                         depth: float = 0.002, force: float = 1.0) -> np.ndarray:
        """
        Generate synthetic GelSight-like tactile image
        
        Args:
            contact_point: 3D contact location
            contact_normal: surface normal at contact
            depth: indentation depth
            force: contact force
            
        Returns:
            Synthetic tactile image (H, W, 3)
        """
        h, w = self.size
        image = np.zeros((h, w, 3), dtype=np.float32)
        
        # Create circular contact region
        center_x, center_y = h // 2, w // 2
        radius = int(20 * depth / 0.005)  # Scale with depth
        
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # GelSight-style markers displacement
        mask = dist < radius
        
        # Create marker pattern
        marker_spacing = 8
        marker_Y, marker_X = np.meshgrid(
            np.arange(0, h, marker_spacing),
            np.arange(0, w, marker_spacing),
            indexing='ij'
        )
        
        # Base green channel (GelSight markers are typically on green)
        image[:, :, 1] = 0.3
        
        # Red and blue based on displacement
        displacement_x = np.zeros_like(image[:, :, 0])
        displacement_y = np.zeros_like(image[:, :, 0])
        
        # Contact region displacement
        if np.any(mask):
            # Simulate marker displacement from contact
            disp_magnitude = np.exp(-dist**2 / (2 * (radius/2)**2)) * depth * 100
            
            # Direction based on normal
            dx = contact_normal[0]
            dy = contact_normal[1]
            
            displacement_x = disp_magnitude * dx
            displacement_y = disp_magnitude * dy
            
            # Color channels
            image[:, :, 0] = 0.5 + displacement_x / (2 * depth * 100)
            image[:, :, 1] = 0.3 - disp_magnitude / (4 * depth * 100)
            image[:, :, 2] = 0.5 - displacement_y / (2 * depth * 100)
        
        # Add noise
        noise = np.random.normal(0, 0.02, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        # Add marker dots
        for my in range(0, h, marker_spacing):
            for mx in range(0, w, marker_spacing):
                if mask[my, mx]:
                    # Displaced marker
                    new_my = int(my + displacement_y[my, mx])
                    new_mx = int(mx + displacement_x[my, mx])
                    if 0 <= new_my < h and 0 <= new_mx < w:
                        cv2_circle_approx(image, new_mx, new_my, 1, [0.8, 0.8, 0.2])
                else:
                    # Static marker
                    cv2_circle_approx(image, mx, my, 1, [0.2, 0.8, 0.2])
        
        return image


def cv2_circle_approx(image, x, y, radius, color):
    """Simple circle drawing without opencv"""
    h, w = image.shape[:2]
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dx*dx + dy*dy <= radius*radius:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    image[ny, nx] = color


class VTLocDataset:
    """Dataset for VTLoc experiment"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.obj_generator = SyntheticObjectGenerator(config.points_per_object)
        self.tactile_generator = SyntheticTactileGenerator(config.tactile_size)
        self.object_types = ['sphere', 'cube', 'cylinder', 'ellipsoid']
        
        # Generate objects
        self.objects = self._generate_objects()
        self.data = self._generate_dataset()
    
    def _generate_objects(self) -> List[np.ndarray]:
        """Generate all synthetic objects"""
        objects = []
        for i in range(self.config.num_objects):
            obj_type = self.object_types[i % len(self.object_types)]
            
            # Random parameters
            center = np.random.uniform(-0.1, 0.1, 3)
            
            if obj_type == 'sphere':
                radius = np.random.uniform(0.02, 0.08)
                obj = self.obj_generator.generate_object(obj_type, center=center, radius=radius)
            elif obj_type == 'cube':
                size = np.random.uniform(0.04, 0.12)
                obj = self.obj_generator.generate_object(obj_type, center=center, size=size)
            elif obj_type == 'cylinder':
                radius = np.random.uniform(0.02, 0.05)
                height = np.random.uniform(0.05, 0.15)
                obj = self.obj_generator.generate_object(obj_type, center=center, radius=radius, height=height)
            else:  # ellipsoid
                radii = np.random.uniform(0.02, 0.08, 3)
                obj = self.obj_generator.generate_object(obj_type, center=center, radii=radii)
            
            objects.append(obj)
        
        return objects
    
    def _generate_dataset(self) -> List[Dict]:
        """Generate contact-tactile pairs"""
        data = []
        
        for obj_idx, obj in enumerate(self.objects):
            for _ in range(self.config.num_contacts_per_object):
                # Sample contact point from object surface
                point_idx = np.random.randint(0, len(obj))
                contact_point = obj[point_idx]
                
                # Compute approximate normal (towards center)
                center = obj.mean(axis=0)
                normal = contact_point - center
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                
                # Random depth and force
                depth = np.random.uniform(0.001, 0.005)
                force = np.random.uniform(0.5, 2.0)
                
                # Generate tactile image
                tactile = self.tactile_generator.generate_tactile(
                    contact_point, normal, depth, force
                )
                
                # Downsample tactile for efficiency
                tactile_small = np.array(tactile).transpose(2, 0, 1)  # C, H, W
                
                data.append({
                    'object_idx': obj_idx,
                    'point_cloud': obj.astype(np.float32),
                    'tactile': tactile_small.astype(np.float32),
                    'contact_point': contact_point.astype(np.float32),
                    'contact_normal': normal.astype(np.float32),
                    'depth': depth,
                    'force': force
                })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'point_cloud': torch.tensor(item['point_cloud']),
            'tactile': torch.tensor(item['tactile']),
            'contact_point': torch.tensor(item['contact_point'])
        }


# ============================================================
# Model Architecture
# ============================================================

class TactileEncoder(nn.Module):
    """ResNet-like encoder for tactile images"""
    
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        
        # Simplified ResNet-like architecture
        self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class PointNetEncoder(nn.Module):
    """PointNet-like encoder for point clouds"""
    
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, latent_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global max pooling
        x = x.max(dim=2)[0]
        return x


class GeometricMultiModalAlignment(nn.Module):
    """
    Geometric Multi-modal Alignment (GMA)
    Reconstructs pseudo-point cloud from fused features
    """
    
    def __init__(self, latent_dim: int = 128, num_points: int = 64):
        super().__init__()
        
        self.num_points = num_points
        
        # Point decoder
        self.point_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_points * 3)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Fused feature vector (B, latent_dim)
            
        Returns:
            pseudo_cloud: Reconstructed pseudo-point cloud (B, num_points, 3)
        """
        batch_size = features.shape[0]
        points = self.point_decoder(features)
        points = points.view(batch_size, self.num_points, 3)
        return points


class IterativeLocalizationUpdater(nn.Module):
    """
    Iterative Localization Updater (ILU)
    GRU-based iterative refinement of contact predictions
    """
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # GRU cell
        self.gru = nn.GRUCell(3 + latent_dim, hidden_dim)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        # Initial guess predictor
        self.initial_guess = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
    def forward(self, features: torch.Tensor, num_iterations: int = 5) -> torch.Tensor:
        """
        Args:
            features: Fused features (B, latent_dim)
            num_iterations: Number of refinement iterations
            
        Returns:
            refined_point: Final contact point prediction (B, 3)
        """
        batch_size = features.shape[0]
        
        # Initial guess
        current_point = self.initial_guess(features)
        
        # Initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_dim, device=features.device)
        
        # Iterative refinement
        for _ in range(num_iterations):
            # Concatenate current point with features
            gru_input = torch.cat([current_point, features], dim=1)
            
            # GRU update
            hidden = self.gru(gru_input, hidden)
            
            # Predict refinement
            delta = self.predictor(hidden)
            
            # Update point
            current_point = current_point + delta
        
        return current_point


class VTLocModel(nn.Module):
    """
    Simplified VTLoc model
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        
        self.config = config
        
        # Encoders
        self.tactile_encoder = TactileEncoder(config.latent_dim)
        self.pointnet_encoder = PointNetEncoder(config.latent_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
        # Geometric Multi-modal Alignment (GMA)
        self.gma = GeometricMultiModalAlignment(config.latent_dim)
        
        # Iterative Localization Updater (ILU)
        self.ilu = IterativeLocalizationUpdater(config.latent_dim)
        
    def forward(self, tactile: torch.Tensor, point_cloud: torch.Tensor,
                num_iterations: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            tactile: Tactile images (B, 3, H, W)
            point_cloud: Object point cloud (B, N, 3)
            num_iterations: Number of ILU iterations (default from config)
            
        Returns:
            Dictionary with predictions and intermediate outputs
        """
        if num_iterations is None:
            num_iterations = self.config.num_refine_iterations
        
        # Encode tactile
        tactile_feat = self.tactile_encoder(tactile)
        
        # Encode point cloud
        point_feat = self.pointnet_encoder(point_cloud)
        
        # Fuse features
        fused = torch.cat([tactile_feat, point_feat], dim=1)
        fused = self.fusion(fused)
        
        # GMA: reconstruct pseudo-point cloud
        pseudo_cloud = self.gma(fused)
        
        # ILU: iterative refinement
        refined_point = self.ilu(fused, num_iterations)
        
        return {
            'contact_point': refined_point,
            'pseudo_cloud': pseudo_cloud,
            'fused_features': fused
        }


class MLPBaseline(nn.Module):
    """Direct MLP baseline without iterative refinement"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        
        self.tactile_encoder = TactileEncoder(config.latent_dim)
        self.pointnet_encoder = PointNetEncoder(config.latent_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(config.latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, tactile: torch.Tensor, point_cloud: torch.Tensor) -> torch.Tensor:
        tactile_feat = self.tactile_encoder(tactile)
        point_feat = self.pointnet_encoder(point_cloud)
        
        fused = torch.cat([tactile_feat, point_feat], dim=1)
        return self.regressor(fused)


class CosineSimilarityBaseline(nn.Module):
    """Cosine similarity matching baseline"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        
        self.tactile_encoder = TactileEncoder(config.latent_dim)
        self.pointnet_encoder = PointNetEncoder(config.latent_dim)
        
        self.projector = nn.Linear(config.latent_dim, 64)
    
    def forward(self, tactile: torch.Tensor, point_cloud: torch.Tensor) -> torch.Tensor:
        batch_size = tactile.shape[0]
        
        tactile_feat = self.tactile_encoder(tactile)
        tactile_feat = self.projector(tactile_feat)
        
        point_feat = self.pointnet_encoder(point_cloud)
        point_feat = self.projector(point_feat)
        
        # Normalize
        tactile_feat = F.normalize(tactile_feat, dim=1)
        point_feat = F.normalize(point_feat, dim=1)
        
        # Cosine similarity
        similarity = torch.sum(tactile_feat * point_feat, dim=1, keepdim=True)
        
        # Return mean point cloud as prediction (simplified)
        mean_point = point_cloud.mean(dim=1)
        
        return mean_point * similarity


# ============================================================
# Loss Functions
# ============================================================

def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer Distance between point clouds
    
    Args:
        pred: Predicted points (B, N, 3)
        target: Target points (B, M, 3)
        
    Returns:
        Chamfer distance
    """
    B, N, _ = pred.shape
    M = target.shape[1]
    
    # Expand for pairwise distances
    pred_exp = pred.unsqueeze(2).expand(B, N, M, 3)
    target_exp = target.unsqueeze(1).expand(B, N, M, 3)
    
    # Compute distances
    distances = torch.sum((pred_exp - target_exp) ** 2, dim=3)
    
    # For each point in pred, find nearest in target
    min_dist_pred, _ = distances.min(dim=2)  # (B, N)
    
    # For each point in target, find nearest in pred
    min_dist_target, _ = distances.min(dim=1)  # (B, M)
    
    # Chamfer distance
    cd = min_dist_pred.mean(dim=1) + min_dist_target.mean(dim=1)
    
    return cd.mean()


def contact_loss(pred_point: torch.Tensor, gt_point: torch.Tensor) -> torch.Tensor:
    """L1 + L2 loss for contact point prediction"""
    l1_loss = F.l1_loss(pred_point, gt_point)
    l2_loss = F.mse_loss(pred_point, gt_point)
    return l1_loss + l2_loss


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor,
                    thresholds: List[float] = [0.01, 0.02, 0.05]) -> Dict[str, float]:
    """Compute evaluation metrics"""
    # L2 distance
    l2_dist = torch.norm(pred - gt, dim=1)
    
    metrics = {
        'l2_mean': l2_dist.mean().item(),
        'l2_std': l2_dist.std().item(),
        'l2_median': l2_dist.median().item(),
    }
    
    # Success rates at different thresholds
    for thresh in thresholds:
        success = (l2_dist < thresh).float().mean()
        metrics[f'success_rate_{thresh*100:.0f}cm'] = success.item()
    
    return metrics


# ============================================================
# Training
# ============================================================

def train_epoch(model: nn.Module, dataloader, optimizer, criterion, device,
                use_gma_loss: bool = True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        tactile = batch['tactile'].to(device)
        point_cloud = batch['point_cloud'].to(device)
        contact_point = batch['contact_point'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - handle both dict-returning models and simple models
        output = model(tactile, point_cloud)
        
        if isinstance(output, dict):
            pred_point = output['contact_point']
            
            # Compute losses
            point_loss = contact_loss(pred_point, contact_point)
            
            # Only add CD loss if pseudo_cloud is a valid point cloud (B, N, 3)
            pseudo_cloud = output.get('pseudo_cloud')
            if use_gma_loss and pseudo_cloud is not None and pseudo_cloud.dim() == 3 and pseudo_cloud.shape[2] == 3:
                cd_loss = chamfer_distance(pseudo_cloud, 
                                           contact_point.unsqueeze(1).expand(-1, pseudo_cloud.shape[1], -1))
                loss = point_loss + 0.5 * cd_loss
            else:
                loss = point_loss
        else:
            pred_point = output
            loss = contact_loss(pred_point, contact_point)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model: nn.Module, dataloader, device, num_iterations: int = None):
    """Evaluate model"""
    model.eval()
    all_metrics = []
    
    for batch in dataloader:
        tactile = batch['tactile'].to(device)
        point_cloud = batch['point_cloud'].to(device)
        contact_point = batch['contact_point'].to(device)
        
        # Handle models that return dict (VTLoc-like) or tensor directly
        # Try with num_iterations first; fall back without it
        try:
            output = model(tactile, point_cloud, num_iterations=num_iterations)
        except TypeError:
            output = model(tactile, point_cloud)
        
        if isinstance(output, dict):
            pred_point = output['contact_point']
        else:
            pred_point = output
        
        metrics = compute_metrics(pred_point, contact_point)
        all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment():
    """Run the full VTLoc reproduction experiment"""
    print("=" * 70)
    print("VTLoc Reproduction Experiment")
    print("Simulation-based verification")
    print("=" * 70)
    
    config = ExperimentConfig()
    device = torch.device(config.device)
    print(f"\nUsing device: {device}")
    
    # Create output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    print("\n[1/5] Generating synthetic dataset...")
    dataset = VTLocDataset(config)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    
    print(f"   Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Initialize models
    print("\n[2/5] Initializing models...")
    vtloc_model = VTLocModel(config).to(device)
    mlp_baseline = MLPBaseline(config).to(device)
    cosine_baseline = CosineSimilarityBaseline(config).to(device)
    
    print(f"   VTLoc parameters: {sum(p.numel() for p in vtloc_model.parameters()):,}")
    print(f"   MLP baseline parameters: {sum(p.numel() for p in mlp_baseline.parameters()):,}")
    print(f"   Cosine baseline parameters: {sum(p.numel() for p in cosine_baseline.parameters()):,}")
    
    # Train VTLoc
    print("\n[3/5] Training VTLoc model...")
    optimizer_vtloc = torch.optim.Adam(vtloc_model.parameters(), 
                                        lr=config.learning_rate,
                                        weight_decay=config.weight_decay)
    scheduler_vtloc = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_vtloc, T_max=config.num_epochs
    )
    
    vtloc_train_losses = []
    vtloc_val_metrics = []
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_epoch(vtloc_model, train_loader, optimizer_vtloc, 
                                  contact_loss, device)
        scheduler_vtloc.step()
        
        vtloc_train_losses.append(train_loss)
        
        # Validate
        if (epoch + 1) % 5 == 0:
            val_metrics = evaluate(vtloc_model, val_loader, device)
            vtloc_val_metrics.append((epoch, val_metrics))
            
            print(f"   Epoch {epoch+1}/{config.num_epochs} - "
                  f"Loss: {train_loss:.4f}, "
                  f"L2: {val_metrics['l2_mean']:.4f}, "
                  f"Success@2cm: {val_metrics['success_rate_2cm']:.2%}")
            
            # Save best model
            if val_metrics['l2_mean'] < best_val_loss:
                best_val_loss = val_metrics['l2_mean']
                torch.save(vtloc_model.state_dict(), output_dir / 'best_vtloc.pth')
    
    # Train baselines
    print("\n[4/5] Training baseline models...")
    
    # MLP baseline
    optimizer_mlp = torch.optim.Adam(mlp_baseline.parameters(),
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
    
    print("   Training MLP baseline...")
    for epoch in range(config.num_epochs // 3):
        train_loss = train_epoch(mlp_baseline, train_loader, optimizer_mlp,
                                  contact_loss, device)
        if (epoch + 1) % 5 == 0:
            print(f"   MLP Epoch {epoch+1}/{config.num_epochs//3} - Loss: {train_loss:.4f}")
    
    # Cosine baseline
    optimizer_cosine = torch.optim.Adam(cosine_baseline.parameters(),
                                         lr=config.learning_rate,
                                         weight_decay=config.weight_decay)
    
    print("   Training Cosine similarity baseline...")
    for epoch in range(config.num_epochs // 3):
        train_loss = train_epoch(cosine_baseline, train_loader, optimizer_cosine,
                                  contact_loss, device)
        if (epoch + 1) % 5 == 0:
            print(f"   Cosine Epoch {epoch+1}/{config.num_epochs//3} - Loss: {train_loss:.4f}")
    
    # Evaluate all models
    print("\n[5/5] Evaluating on test set...")
    
    # Load best VTLoc
    vtloc_model.load_state_dict(torch.load(output_dir / 'best_vtloc.pth'))
    
    # Evaluate VTLoc
    vtloc_metrics = evaluate(vtloc_model, test_loader, device)
    print("\n   VTLoc Model:")
    for k, v in vtloc_metrics.items():
        print(f"      {k}: {v:.4f}")
    
    # Evaluate MLP baseline
    mlp_metrics = evaluate(mlp_baseline, test_loader, device)
    print("\n   MLP Baseline:")
    for k, v in mlp_metrics.items():
        print(f"      {k}: {v:.4f}")
    
    # Evaluate Cosine baseline
    cosine_metrics = evaluate(cosine_baseline, test_loader, device)
    print("\n   Cosine Similarity Baseline:")
    for k, v in cosine_metrics.items():
        print(f"      {k}: {v:.4f}")
    
    # Ablation: ILU iterations
    print("\n   Ablation: ILU Iterations")
    ilu_results = {}
    for n_iter in config.eval_refine_iterations:
        iter_metrics = evaluate(vtloc_model, test_loader, device, num_iterations=n_iter)
        ilu_results[n_iter] = iter_metrics
        print(f"      N={n_iter}: L2={iter_metrics['l2_mean']:.4f}, "
              f"Success@2cm={iter_metrics['success_rate_2cm']:.2%}")
    
    # Ablation: GMA module
    print("\n   Ablation: GMA Module")
    # Train model without GMA - use a model that doesn't use GMA in forward pass
    vtloc_no_gma = VTLocModel(config).to(device)
    
    optimizer_no_gma = torch.optim.Adam(vtloc_no_gma.parameters(),
                                         lr=config.learning_rate)
    
    # Train without GMA loss (and modify forward to skip GMA)
    # We create a wrapper that bypasses GMA
    class VTLocNoGMA(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.tactile_encoder = base_model.tactile_encoder
            self.pointnet_encoder = base_model.pointnet_encoder
            self.fusion = base_model.fusion
            self.ilu = base_model.ilu
            self.latent_dim = base_model.config.latent_dim
        
        def forward(self, tactile, point_cloud, num_iterations=None):
            if num_iterations is None:
                num_iterations = 5
            
            tactile_feat = self.tactile_encoder(tactile)
            point_feat = self.pointnet_encoder(point_cloud)
            fused = torch.cat([tactile_feat, point_feat], dim=1)
            fused = self.fusion(fused)
            
            refined_point = self.ilu(fused, num_iterations=num_iterations)
            # Return dict for compatibility with evaluate function; use refined_point as pseudo_cloud placeholder
            return {'contact_point': refined_point, 'pseudo_cloud': refined_point.unsqueeze(1).expand(-1, 64, -1), 'fused_features': fused}
    
    vtloc_no_gma_wrapper = VTLocNoGMA(vtloc_no_gma).to(device)
    
    for epoch in range(config.num_epochs // 2):
        train_loss = train_epoch(vtloc_no_gma_wrapper, train_loader, optimizer_no_gma,
                                  contact_loss, device, use_gma_loss=False)
    
    no_gma_metrics = evaluate(vtloc_no_gma_wrapper, test_loader, device)
    print(f"      Without GMA: L2={no_gma_metrics['l2_mean']:.4f}")
    print(f"      With GMA: L2={vtloc_metrics['l2_mean']:.4f}")
    gma_improvement = (no_gma_metrics['l2_mean'] - vtloc_metrics['l2_mean']) / no_gma_metrics['l2_mean']
    print(f"      GMA improvement: {gma_improvement:.1%}")
    
    # Save results
    results = {
        'config': {
            'num_objects': config.num_objects,
            'points_per_object': config.points_per_object,
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'device': str(device)
        },
        'vtloc': vtloc_metrics,
        'mlp_baseline': mlp_metrics,
        'cosine_baseline': cosine_metrics,
        'ilu_ablation': {str(k): v for k, v in ilu_results.items()},
        'gma_ablation': {
            'without_gma': no_gma_metrics,
            'with_gma': vtloc_metrics,
            'improvement': gma_improvement
        },
        'training': {
            'vtloc_losses': vtloc_train_losses
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(vtloc_train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VTLoc Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    n_iters = list(ilu_results.keys())
    l2_vals = [ilu_results[n]['l2_mean'] for n in n_iters]
    plt.plot(n_iters, l2_vals, 'o-')
    plt.xlabel('Number of ILU Iterations')
    plt.ylabel('L2 Distance')
    plt.title('ILU Iterations Ablation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    models = ['VTLoc', 'MLP', 'Cosine']
    l2_scores = [vtloc_metrics['l2_mean'], mlp_metrics['l2_mean'], cosine_metrics['l2_mean']]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    plt.bar(models, l2_scores, color=colors)
    plt.ylabel('L2 Distance')
    plt.title('Model Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.json")
    print(f"  - figures.png")
    print(f"  - best_vtloc.pth")
    
    return results, vtloc_metrics, mlp_metrics, cosine_metrics, ilu_results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results, vtloc_m, mlp_m, cosine_m, ilu_r = run_experiment()

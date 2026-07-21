#!/usr/bin/env python3
"""
Reproduction: Handroid - Bridging Dexterous Hand and Humanoid (arXiv 2607.16187)

Core algorithms reproduced:
1. Anthropomorphic hand kinematics (20-DoF)
2. Humanoid lower-limb locomotion (12-DoF) via RL-inspired policy
3. Gait generation for bipedal walking

Scaled-down: Simulated environment (no real hardware), simplified dynamics.
"""

import numpy as np
import json
import time

# ─── Kinematic Model ────────────────────────────────────────────────────────
# Human hand: 5 fingers, each with MCP, PIP, DIP (thumb has CMC, MCP, IP)
# Total: 4 fingers × 3 joints + thumb × 4 joints + 5 finger abduction = 20 DoF

class DexterousHandKinematics:
    """Forward kinematics for a 20-DoF anthropomorphic hand."""

    # Link lengths (meters) based on human hand proportions
    FINGER_LENGTHS = {
        'index':  [0.04, 0.025, 0.02],   # MCP-PIP, PIP-DIP, DIP-tip
        'middle': [0.045, 0.028, 0.022],
        'ring':   [0.042, 0.026, 0.021],
        'pinky':  [0.035, 0.022, 0.018],
        'thumb':  [0.03, 0.03, 0.025],    # CMC-MCP, MCP-IP, IP-tip
    }

    FINGER_ABDUCTION_INDICES = [3, 6, 9, 12, 15]  # abduction for each finger

    def __init__(self):
        self.n_dofs = 20
        self.n_fingers = 5
        self.joint_limits = np.array([
            [-10, 90] for _ in range(20)  # degrees
        ])

    def forward_kinematics(self, joint_angles_deg):
        """
        Compute fingertip positions given joint angles.

        Args:
            joint_angles_deg: (20,) array of joint angles in degrees
        Returns:
            fingertip_positions: (5, 3) array of (x, y, z) for each fingertip
        """
        angles = np.deg2rad(joint_angles_deg)
        fingertips = np.zeros((5, 3))

        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, name in enumerate(finger_names):
            lengths = self.FINGER_LENGTHS[name]
            # Simple planar chain in the palm plane
            cumulative_angle = 0.0
            pos = np.zeros(3)

            # Finger abduction (z-offset)
            abduction_angle = angles[i * 4] if i < 4 else angles[0]
            pos[2] = abduction_angle * 0.001  # small z-offset per degree

            # Joint angles for this finger
            if name == 'thumb':
                joint_angles = angles[i*4:(i+1)*4]
            else:
                start = 4 + (i-1) * 3
                joint_angles = angles[start:start+3]

            for j, (angle, length) in enumerate(zip(joint_angles, lengths)):
                cumulative_angle += angle
                pos[0] += length * np.cos(cumulative_angle)
                pos[1] += length * np.sin(cumulative_angle)

            fingertips[i] = pos

        return fingertips


class HumanoidLocomotion:
    """
    12-DoF lower-limb locomotion controller (simplified RL policy).

    Joint layout: 6 per leg × 2 legs
    Hip: 3 DoF (flexion, abduction, rotation)
    Knee: 1 DoF (flexion)
    Ankle: 2 DoF (flexion, inversion)
    """

    def __init__(self, dt=0.02):
        self.dt = dt
        self.n_joints = 12
        self.step_length = 0.15  # meters
        self.stance_duration = 0.4  # seconds
        self.swing_duration = 0.3  # seconds
        self.step_height = 0.05  # meters

        # Simplified policy weights (proportional controller + sinusoidal gait)
        self.kp = np.array([20.0, 15.0, 5.0, 30.0, 10.0, 8.0] * 2)  # P gains
        self.kd = np.array([5.0, 4.0, 2.0, 8.0, 3.0, 2.0] * 2)       # D gains
        self.gait_phase = 0.0

    def generate_gait_pattern(self, t):
        """
        Generate sinusoidal gait reference trajectory.

        Returns joint reference angles (rad) for both legs.
        """
        omega = 2 * np.pi / (self.stance_duration + self.swing_duration)
        phase_l = omega * t
        phase_r = omega * t + np.pi  # half-cycle offset

        refs = np.zeros(self.n_joints)

        # Left leg (indices 0-5)
        refs[0] = 0.3 * np.sin(phase_l)       # hip flexion
        refs[1] = 0.1 * np.abs(np.sin(phase_l))  # hip abduction
        refs[2] = 0.05 * np.sin(phase_l)      # hip rotation
        refs[3] = 0.5 * max(0, np.sin(phase_l))  # knee flexion (stance)
        refs[4] = 0.2 * np.sin(phase_l + 0.5)  # ankle flexion
        refs[5] = 0.05 * np.sin(phase_l)      # ankle inversion

        # Right leg (indices 6-11), phase-shifted
        refs[6] = 0.3 * np.sin(phase_r)
        refs[7] = 0.1 * np.abs(np.sin(phase_r))
        refs[8] = 0.05 * np.sin(phase_r)
        refs[9] = 0.5 * max(0, np.sin(phase_r))
        refs[10] = 0.2 * np.sin(phase_r + 0.5)
        refs[11] = 0.05 * np.sin(phase_r)

        return refs

    def compute_torques(self, state, reference):
        """PD controller for joint torque computation."""
        error = reference - state['joint_angles']
        d_error = -state['joint_velocities']
        torques = self.kp * error + self.kd * d_error
        return np.clip(torques, -50, 50)

    def simulate_step(self, state, t):
        """Simulate one control step."""
        ref = self.generate_gait_pattern(t)
        torques = self.compute_torques(state, ref)

        # Simplified dynamics: angular acceleration = torque / inertia
        inertia = np.array([0.1, 0.08, 0.05, 0.15, 0.03, 0.02] * 2)
        accel = torques / inertia
        new_vel = state['joint_velocities'] + accel * self.dt
        new_angle = state['joint_angles'] + new_vel * self.dt

        return {
            'joint_angles': new_angle,
            'joint_velocities': new_vel,
            'torques': torques,
            'reference': ref,
        }


def run_hand_kinematics():
    """Test dexterous hand kinematics."""
    hand = DexterousHandKinematics()

    # Grasping configuration: fingers curled around a sphere
    grasp_config = np.zeros(20)
    grasp_config[4:7] = np.deg2rad([60, 70, 50])    # index
    grasp_config[7:10] = np.deg2rad([65, 75, 55])   # middle
    grasp_config[10:13] = np.deg2rad([55, 65, 45])  # ring
    grasp_config[13:16] = np.deg2rad([50, 60, 40])  # pinky
    grasp_config[0:4] = np.deg2rad([45, 55, 50, 30]) # thumb

    fingertips = hand.forward_kinematics(grasp_config)

    # Compute grasp quality metrics
    palm_center = np.mean(fingertips, axis=0)
    distances = np.linalg.norm(fingertips - palm_center, axis=1)
    grasp_volume = np.prod(distances)  # proxy for grasp quality

    return {
        'fingertip_positions': fingertips.tolist(),
        'palm_center': palm_center.tolist(),
        'finger_distances': distances.tolist(),
        'grasp_volume': float(grasp_volume),
    }


def run_locomotion(n_steps=200):
    """Simulate humanoid locomotion for n_steps."""
    humanoid = HumanoidLocomotion(dt=0.02)
    state = {
        'joint_angles': np.zeros(12),
        'joint_velocities': np.zeros(12),
    }

    trajectory = []
    for step in range(n_steps):
        t = step * humanoid.dt
        state = humanoid.simulate_step(state, t)
        trajectory.append({
            't': t,
            'angles': state['joint_angles'].copy(),
            'velocities': state['joint_velocities'].copy(),
        })

    # Compute gait metrics
    angles = np.array([traj['angles'] for traj in trajectory])
    velocities = np.array([traj['velocities'] for traj in trajectory])

    # Forward progress estimate (hip flexion integration)
    forward_disp = np.sum(angles[:, 0] * humanoid.dt) * humanoid.step_length

    # Symmetry metric: difference between left and right hip flexion
    symmetry_error = np.mean(np.abs(angles[:, 0] - angles[:, 6]))

    # Energy proxy
    energy = np.sum(np.abs(velocities)) * humanoid.dt

    return {
        'n_steps': n_steps,
        'duration': n_steps * humanoid.dt,
        'forward_displacement_m': float(forward_disp),
        'symmetry_error': float(symmetry_error),
        'energy_proxy': float(energy),
        'joint_trajectory_shape': list(angles.shape),
    }


if __name__ == '__main__':
    results = {}

    print("=== Handroid Reproduction ===")
    print("1. Dexterous Hand Kinematics")
    hand_results = run_hand_kinematics()
    results['hand_kinematics'] = hand_results
    print(f"   Grasp volume: {hand_results['grasp_volume']:.6f}")

    print("2. Humanoid Locomotion")
    loco_results = run_locomotion(n_steps=200)
    results['locomotion'] = loco_results
    print(f"   Duration: {loco_results['duration']:.1f}s")
    print(f"   Forward displacement: {loco_results['forward_displacement_m']:.3f}m")
    print(f"   Symmetry error: {loco_results['symmetry_error']:.4f} rad")

    out_path = '/root/git/mimo/paper-pipeline/reproduction/code_gen/results_handroid.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

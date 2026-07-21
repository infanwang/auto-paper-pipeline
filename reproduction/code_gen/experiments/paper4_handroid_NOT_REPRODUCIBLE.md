# Paper 4: Handroid (2607.16187) - NOT REPRODUCIBLE

## Paper
Handroid: Bridging Dexterous Hand and Humanoid
Authors: Ruogu Li, Chenyang Ma, et al. (UNC Chapel Hill, Stanford)

## Why Not Reproducible

This paper presents a **physical dual-embodiment robot** (27-DoF, 0.33m height, 2.05kg).
All experiments require:

1. **Physical hardware**: Custom 27-DoF electromechanical robot with Dynamixel actuators,
   custom mainboard, IMU sensors, and reconfigurable sliding mechanisms
2. **VR teleoperation**: Apple Vision Pro-based teleoperation interface
3. **Real-world manipulation**: Physical grasping, in-hand reorientation, locomotion
4. **Franka robot arm**: External arm used in manipulation experiments

The paper validates through:
- Real-world dexterous manipulation (teleoperation, grasping 10 objects, in-hand cube reorientation)
- Real-world humanoid locomotion (RL-based walking, keyframe motions)
- Long-horizon cross-embodiment task (reconfiguration + locomotion + docking + pick-and-place)

## What Could Be Simulated (Partial)

The **control algorithms** (RL tracking, velocity control, diffusion policy for grasping)
could be simulated in MuJoCo, but:
- The specific Handroid robot model is not publicly available
- The paper provides no open-source simulation code (only the project website android.org)
- Simulation results would not match the paper's real-robot results

## Conclusion
Fully blocked on physical hardware. Cannot reproduce any experimental results.

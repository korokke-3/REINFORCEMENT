# Unitree Go2 Reinforcement Learning & Dynamic Locomotion Research

This repository contains reinforcement learning environments, physical dynamics analyses, and control algorithms for the **Unitree Go2 quadruped robot** implemented on NVIDIA Isaac Lab / Isaac Sim.

---

## 📁 Repository Structure

* **`go2_single_leg/`**: Single-leg explosive dynamic jumping & hopping RL environment (IsaacLab + RSL-RL / PPO).
* **`go2_hopping/`**: 3-legged hopping locomotion environment (Successfully trained, 3.62 m in 6s).
* **`go2_right_side/`**: Right-side 2-legged bipedal running environment (Successfully trained, 3.74 m in 6s).
* **`go2_bipedal/`**: Hind-leg 2-legged standing and walking environment.
* **Physics & Validation Scripts**:
  * `eval_explosive_jump.py`: Evaluation script for trained true explosive single-leg jumping policy.
  * `test_strict_single_leg_knee.py`: Absolute zero-tolerance multi-link clearance validation.
  * `opt_knee_foot_balance.py`: Numerical optimizer for exact single-leg line-support CoM alignment.

---

## 🔬 Single-Leg True Explosive Jump Research Summary

### 1. Dynamic True Explosive Jumping Performance
* **Max Foot Clearance in Flight**: **$+35.4\text{ cm}$** ground clearance (Genuine ballistic jump).
* **Max Vertical Launch Velocity ($V_z$)**: **$+1.28\text{ m/s}$** explosive vertical push-off.
* **Airborne Ratio**: **$94.0\%$** of entire evaluation time in full ballistic flight ($3.76\text{ s}$ out of $4.00\text{ s}$).
* **Strict 3-Leg Airborne Clearance**: Other 3 legs (FL, FR, RL) consistently tucked at $+12.6\text{ cm}$ to $+33.7\text{ cm}$ altitude with **zero floor contact**.
* **Continuous Jump Survival**: Reached **$1.42\text{ s}$ ($71\text{ steps}$)** per episode.

---

## 🚀 How to Train & Evaluate

### Prerequisites
* Ubuntu 22.04 / 24.04
* NVIDIA GPU (RTX 4060 or higher) with CUDA 12+
* Isaac Lab 3.0+ & RSL-RL

### Run Training
```bash
conda activate isaaclab
cd go2_single_leg
python train.py --num_envs 4096 --max_iterations 1000
```

### Play / Visualize Trained Explosive Jump Policy
```bash
python eval_explosive_jump.py
```

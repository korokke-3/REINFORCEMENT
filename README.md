# Unitree Go2 Reinforcement Learning & Dynamic Locomotion Research

This repository contains reinforcement learning environments, physical dynamics analyses, and control algorithms for the **Unitree Go2 quadruped robot** implemented on NVIDIA Isaac Lab / Isaac Sim.

---

## 📁 Repository Structure

* **`go2_single_leg/`**: Single-leg active dynamic forward hopping RL environment (IsaacLab + RSL-RL / PPO).
* **`go2_hopping/`**: 3-legged hopping locomotion environment (Successfully trained, 3.62 m in 6s).
* **`go2_right_side/`**: Right-side 2-legged bipedal running environment (Successfully trained, 3.74 m in 6s).
* **`go2_bipedal/`**: Hind-leg 2-legged standing and walking environment.
* **Physics & Validation Scripts**:
  * `opt_knee_foot_balance.py`: Numerical optimizer for exact single-leg line-support CoM alignment.
  * `test_strict_single_leg_knee.py`: Absolute zero-tolerance multi-link clearance validation.
  * `eval_true_hopping.py`: Evaluation script for trained true single-leg forward hopping policy.

---

## 🔬 Single-Leg Locomotion & Dynamics Research Summary

### 1. Actuator Dynamics & Forward Thrust
* **Single Leg Peak Forward Thrust**: Successfully accelerates up to $V_x = +1.54\text{ m/s}$ ($5.5\text{ km/h}$) using single hind-leg push-off.
* **Strict 3-Leg Airborne Clearance**: Other 3 legs (FL, FR, RL) consistently maintained at $+8\text{ cm}$ to $+23\text{ cm}$ altitude with zero floor contact.
* **Forward Single-Leg Travel**: Reached $\approx 0.50\text{ m}$ continuous travel per episode.

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

### Play / Visualize Trained Policy
```bash
python eval_true_hopping.py
```

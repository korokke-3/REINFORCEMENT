# Unitree Go2 Reinforcement Learning & Dynamic Locomotion Research

This repository contains reinforcement learning environments, physical dynamics analyses, and control algorithms for the **Unitree Go2 quadruped robot** implemented on NVIDIA Isaac Lab / Isaac Sim.

---

## 📁 Repository Structure

* **`success_archive_5s_continuous_jump/`**: **[★ 5.00s Full Completion Master Archive]** (Benchmark models, videos, scripts).
* **`go2_single_leg/`**: Single-leg dynamic continuous explosive hopping & jumping RL environment (IsaacLab + RSL-RL / PPO).
* **`go2_hopping/`**: 3-legged hopping locomotion environment (Successfully trained, 3.62 m in 6s).
* **`go2_right_side/`**: Right-side 2-legged bipedal running environment (Successfully trained, 3.74 m in 6s).
* **`go2_bipedal/`**: Hind-leg 2-legged standing and walking environment.
* **Physics & Validation Scripts**:
  * `eval_5000_iter_master.py`: Evaluation script for extended 5000-iteration single-leg continuous jump master.
  * `eval_continuous_jump_master.py`: 2500-iteration master jump evaluation script.
  * `analyze_convergence.py`: Learning curve and convergence analysis script.

---

## 🔬 Single-Leg True Continuous Jump Grand Master Results (5,000 Iterations)

### 1. Benchmark Achievements
* **Max Continuous Single-Leg Survival**: **$5.00\text{ s}$ ($250\text{ steps}$ / Full Episode Completion with Zero Resets)**.
* **Max Foot Ground Clearance**: **$+51.6\text{ cm}$** ground clearance in flight (Over half-meter ballistic launch).
* **Max Vertical Launch Velocity ($V_z$)**: **$+1.02\text{ m/s}$**.
* **Airborne Flight Ratio**: **$98.8\%$** of entire evaluation time in full ballistic flight ($4.94\text{ s}$ out of $5.00\text{ s}$).
* **Strict 3-Leg Airborne Clearance**: Other 3 legs (FL, FR, RL) consistently tucked at $+11.4\text{ cm}$ to $+46.7\text{ cm}$ altitude with **zero floor contact**.

---

## 🚀 How to Train & Evaluate

### Prerequisites
* Ubuntu 22.04 / 24.04
* NVIDIA GPU (RTX 4060 or higher) with CUDA 12+
* Isaac Lab 3.0+ & RSL-RL

### Play 5000-Iteration Grand Master Policy
```bash
conda activate isaaclab
python eval_5000_iter_master.py
```

# Unitree Go2 Reinforcement Learning & Dynamic Locomotion Research

This repository contains reinforcement learning environments, physical dynamics analyses, and control algorithms for the **Unitree Go2 quadruped robot** implemented on NVIDIA Isaac Lab / Isaac Sim.

---

## 📁 Repository Structure

* **`success_archive_5s_continuous_jump/`**: **[★ Full Completion Master Archive]** (2500, 5000, 7400 iter models, videos, scripts).
* **`go2_single_leg/`**: Single-leg dynamic continuous explosive hopping & jumping RL environment (IsaacLab + RSL-RL / PPO).
* **`go2_hopping/`**: 3-legged hopping locomotion environment (Successfully trained, 3.62 m in 6s).
* **`go2_right_side/`**: Right-side 2-legged bipedal running environment (Successfully trained, 3.74 m in 6s).
* **`go2_bipedal/`**: Hind-leg 2-legged standing and walking environment.
* **Physics & Validation Scripts**:
  * `eval_7400_iter_master.py`: Evaluation script for extended 7400-iteration single-leg continuous jump master.
  * `eval_5000_iter_master.py`: 5000-iteration master jump evaluation script.
  * `eval_continuous_jump_master.py`: 2500-iteration master jump evaluation script.
  * `analyze_convergence.py`: Learning curve and convergence analysis script.

---

## 🔬 Single-Leg True Continuous Jump Grand Master Results (7,400 Iterations)

### 1. Benchmark Achievements Across Extended Training
| Training Stage | Continuous Survival | Max Foot Clearance | Launch Velocity ($V_z$) | Airborne Ratio | 5s Completion Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (1,000 Iterations)** | $1.42\text{ s}$ | $+35.4\text{ cm}$ | $+1.28\text{ m/s}$ | $94.0\%$ | $7.4\%$ |
| **Stage 2 (2,500 Iterations)** | **$5.00\text{ s}$ (Zero Reset)** | $+41.3\text{ cm}$ | $+0.97\text{ m/s}$ | $99.2\%$ | $26.5\%$ |
| **Stage 3 (5,000 Iterations)** | **$5.00\text{ s}$ (Zero Reset)** | **$+51.6\text{ cm}$** | $+1.02\text{ m/s}$ | $98.8\%$ | $28.9\%$ |
| **Stage 4 (7,400 Iterations)** | **$5.00\text{ s}$ (Zero Reset)** | $+43.7\text{ cm}$ | **$+1.23\text{ m/s}$** | **$98.8\%$** | **$37.8\%$ (Peak)** |

* **Strict 3-Leg Airborne Clearance**: Other 3 legs (FL, FR, RL) consistently tucked at $+14.5\text{ cm}$ to $+45.6\text{ cm}$ altitude with **zero floor contact**.

---

## 🚀 How to Train & Evaluate

### Prerequisites
* Ubuntu 22.04 / 24.04
* NVIDIA GPU (RTX 4060 or higher) with CUDA 12+
* Isaac Lab 3.0+ & RSL-RL

### Play 7400-Iteration Peak Policy
```bash
conda activate isaaclab
python eval_7400_iter_master.py
```

# Unitree Go2 Reinforcement Learning & Dynamic Locomotion Research

This repository contains reinforcement learning environments, physical dynamics analyses, and control algorithms for the **Unitree Go2 quadruped robot** implemented on NVIDIA Isaac Lab / Isaac Sim.

---

## 📁 Repository Structure

* **`success_archive_5s_continuous_jump/`**: **[★ 5.00s Full Completion Master Archive]** (2500 iter benchmark model, video, scripts).
* **`go2_single_leg/`**: Single-leg pure toe-point dynamic explosive jumping RL environment.
* **`go2_hopping/`**: 3-legged hopping locomotion environment (Successfully trained, 3.62 m in 6s).
* **`go2_right_side/`**: Right-side 2-legged bipedal running environment (Successfully trained, 3.74 m in 6s).
* **`go2_bipedal/`**: Hind-leg 2-legged standing and walking environment.
* **Physics & Validation Scripts**:
  * `eval_pure_toe_jump.py`: Evaluation script for pure toe single-leg jumping policy (Zero knee contact).
  * `eval_continuous_jump_master.py`: 5.00s master jump evaluation script.
  * `analyze_convergence.py`: Learning curve and convergence analysis script.

---

## 🔬 Pure Toe Single-Leg Jump (Knee Contact Prohibited) Research

### 1. Pure Toe Dynamic Jump Performance
* **Max Foot Ground Clearance in Flight**: **$+28.9\text{ cm}$** ground clearance.
* **Airborne Flight Ratio**: **$87.5\%$** in flight ($3.50\text{ s}$ out of $4.00\text{ s}$).
* **Knee Clearance in Flight**: Maintained at **$+8.6\text{ cm}$ to $+9.8\text{ cm}$** altitude with zero floor contact.
* **Physical Insight**: Dynamic single toe launch is verified achievable (+28.9cm launch). Current touchdown compliance requires slightly stiffer leg damping on ground impact to prevent knee over-flexion.

---

## 🚀 How to Train & Evaluate

### Prerequisites
* Ubuntu 22.04 / 24.04
* NVIDIA GPU (RTX 4060 or higher) with CUDA 12+
* Isaac Lab 3.0+ & RSL-RL

### Play 5.00s Continuous Jump Master Policy
```bash
conda activate isaaclab
cd success_archive_5s_continuous_jump
python eval_continuous_jump_master.py
```

### Play Pure Toe Jump Policy
```bash
python eval_pure_toe_jump.py
```

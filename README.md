# Unitree Go2 Reinforcement Learning & Dynamic Locomotion Research

This repository contains reinforcement learning environments, physical dynamics analyses, and control algorithms for the **Unitree Go2 quadruped robot** implemented on NVIDIA Isaac Lab / Isaac Sim.

---

## 📁 Repository Structure

* **`go2_single_leg/`**: Single-leg dynamic continuous explosive hopping & jumping RL environment (IsaacLab + RSL-RL / PPO).
* **`go2_hopping/`**: 3-legged hopping locomotion environment (Successfully trained, 3.62 m in 6s).
* **`go2_right_side/`**: Right-side 2-legged bipedal running environment (Successfully trained, 3.74 m in 6s).
* **`go2_bipedal/`**: Hind-leg 2-legged standing and walking environment.
* **Physics & Validation Scripts**:
  * `eval_continuous_jump_master.py`: Evaluation script for extended 2500-iteration continuous single-leg jump policy.
  * `eval_explosive_jump.py`: Single-leg explosive jump evaluation.
  * `test_strict_single_leg_knee.py`: Absolute zero-tolerance multi-link clearance validation.

---

## 🔬 Single-Leg True Continuous Jump Master Results (2,500 Iterations)

### 1. Benchmark Achievements
* **Max Continuous Single-Leg Survival**: **$5.00\text{ s}$ ($250\text{ steps}$ / Full Episode Completion with Zero Resets)**.
* **Max Foot Ground Clearance**: **$+41.3\text{ cm}$** ground clearance in flight.
* **Airborne Flight Ratio**: **$99.2\%$** of entire evaluation time in full ballistic flight ($4.96\text{ s}$ out of $5.00\text{ s}$).
* **Strict 3-Leg Airborne Clearance**: Other 3 legs (FL, FR, RL) consistently tucked at $+15.6\text{ cm}$ to $+41.4\text{ cm}$ altitude with **zero floor contact**.
* **Cyclic Re-bounding**: Seamless multi-hop cycle (Launch $\to$ Flight $\to$ Touchdown $\to$ Instant Re-launch).

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
python train.py --num_envs 4096 --max_iterations 2500
```

### Play / Visualize Continuous Jump Master Policy
```bash
python eval_continuous_jump_master.py
```

# Unitree Go2 Reinforcement Learning & Dynamic Locomotion Research

This repository contains reinforcement learning environments, physical dynamics analyses, and control algorithms for the **Unitree Go2 quadruped robot** implemented on NVIDIA Isaac Lab / Isaac Sim.

---

## 📁 Repository Structure

* **`go2_single_leg/`**: Single-leg active dynamic hopping & balancing RL environment (IsaacLab + RSL-RL / PPO).
* **`go2_hopping/`**: 3-legged hopping locomotion environment (Successfully trained, 3.62 m in 6s).
* **`go2_right_side/`**: Right-side 2-legged bipedal running environment (Successfully trained, 3.74 m in 6s).
* **`go2_bipedal/`**: Hind-leg 2-legged standing and walking environment.
* **Physics & Validation Scripts**:
  * `opt_balance.py`: Numerical optimizer for exact CoM (Center of Mass) vertical alignment.
  * `test_true_vertical_jump.py`: Momentum-transfer inertial assist jump test.
  * `compare_torque_limits.py`: Actuator torque limit (23.5Nm vs 45Nm vs 80Nm) comparative analysis.
  * `eval_active_hopping.py`: Evaluation script for trained active continuous hopping policy.

---

## 🔬 Single-Leg Jump & Active Hopping Research Summary

### 1. Actuator Dynamics & Vertical Lift Limits
* **Total Robot Mass**: $14.47\text{ kg}$ (Gravity: $142\text{ N}$)
* **Single Leg Peak Vertical Thrust**: $\approx 156\text{ N}$ (Torque limit: $23.5\text{ Nm}$)
* **Net Accelerating Force**: $156\text{ N} - 142\text{ N} = 14\text{ N}$ ($\approx 0.1G$)
* **Conclusion**: With point-contact rubber feet and a single leg carrying the entire 14.5 kg body, jumping height is limited to $\approx 2\text{--}5\text{ cm}$ unless assisted by multi-leg cyclic inertial pumping (Raibert hopper dynamic principle).

### 2. Precise CoM-Foot Alignment (Flamingo Stance)
* **Optimal Roll**: $+28.50^\circ$
* **Optimal Pitch**: $-30.50^\circ$
* **Horizontal CoM-Foot Offset**: $1.47\text{ mm}$ (Eliminates gravitational toppling torque at spawn).

### 3. Active Hopping Policy Results (Trained on IsaacLab PPO)
* **Airborne Ratio**: $\approx 40\%$ of evaluation time in active ballistic flight.
* **Foot Clearance in Flight**: Up to $+13.9\text{ cm}$ ground clearance.
* **Continuous Resilient Hopping**: AI successfully learned to push off the ground, adjust foot placement in mid-air, and absorb touchdown to counteract toppling moments.

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
python eval_active_hopping.py
```

# 🏆 Success Archive: 5.00s Continuous Single-Leg Explosive Jump Master

* **Date Achieved**: 2026-08-25
* **Benchmark Performance**:
  * **Continuous Survival**: **5.00 seconds (250 steps / Full Episode Completion with Zero Resets)**
  * **Foot Flight Ground Clearance**: **+41.3 cm**
  * **Airborne Ratio**: **99.2% (4.96s in flight / 5.00s total)**
  * **Other 3 Legs Clearance**: **+15.6 cm to +41.4 cm (Zero Contact)**
  * **Peak Mean Reward**: **+201.35**

## Archived Files
* `best_5s_jump_model.pt`: Saved PyTorch policy weights (2,498 iterations).
* `eval_continuous_jump_master.py`: Standalone evaluation script.
* `go2_single_leg_env_cfg_backup.py`: Environment configuration backup.
* `go2_single_leg_rewards_backup.py`: Reward function backup.
* `continuous_jump_master-step-0.mp4`: Recorded 5.00s validation video.

## How to Play this Benchmark Model
```bash
conda activate isaaclab
cd /home/exhibition-spakona/Desktop/REINFORCEMENT/success_archive_5s_continuous_jump
python eval_continuous_jump_master.py
```

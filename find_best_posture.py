import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import envs
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env_cfg.terminations.illegal_parts_contact = None
env_cfg.terminations.forward_fall = None
env_cfg.terminations.root_height_below_minimum = None
env_cfg.terminations.base_contact = None

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)
obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

masses = robot.data.default_mass[0].cpu().numpy()
total_mass = np.sum(masses)
body_names = robot.data.body_names

def get_com_offset(roll_deg, pitch_deg, rr_thigh, rr_calf, fl_thigh, fr_thigh, rl_thigh):
    # 関節角度を設定
    joint_pos = robot.data.default_joint_pos.clone()
    joint_pos[0, 0] = 0.1   # FL_hip
    joint_pos[0, 1] = -0.1  # FR_hip
    joint_pos[0, 2] = 0.2   # RL_hip
    joint_pos[0, 3] = -0.1  # RR_hip
    joint_pos[0, 4] = fl_thigh # FL_thigh
    joint_pos[0, 5] = fr_thigh # FR_thigh
    joint_pos[0, 6] = rl_thigh # RL_thigh
    joint_pos[0, 7] = rr_thigh # RR_thigh
    joint_pos[0, 8] = -1.0  # FL_calf
    joint_pos[0, 9] = -1.0  # FR_calf
    joint_pos[0, 10] = -2.5 # RL_calf
    joint_pos[0, 11] = rr_calf # RR_calf
    
    r = R.from_euler('xyz', [roll_deg, pitch_deg, 0], degrees=True)
    quat = r.as_quat() # (x, y, z, w)
    quat_isaac = torch.tensor([[quat[3], quat[0], quat[1], quat[2]]], device=env.unwrapped.device)
    pos_isaac = torch.tensor([[0.0, 0.0, 0.35]], device=env.unwrapped.device)
    
    robot.write_root_pose_to_sim(torch.cat([pos_isaac, quat_isaac], dim=-1))
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    robot.reset()
    
    # 計算
    body_pos_w = robot.data.body_pos_w[0].cpu().numpy()
    com_w = np.sum(body_pos_w * masses[:, None], axis=0) / total_mass
    rr_foot_w = body_pos_w[body_names.index('RR_foot')]
    
    offset_xy = com_w[:2] - rr_foot_w[:2]
    return offset_xy, com_w, rr_foot_w

# グリッドサーチで完全重心直下アライメント姿勢を探索
best_err = 1e9
best_params = None

for roll in np.linspace(-30, 30, 13):
    for pitch in np.linspace(-60, 0, 13):
        for fl_th in [-1.5, -0.5, 0.5]:
            offset, com, foot = get_com_offset(roll, pitch, 0.6, -1.8, fl_th, fl_th, 1.5)
            err = np.linalg.norm(offset)
            if err < best_err:
                best_err = err
                best_params = (roll, pitch, fl_th)

print(f"Best CoM-Foot Offset Error: {best_err*100:.2f} cm")
print(f"Best Params: Roll={best_params[0]:.1f} deg, Pitch={best_params[1]:.1f} deg, FL_thigh={best_params[2]:.2f}")

env.close()
simulation_app.close()

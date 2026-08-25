from __future__ import annotations
import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
import os
import cv2
import imageio
import gymnasium as gym
import torch
import math
import numpy as np
from scipy.optimize import minimize

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless', '--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env_cfg.viewer.eye = (0.0, 2.0, 0.45)
env_cfg.viewer.lookat = (0.0, 0.0, 0.28)

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
robot = env.unwrapped.scene['robot']

print("\n" + "="*80)
print("=== NUMERICAL OPTIMIZATION: PURE TOE EXACT GROUND & COM ALIGNMENT ===")
print("="*80)

# 最適化変数: [Roll, Pitch, Base_Z, RR_thigh, RR_calf, RR_hip]
x0 = np.array([math.radians(24.0), math.radians(-18.0), 0.320, 0.70, -1.85, 0.05])

def loss_func(x):
    roll, pitch, base_z, thigh, calf, hip = x
    
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp
    
    root_pos = torch.tensor([[0.0, 0.0, base_z]], dtype=torch.float32, device='cuda:0')
    root_rot = torch.tensor([[qx, qy, qz, qw]], dtype=torch.float32, device='cuda:0')
    
    joint_pos = torch.tensor([[
        0.1, -0.1, 0.2, hip,
        -1.4, -1.4, 1.8, thigh,
        -0.9, -0.9, -2.6, calf
    ]], dtype=torch.float32, device='cuda:0')
    joint_vel = torch.zeros_like(joint_pos)
    
    robot.write_root_pose_to_sim(torch.cat([root_pos, root_rot], dim=-1))
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device='cuda:0'))
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.update(0.0)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    
    com = robot.data.root_pos_w[0].cpu().numpy()
    
    # 1. 足先が床 Z=0.020m にピタリと接地
    foot_z_loss = (rr_foot[2] - 0.020)**2 * 2000.0
    
    # 2. 重心が足先の真上 (X_com == X_foot, Y_com == Y_foot)
    com_x_loss = (com[0] - rr_foot[0])**2 * 800.0
    com_y_loss = (com[1] - rr_foot[1])**2 * 800.0
    
    # 3. 膝 (RR_calf) が床から 8cm 以上高い
    knee_loss = max(0.0, 0.08 - rr_calf[2])**2 * 1000.0
    
    # 4. 他3脚が 10cm 以上高い
    other_loss = max(0.0, 0.10 - min(fl_foot[2], fr_foot[2], rl_foot[2]))**2 * 1000.0
    
    return foot_z_loss + com_x_loss + com_y_loss + knee_loss + other_loss

res = minimize(loss_func, x0, method='Nelder-Mead', options={'maxiter': 300, 'disp': False})
opt_roll, opt_pitch, opt_z, opt_thigh, opt_calf, opt_hip = res.x

print("\n" + "="*80)
print("=== OPTIMIZATION RESULT ===")
print(f"Optimal Roll    : {math.degrees(opt_roll):.2f}° ({opt_roll:.4f} rad)")
print(f"Optimal Pitch   : {math.degrees(opt_pitch):.2f}° ({opt_pitch:.4f} rad)")
print(f"Optimal Base Z  : {opt_z*100:.2f} cm ({opt_z:.4f} m)")
print(f"Optimal RR_thigh: {opt_thigh:.4f} rad")
print(f"Optimal RR_calf : {opt_calf:.4f} rad")
print(f"Optimal RR_hip  : {opt_hip:.4f} rad")

cr = math.cos(opt_roll * 0.5); sr = math.sin(opt_roll * 0.5)
cp = math.cos(opt_pitch * 0.5); sp = math.sin(opt_pitch * 0.5)
qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

root_pos = torch.tensor([[0.0, 0.0, opt_z]], dtype=torch.float32, device='cuda:0')
root_rot = torch.tensor([[qx, qy, qz, qw]], dtype=torch.float32, device='cuda:0')
joint_pos = torch.tensor([[0.1, -0.1, 0.2, opt_hip, -1.4, -1.4, 1.8, opt_thigh, -0.9, -0.9, -2.6, opt_calf]], dtype=torch.float32, device='cuda:0')

robot.write_root_pose_to_sim(torch.cat([root_pos, root_rot], dim=-1))
robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device='cuda:0'))
robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
robot.reset()
robot.update(0.0)

body_pos = robot.data.body_pos_w[0].cpu().numpy()
rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
rl_foot = body_pos[robot.data.body_names.index('RL_foot')]

print(f"\n[Validation at Optimal Pose]")
print(f"RR_Foot Pos: X={rr_foot[0]:+.3f}m, Y={rr_foot[1]:+.3f}m, Z={rr_foot[2]*100:+.2f}cm (Target: +2.0cm)")
print(f"RR_Knee Pos: Z={rr_calf[2]*100:+.2f}cm (Target: >= +8.0cm)")
print(f"Other Min Z: Z={min(fl_foot[2], fr_foot[2], rl_foot[2])*100:+.2f}cm (Target: >= +10.0cm)")
print(f"Quaternion (x,y,z,w): ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})")
print("="*80)

env.close()
simulation_app.close()

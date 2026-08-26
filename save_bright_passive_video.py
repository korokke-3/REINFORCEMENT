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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env_cfg.viewer.eye = (0.0, 1.8, 0.38)
env_cfg.viewer.lookat = (0.0, 0.0, 0.25)
env_cfg.sim.render_interval = 1

# 最適受動自立姿勢をセット
opt_roll = math.radians(19.0)
opt_pitch = math.radians(-13.0)
cr = math.cos(opt_roll * 0.5); sr = math.sin(opt_roll * 0.5)
cp = math.cos(opt_pitch * 0.5); sp = math.sin(opt_pitch * 0.5)
qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.420)
env_cfg.scene.robot.init_state.rot = (qx, qy, qz, qw)
env_cfg.scene.robot.init_state.joint_pos = {
    "FL_hip_joint": 0.30,
    "FR_hip_joint": -0.30,
    "RL_hip_joint": 0.30,
    "RR_hip_joint": 0.05,
    "FL_thigh_joint": -1.3,
    "FR_thigh_joint": -1.3,
    "RL_thigh_joint": 1.7,
    "RR_thigh_joint": 0.00,
    "FL_calf_joint": -1.0,
    "FR_calf_joint": -1.0,
    "RL_calf_joint": -2.5,
    "RR_calf_joint": -0.838,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')

obs, _ = env.reset()
frames = []

for step in range(60):
    actions = torch.zeros((1, 3), device='cuda:0')
    obs, rewards, dones, truncated, infos = env.step(actions)
    
    frame = env.render()
    if frame is not None and step > 0: # 最初のダミー黒フレーム(step 0)を除外
        frames.append(frame)
        print(f"Captured frame {step}: shape={frame.shape}, mean_brightness={np.mean(frame):.2f}")

out_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/passive_stand_proof'
os.makedirs(out_dir, exist_ok=True)

mp4_path = f"{out_dir}/passive_stand_balance.mp4"
gif_path = f"{out_dir}/passive_stand_balance.gif"

if frames:
    print(f"Saving {len(frames)} frames with mean brightness {np.mean(frames):.2f} to {mp4_path}...")
    imageio.mimsave(mp4_path, frames, fps=25)
    imageio.mimsave(gif_path, frames, fps=25, loop=0)
    print("SUCCESSFULLY SAVED BRIGHT VIDEO AND GIF!")

env.close()
simulation_app.close()

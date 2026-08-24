import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless', '--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import cv2
import imageio
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

# 真横から足元と跳躍を綺麗に撮影
env_cfg.viewer.eye = (0.0, 1.3, 0.25)
env_cfg.viewer.lookat = (0.0, 0.0, 0.15)

# しゃがみ込みタメ姿勢 (Roll=+25.4度, Pitch=-50.0度)
r_init = R.from_euler('xyz', [25.38, -50.0, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

# 足先が地面(Z=0.023m)にピタリと接地する胴体高度
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.14)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    'FL_hip_joint': 0.1, 'FR_hip_joint': -0.1, 'RL_hip_joint': 0.2, 'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2, 'FR_thigh_joint': -1.2, 'RL_thigh_joint': 1.5, 'RR_thigh_joint': 0.8,
    'FL_calf_joint': -1.0, 'FR_calf_joint': -1.0, 'RL_calf_joint': -2.5, 'RR_calf_joint': -2.2,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
video_folder = '/home/exhibition-spakona/Desktop/REINFORCEMENT/ik_aligned_jump_proof'
os.makedirs(video_folder, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=video_folder,
    step_trigger=lambda step: step == 0,
    video_length=70,
    name_prefix='ik_ground_jump',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print('=== EXECUTING PERFECT GROUND-CONTACT SINGLE-LEG JUMP ===')

for step in range(70):
    t = step * 0.02
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 3脚は空中に高く折りたたみ固定
    actions[0, 4] = -1.2 # FL_thigh
    actions[0, 5] = -1.2 # FR_thigh
    actions[0, 6] = 1.5  # RL_thigh
    actions[0, 8] = -1.0 # FL_calf
    actions[0, 9] = -1.0 # FR_calf
    actions[0, 10] = -2.5 # RL_calf
    
    # 0.00s ~ 0.10s (5 steps): 地面を踏みしめてタメ (接地保持)
    if t < 0.10:
        actions[0, 7] = 0.8   # RR_thigh
        actions[0, 11] = -2.2 # RR_calf
    # 0.10s ~ 0.22s (6 steps): ★ 垂直爆発的キック (急伸展) ★
    elif t < 0.22:
        actions[0, 7] = -0.6  # RR_thigh (大腿を突き上げる)
        actions[0, 11] = 2.4  # RR_calf (膝を一気に最大急伸展)
    # 0.22s ~ 0.36s: 滞空中の足引き戻し (Tuck in air)
    elif t < 0.36:
        actions[0, 7] = 0.5
        actions[0, 11] = -1.6
    else:
        actions[0, 7] = 0.8
        actions[0, 11] = -1.2

    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_z = robot.data.root_pos_w[0, 2].item()
    
    print(f'Step {step:02d} (t={t:.2f}s): Foot Z = {rr_foot[2]*100:+.2f} cm, Body Z = {root_z*100:+.2f} cm')

env.close()
simulation_app.close()

# GIF 生成
video_path = '/home/exhibition-spakona/Desktop/REINFORCEMENT/ik_aligned_jump_proof/ik_ground_jump-step-0.mp4'
gif_path = '/home/exhibition-spakona/Desktop/REINFORCEMENT/ik_aligned_jump_proof/ik_jump_animation.gif'

cap = cv2.VideoCapture(video_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

if len(frames) > 0:
    imageio.mimsave(gif_path, frames, fps=25, loop=0)
    print(f'Saved {gif_path} successfully.')

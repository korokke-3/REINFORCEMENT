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

# カメラアングル: ジャンプ全体が横から迫力あるサイズで見える画角
env_cfg.viewer.eye = (0.0, 1.2, 0.3)
env_cfg.viewer.lookat = (0.0, 0.0, 0.2)

# 直立・右後足接地構え (Roll=+15度, Pitch=-40.3度)
r_init = R.from_euler('xyz', [15.0, -40.3, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.26)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    'FL_hip_joint': 0.1, 'FR_hip_joint': -0.1, 'RL_hip_joint': 0.2, 'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2, 'FR_thigh_joint': -1.2, 'RL_thigh_joint': 1.5, 'RR_thigh_joint': 0.6,
    'FL_calf_joint': -1.0, 'FR_calf_joint': -1.0, 'RL_calf_joint': -2.5, 'RR_calf_joint': -1.8, # しっかり曲げた構え
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
video_folder = '/home/exhibition-spakona/Desktop/REINFORCEMENT/big_jump_proof'
os.makedirs(video_folder, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=video_folder,
    step_trigger=lambda step: step == 0,
    video_length=60,
    name_prefix='big_jump',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print('=== EXECUTING MAXIMUM TORQUE SINGLE-LEG EXPLOSIVE JUMP ===')

foot_z_list = []
base_z_list = []

for step in range(60):
    t = step * 0.02
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 3脚は空中に高く固定
    actions[0, 4] = -1.2 # FL_thigh
    actions[0, 5] = -1.2 # FR_thigh
    actions[0, 6] = 1.5  # RL_thigh
    actions[0, 8] = -1.0 # FL_calf
    actions[0, 9] = -1.0 # FR_calf
    actions[0, 10] = -2.5 # RL_calf
    
    # 0.00s ~ 0.08s: 接地タメ
    if t < 0.08:
        actions[0, 7] = 0.6   # RR_thigh
        actions[0, 11] = -1.8 # RR_calf
    # 0.08s ~ 0.20s: ★ 全開フルパワースラストキック (最大トルク急伸展) ★
    elif t < 0.20:
        actions[0, 7] = -1.0  # RR_thigh 全力伸展
        actions[0, 11] = 3.0  # RR_calf 最大全開急伸展
    # 0.20s ~ 0.35s: 空中での足引き戻し (Tuck)
    elif t < 0.35:
        actions[0, 7] = 0.5
        actions[0, 11] = -2.0
    else:
        actions[0, 7] = 0.6
        actions[0, 11] = -1.5

    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_z = robot.data.root_pos_w[0, 2].item()
    
    foot_z_list.append(rr_foot[2])
    base_z_list.append(root_z)
    
    print(f'Step {step:02d} (t={t:.2f}s): Foot Z = {rr_foot[2]*100:+.2f} cm, Body Z = {root_z*100:+.2f} cm')

env.close()
simulation_app.close()

# フレーム切り出し
video_path = '/home/exhibition-spakona/Desktop/REINFORCEMENT/big_jump_proof/big_jump-step-0.mp4'
cap = cv2.VideoCapture(video_path)
frames = []
idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # ピークフレーム (Step 9, 10, 11) をPNG保存
    if idx in [4, 8, 10, 12, 15]:
        cv2.imwrite(f'/home/exhibition-spakona/Desktop/REINFORCEMENT/big_jump_proof/frame_{idx}.png', frame)
    idx += 1
cap.release()

gif_path = '/home/exhibition-spakona/Desktop/REINFORCEMENT/big_jump_proof/big_jump.gif'
imageio.mimsave(gif_path, frames, fps=20, loop=0)
print('Generated big_jump.gif and frames successfully.')

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

# カメラアングル: ジャンプ全体が横から美しく見える画角
env_cfg.viewer.eye = (0.0, 1.8, 0.4)
env_cfg.viewer.lookat = (0.0, 0.0, 0.25)

# スポーン: 水平に近い低重心構え (Pitch = -15度, Roll = +12度)
r = R.from_euler('xyz', [12, -15, 0], degrees=True)
quat = tuple(r.as_quat().tolist())
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.28)
env_cfg.scene.robot.init_state.rot = quat
env_cfg.scene.robot.init_state.joint_pos = {
    'FL_hip_joint': 0.1, 'FR_hip_joint': -0.1, 'RL_hip_joint': 0.2, 'RR_hip_joint': -0.1,
    'FL_thigh_joint': 0.0, 'FR_thigh_joint': 0.0, 'RL_thigh_joint': 0.5, 'RR_thigh_joint': 1.0,
    'FL_calf_joint': -1.0, 'FR_calf_joint': -1.0, 'RL_calf_joint': -1.5, 'RR_calf_joint': -2.4, # 深いしゃがみ込み
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
video_folder = '/home/exhibition-spakona/Desktop/REINFORCEMENT/arm_swing_jump_proof'
os.makedirs(video_folder, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=video_folder,
    step_trigger=lambda step: step == 0,
    video_length=100,
    name_prefix='arm_swing_single_jump',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print('=== EXECUTING ARM-SWING & EXPLOSIVE KICK SINGLE-LEG JUMP ===')

foot_z_list = []
base_z_list = []
flight_frames = []

for step in range(100):
    t = step * 0.02 # 秒
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 制御タイムライン:
    # 0.00s ~ 0.10s (5 steps): しゃがみ込み保持 (Squat)
    if t < 0.10:
        actions[0, 7] = 0.5   # RR_thigh
        actions[0, 11] = -1.0 # RR_calf
    # 0.10s ~ 0.22s (6 steps): ★ 爆発的キック ＋ 前脚・左脚の強烈な上方振り上げ ★
    elif t < 0.22:
        actions[0, 7] = -1.2  # RR_thigh 急伸展 (キック)
        actions[0, 11] = 2.2  # RR_calf 最大急伸展 (キック)
        # 前脚2本とお腹側脚を上方にフルスイング (反動)
        actions[0, 4] = -2.0  # FL_thigh 上方振り上げ
        actions[0, 5] = -2.0  # FR_thigh 上方振り上げ
        actions[0, 6] = 2.0   # RL_thigh 上方振り上げ
    # 0.22s ~ 0.40s: 空中での足引き戻し (Tuck in air)
    elif t < 0.40:
        actions[0, 7] = 0.5
        actions[0, 11] = -1.5 # 右足を空中に引き上げる
    # 0.40s 以降: 着地構え
    else:
        actions[0, 7] = 0.8
        actions[0, 11] = -1.2

    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_z = robot.data.root_pos_w[0, 2].item()
    
    foot_z_list.append(rr_foot[2])
    base_z_list.append(root_z)
    
    # 地面 (Z <= 0.025m) から離れて空中に浮いているか
    if rr_foot[2] > 0.035:
        flight_frames.append(step)
        print(f'>>> FLIGHT DETECTED at Step {step:02d} (t={t:.2f}s): Foot Z = {rr_foot[2]*100:+.1f} cm, Body Z = {root_z*100:+.1f} cm')

foot_z_arr = np.array(foot_z_list)
base_z_arr = np.array(base_z_list)

max_foot_z = np.max(foot_z_arr)
max_base_z = np.max(base_z_arr)

print(f'\n=== FINAL MEASUREMENT RESULTS ===')
print(f'Initial Foot Height : {foot_z_arr[0]*100:.1f} cm')
print(f'MAX Foot Air Height : {max_foot_z*100:.1f} cm (Pure Jump Lift: +{(max_foot_z - 0.023)*100:.1f} cm)')
print(f'MAX Body Air Height : {max_base_z*100:.1f} cm')
print(f'Total Flight Frames : {len(flight_frames)} frames ({len(flight_frames)*0.02:.2f} seconds of true air time)')

env.close()
simulation_app.close()
print('Recorded arm_swing_single_jump.')

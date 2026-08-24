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
env_cfg.actions.joint_pos.scale = 1.0

# カメラ設定
env_cfg.viewer.eye = (0.0, 1.4, 0.35)
env_cfg.viewer.lookat = (0.0, 0.0, 0.22)

# 最適直立バランス角
TARGET_ROLL_DEG = 28.5
TARGET_PITCH_DEG = -30.5
r_init = R.from_euler('xyz', [TARGET_ROLL_DEG, TARGET_PITCH_DEG, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

FIXED_JOINTS = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,
    'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,
}

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.215)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    **FIXED_JOINTS,
    'RR_thigh_joint': 0.75,
    'RR_calf_joint': -2.15,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/resonance_hop_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=120,
    name_prefix='resonance_hop',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*75)
print("=== EXPERIMENT 3: RESONANT REBOUND-SYNCHRONIZED SINGLE-LEG HOPPING ===")
print("="*75)

joint_names = robot.data.joint_names
rr_hip_idx = joint_names.index('RR_hip_joint')
rr_thigh_idx = joint_names.index('RR_thigh_joint')
rr_calf_idx = joint_names.index('RR_calf_joint')

# リバウンド同期制御
# 状態:
# 'SQUAT_FALL': 着地後の沈み込み (下降中 Vz < 0)
# 'THRUST': 最下点反転時の瞬間爆発キック
# 'FLIGHT_TUCK': 上昇・滞空中 (足引き戻し)
# 'PRE_TOUCH': 下降開始・足伸展準備
current_mode = 'THRUST'
mode_step = 0
hop_records = []

prev_vz = 0.0

for step in range(120):
    t = step * 0.02
    mode_step += 1
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    rot = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
    euler_deg = rot.as_euler('xyz', degrees=True)
    
    vz = root_vel[2]
    is_airborne = rr_foot[2] > 0.035
    is_grounded = rr_foot[2] <= 0.026
    
    # リバウンド反転検知: Vzが負から正へ切り替わる瞬間、または最下点
    bottom_reached = (prev_vz < -0.05) and (vz >= -0.02)
    prev_vz = vz
    
    # モード遷移
    if current_mode == 'THRUST':
        target_thigh = -0.30
        target_calf = -0.60
        if mode_step >= 4: # 0.08秒キック
            current_mode = 'FLIGHT_TUCK'
            mode_step = 0
    elif current_mode == 'FLIGHT_TUCK':
        target_thigh = 0.75
        target_calf = -2.20
        # 頂点通過または下降開始で着地準備へ
        if vz < 0.0 or mode_step >= 8:
            current_mode = 'PRE_TOUCH'
            mode_step = 0
    elif current_mode == 'PRE_TOUCH':
        target_thigh = 0.60
        target_calf = -1.80
        if is_grounded:
            current_mode = 'SQUAT_FALL'
            mode_step = 0
    elif current_mode == 'SQUAT_FALL':
        # 着地沈み込み中は深く曲げて衝撃吸収＆バネチャージ
        target_thigh = 0.85
        target_calf = -2.35
        # 最下点に到達、または3ステップ沈み込んだら即座にキック発動！
        if bottom_reached or mode_step >= 3:
            current_mode = 'THRUST'
            mode_step = 0
            hop_records.append((step, t, root_pos[2]))
            print(f">>> [REBOUND KICK TRIGGERED] at step {step:03d} (t={t:.2f}s, Height={root_pos[2]*100:.1f}cm) <<<")

    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    for jname, val in FIXED_JOINTS.items():
        j_idx = joint_names.index(jname)
        actions[0, j_idx] = val - robot.data.default_joint_pos[0, j_idx]

    actions[0, rr_thigh_idx] = target_thigh - robot.data.default_joint_pos[0, rr_thigh_idx]
    actions[0, rr_calf_idx] = target_calf - robot.data.default_joint_pos[0, rr_calf_idx]

    obs, reward, terminated, truncated, info = env.step(actions)
    
    status = "★ AIRBORNE ★" if is_airborne else "Grounded"
    if step % 2 == 0 or is_airborne or current_mode == 'THRUST':
        print(f"Step {step:03d} (t={t:.2f}s | {current_mode:<11}) | Foot_Z={rr_foot[2]*100:+5.1f}cm | Body_Z={root_pos[2]*100:5.1f}cm | Vz={vz:+5.2f}m/s | {status}")

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/resonance_hop-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/resonance_hop.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

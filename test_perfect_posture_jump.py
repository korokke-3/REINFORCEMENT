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
# 終了条件を無効化
env_cfg.terminations.illegal_parts_contact = None
env_cfg.terminations.forward_fall = None
env_cfg.terminations.root_height_below_minimum = None
env_cfg.terminations.base_contact = None

# アクションスケールを1.0
env_cfg.actions.joint_pos.scale = 1.0

# カメラ設定
env_cfg.viewer.eye = (0.0, 1.4, 0.32)
env_cfg.viewer.lookat = (0.0, 0.0, 0.20)

# 最適バランス姿勢 (Roll=+28.5度, Pitch=-30.5度)
r_init = R.from_euler('xyz', [28.5, -30.5, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

# 固定する他3脚の姿勢（綺麗に折りたたみ）
FIXED_JOINTS = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,
    'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,
}

# 初期状態
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.215)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    **FIXED_JOINTS,
    'RR_thigh_joint': 0.70,
    'RR_calf_joint': -2.00,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/perfect_posture_jump_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=80,
    name_prefix='perfect_posture_jump',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*70)
print("=== EXPERIMENT: PERFECT SAME-POSTURE SINGLE-LEG JUMP ===")
print("=== (ZERO ARM-SWING, RIGID POSTURE, CO-LINEAR PISTON JUMP) ===")
print("="*70)

log_data = []

for step in range(80):
    t = step * 0.02
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 1. 他3脚は完全に一定角度に固定（一切動かさない）
    for idx, (jname, val) in enumerate(FIXED_JOINTS.items()):
        j_idx = robot.data.joint_names.index(jname)
        actions[0, j_idx] = val - robot.data.default_joint_pos[0, j_idx]

    # 2. 支持脚（RR）のピストン伸縮シーケンス
    # 0.00s ~ 0.08s (4 steps): タメ（接地圧着）
    if t < 0.08:
        target_rr_thigh = 0.70
        target_rr_calf = -2.00
    # 0.08s ~ 0.16s (4 steps): ★ 垂直ピストン急伸展（キック） ★
    elif t < 0.16:
        target_rr_thigh = -0.20
        target_rr_calf = -0.70
    # 0.16s ~ 0.28s (6 steps): ★ 空中足引き込み（滞空クリアランス確保） ★
    elif t < 0.28:
        target_rr_thigh = 0.70
        target_rr_calf = -2.20
    # 0.28s 以降: 着地姿勢保持
    else:
        target_rr_thigh = 0.60
        target_rr_calf = -1.70

    rr_thigh_idx = robot.data.joint_names.index('RR_thigh_joint')
    rr_calf_idx = robot.data.joint_names.index('RR_calf_joint')
    actions[0, rr_thigh_idx] = target_rr_thigh - robot.data.default_joint_pos[0, rr_thigh_idx]
    actions[0, rr_calf_idx] = target_rr_calf - robot.data.default_joint_pos[0, rr_calf_idx]

    obs, reward, terminated, truncated, info = env.step(actions)
    
    # 計測
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy() # (w, x, y, z)
    rot = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
    euler_deg = rot.as_euler('xyz', degrees=True)
    
    is_airborne = rr_foot[2] > 0.035
    
    log_data.append({
        'step': step, 't': t, 'rr_z': rr_foot[2], 'body_z': root_pos[2],
        'vz': root_vel[2], 'roll': euler_deg[0], 'pitch': euler_deg[1], 'air': is_airborne
    })
    
    status = "★ AIRBORNE ★" if is_airborne else "Grounded"
    print(f"Step {step:02d} (t={t:.2f}s) | RR_Z={rr_foot[2]*100:+5.1f}cm | Body_Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | Roll={euler_deg[0]:+5.1f}° Pitch={euler_deg[1]:+5.1f}° | {status}")

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/perfect_posture_jump-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/perfect_posture_jump.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

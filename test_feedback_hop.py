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

# 環境設定
env_cfg = Go2SingleLegEnvCfg_PLAY()
env_cfg.terminations.illegal_parts_contact = None
env_cfg.terminations.forward_fall = None
env_cfg.terminations.root_height_below_minimum = None
env_cfg.terminations.base_contact = None
env_cfg.actions.joint_pos.scale = 1.0

# カメラ設定
env_cfg.viewer.eye = (0.0, 1.4, 0.35)
env_cfg.viewer.lookat = (0.0, 0.0, 0.22)

# 最適直立バランス姿勢 (Roll=+28.5°, Pitch=-30.5°)
TARGET_ROLL_DEG = 28.5
TARGET_PITCH_DEG = -30.5
r_init = R.from_euler('xyz', [TARGET_ROLL_DEG, TARGET_PITCH_DEG, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

# 固定する他3脚の関節角
FIXED_JOINTS = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,
    'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,
}

# 初期設定
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.215)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    **FIXED_JOINTS,
    'RR_thigh_joint': 0.70,
    'RR_calf_joint': -2.00,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/feedback_single_leg_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=120, # 2.4秒
    name_prefix='feedback_hop',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*70)
print("=== EXPERIMENT 1: CLOSED-LOOP POSTURE FEEDBACK + PISTON HOPPING ===")
print("="*70)

# PDフィードバックゲイン (姿勢を目標角度に保つための股関節微小補正)
Kp_roll = 0.8
Kd_roll = 0.08
Kp_pitch = 1.0
Kd_pitch = 0.10

# 関節インデックスの取得
joint_names = robot.data.joint_names
rr_hip_idx = joint_names.index('RR_hip_joint')
rr_thigh_idx = joint_names.index('RR_thigh_joint')
rr_calf_idx = joint_names.index('RR_calf_joint')

log_data = []

# 連続周期ホッピングのパラメータ
HOP_PERIOD = 0.30 # 0.3秒周期 (約3.3Hz)
# 1周期内のフェーズ比率:
# 0% ~ 30%: 接地タメ (Squat)
# 30% ~ 55%: 爆発的キック (Thrust)
# 55% ~ 85%: 空中タック・足引き戻し (Flight & Tuck)
# 85% ~ 100%: タッチダウン準備 (Prepare)

for step in range(120):
    t = step * 0.02
    phase = (t % HOP_PERIOD) / HOP_PERIOD # 0.0 ~ 1.0
    
    # 1. 姿勢・角速度の取得
    root_quat = robot.data.root_quat_w[0].cpu().numpy() # (w, x, y, z)
    root_ang_vel = robot.data.root_ang_vel_w[0].cpu().numpy()
    rot = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
    euler_deg = rot.as_euler('xyz', degrees=True)
    roll_cur, pitch_cur, yaw_cur = euler_deg
    
    # 姿勢誤差の計算
    roll_err = np.radians(TARGET_ROLL_DEG - roll_cur)
    pitch_err = np.radians(TARGET_PITCH_DEG - pitch_cur)
    roll_rate = root_ang_vel[0]
    pitch_rate = root_ang_vel[1]
    
    # 姿勢復元用フィードバックオフセット
    delta_hip = -(Kp_roll * roll_err - Kd_roll * roll_rate)
    delta_thigh = -(Kp_pitch * pitch_err - Kd_pitch * pitch_rate)
    
    # 2. 支持脚（RR）の周期ホッピング軌道
    if phase < 0.30:
        # タメ (Squat)
        base_rr_thigh = 0.70
        base_rr_calf = -2.00
    elif phase < 0.55:
        # ★ 爆発的キック (Thrust) ★
        base_rr_thigh = -0.30
        base_rr_calf = -0.60
    elif phase < 0.85:
        # ★ 空中引き込み (Tuck in flight) ★
        base_rr_thigh = 0.75
        base_rr_calf = -2.30
    else:
        # 接地準備 (Prepare)
        base_rr_thigh = 0.65
        base_rr_calf = -1.80

    # 3. アクション指令の合成
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 他3脚は完全固定
    for jname, val in FIXED_JOINTS.items():
        j_idx = joint_names.index(jname)
        actions[0, j_idx] = val - robot.data.default_joint_pos[0, j_idx]

    # 支持脚: フィードバック補正を加算
    target_hip = FIXED_JOINTS['RR_hip_joint'] + np.clip(delta_hip, -0.3, 0.3)
    target_thigh = base_rr_thigh + np.clip(delta_thigh, -0.4, 0.4)
    target_calf = base_rr_calf

    actions[0, rr_hip_idx] = target_hip - robot.data.default_joint_pos[0, rr_hip_idx]
    actions[0, rr_thigh_idx] = target_thigh - robot.data.default_joint_pos[0, rr_thigh_idx]
    actions[0, rr_calf_idx] = target_calf - robot.data.default_joint_pos[0, rr_calf_idx]

    obs, reward, terminated, truncated, info = env.step(actions)
    
    # 計測
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    is_airborne = rr_foot[2] > 0.035
    status = "★ AIRBORNE ★" if is_airborne else "Grounded"
    
    log_data.append({
        'step': step, 't': t, 'rr_z': rr_foot[2], 'body_z': root_pos[2],
        'vz': root_vel[2], 'roll': roll_cur, 'pitch': pitch_cur, 'air': is_airborne
    })
    
    if step % 2 == 0 or is_airborne:
        print(f"Step {step:03d} (t={t:.2f}s, Phase={phase:.2f}) | RR_Z={rr_foot[2]*100:+5.1f}cm | Body_Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | Roll={roll_cur:+5.1f}° Pitch={pitch_cur:+5.1f}° | {status}")

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/feedback_hop-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/feedback_hop.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

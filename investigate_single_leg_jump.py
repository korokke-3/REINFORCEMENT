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
# 転倒終了を無効化して物理計測
env_cfg.terminations.illegal_parts_contact = None
env_cfg.terminations.forward_fall = None
env_cfg.terminations.root_height_below_minimum = None
env_cfg.terminations.base_contact = None

# カメラ設定
env_cfg.viewer.eye = (0.0, 1.3, 0.3)
env_cfg.viewer.lookat = (0.0, 0.0, 0.2)

# 重心直下アライメント姿勢
# RR_foot が原点 (0, 0, 0.023) に接地し、全体の重心が鉛直上方にくるように設定
r_init = R.from_euler('xyz', [18.0, -38.0, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.24)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    'FL_hip_joint': 0.1,  'FR_hip_joint': -0.1, 'RL_hip_joint': 0.2,  'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2, 'FR_thigh_joint': -1.2, 'RL_thigh_joint': 1.5,  'RR_thigh_joint': 0.7,
    'FL_calf_joint': -1.0,  'FR_calf_joint': -1.0,  'RL_calf_joint': -2.5, 'RR_calf_joint': -2.0,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/single_leg_jump_investigation'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=80,
    name_prefix='investigation_jump',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*60)
print("=== INVESTIGATION: PROGRAMMED SINGLE-LEG JUMP EXPERIMENT ===")
print("="*60)

log_data = []

# シミュレーションループ (80 steps = 1.6s)
for step in range(80):
    t = step * 0.02
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 3脚の初期保持
    actions[0, 4] = -1.2 # FL_thigh
    actions[0, 5] = -1.2 # FR_thigh
    actions[0, 6] = 1.5  # RL_thigh
    actions[0, 8] = -1.0 # FL_calf
    actions[0, 9] = -1.0 # FR_calf
    actions[0, 10] = -2.5 # RL_calf

    # 制御タイムライン:
    # Phase 1 (0.00s ~ 0.08s): 接地タメ (Squat & Settle)
    if t < 0.08:
        actions[0, 7] = 0.7   # RR_thigh
        actions[0, 11] = -2.0 # RR_calf
    # Phase 2 (0.08s ~ 0.18s): ★ 爆発的同時キック + 3脚反動スイング (Explosive Jump) ★
    elif t < 0.18:
        # 支持脚のフルパワー伸展
        actions[0, 7] = -1.5  # RR_thigh
        actions[0, 11] = 3.5  # RR_calf 最大伸展
        # 他脚3本を上方へ強烈にスイング（反動加速）
        actions[0, 4] = 2.0   # FL_thigh 上方スイング
        actions[0, 5] = 2.0   # FR_thigh 上方スイング
        actions[0, 6] = -1.0  # RL_thigh 上方スイング
    # Phase 3 (0.18s ~ 0.35s): ★ 空中タック（足を折りたたんで滞空クリアランス確保） ★
    elif t < 0.35:
        actions[0, 7] = 0.8   # RR_thigh
        actions[0, 11] = -2.2 # RR_calf 折りたたみ
        actions[0, 4] = -1.2
        actions[0, 5] = -1.2
        actions[0, 6] = 1.5
    # Phase 4 (0.35s 以降): 着地準備
    else:
        actions[0, 7] = 0.5
        actions[0, 11] = -1.5

    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot_pos = body_pos[robot.data.body_names.index('RR_foot')]
    fl_foot_pos = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot_pos = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot_pos = body_pos[robot.data.body_names.index('RL_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    # 全脚の最低高度
    all_feet_z = [rr_foot_pos[2], fl_foot_pos[2], fr_foot_pos[2], rl_foot_pos[2]]
    min_foot_z = min(all_feet_z)
    
    is_airborne = min_foot_z > 0.035 # 地面から3.5cm以上浮いているか
    
    log_data.append({
        'step': step,
        't': t,
        'rr_foot_z': rr_foot_pos[2],
        'root_z': root_pos[2],
        'vz': root_vel[2],
        'min_foot_z': min_foot_z,
        'is_airborne': is_airborne,
    })
    
    status = "AIRBORNE!" if is_airborne else "Grounded"
    print(f"Step {step:02d} (t={t:.2f}s) | RR_Foot Z={rr_foot_pos[2]*100:+5.1f}cm | Body Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | [{status}]")

env.close()
simulation_app.close()

# 分析集計
airborne_frames = [d for d in log_data if d['is_airborne']]
max_vz = max(d['vz'] for d in log_data)
max_body_z = max(d['root_z'] for d in log_data)
max_foot_z = max(d['rr_foot_z'] for d in log_data)

print("\n" + "="*60)
print("=== INVESTIGATION RESULTS SUMMARY ===")
print(f"Maximum Vertical Velocity (Vz)  : {max_vz:+.3f} m/s")
print(f"Maximum Body Height             : {max_body_z*100:.1f} cm")
print(f"Maximum RR Foot Height          : {max_foot_z*100:.1f} cm")
print(f"Total True Airborne Frames      : {len(airborne_frames)} steps ({len(airborne_frames)*0.02:.2f} s)")
print("="*60)

# GIF生成
video_path = f"{output_dir}/investigation_jump-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/investigation_jump.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

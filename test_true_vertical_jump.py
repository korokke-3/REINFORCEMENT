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

# アクションスケールを1.0にしてフルレンジ駆動を可能にする
env_cfg.actions.joint_pos.scale = 1.0

# カメラ設定
env_cfg.viewer.eye = (0.0, 1.5, 0.3)
env_cfg.viewer.lookat = (0.0, 0.0, 0.2)

# ロボットの初期姿勢:
# 右後足（RR）が地面に接地した状態で、全体の合成重心が真上に来るようにピッチとロールを精密調整
# Roll=+15度, Pitch=-35度
r_init = R.from_euler('xyz', [15.0, -35.0, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.22)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    'FL_hip_joint': 0.1,  'FR_hip_joint': -0.1, 'RL_hip_joint': 0.2,  'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2, 'FR_thigh_joint': -1.2, 'RL_thigh_joint': 1.5,  'RR_thigh_joint': 0.8,
    'FL_calf_joint': -1.0,  'FR_calf_joint': -1.0,  'RL_calf_joint': -2.5, 'RR_calf_joint': -2.2, # 深いスクワット
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/true_vertical_jump_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=60,
    name_prefix='true_vertical_jump',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']
contact_forces_sensor = env.unwrapped.scene.sensors.get('contact_forces', None)

print("\n" + "="*65)
print("=== HIGH-IMPULSE TRUE VERTICAL SINGLE-LEG JUMP TEST ===")
print("="*65)

log_data = []

for step in range(60):
    t = step * 0.02
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 3脚は空中に維持
    actions[0, 4] = -1.2 - robot.data.default_joint_pos[0, 4]
    actions[0, 5] = -1.2 - robot.data.default_joint_pos[0, 5]
    actions[0, 6] = 1.5  - robot.data.default_joint_pos[0, 6]
    actions[0, 8] = -1.0 - robot.data.default_joint_pos[0, 8]
    actions[0, 9] = -1.0 - robot.data.default_joint_pos[0, 9]
    actions[0, 10] = -2.5 - robot.data.default_joint_pos[0, 10]

    # RR支持脚の制御
    # 0.00s ~ 0.06s (3 steps): 接地圧着
    if t < 0.06:
        actions[0, 7] = 0.8 - robot.data.default_joint_pos[0, 7]
        actions[0, 11] = -2.2 - robot.data.default_joint_pos[0, 11]
    # 0.06s ~ 0.14s (4 steps): ★ 垂直フルスラスト急伸展 ★
    elif t < 0.14:
        actions[0, 7] = -0.4 - robot.data.default_joint_pos[0, 7]
        actions[0, 11] = -0.8 - robot.data.default_joint_pos[0, 11] # 膝を最大伸展
        # 他脚を同時に上方へスイングして反動を乗せる
        actions[0, 4] = 1.5 - robot.data.default_joint_pos[0, 4]
        actions[0, 5] = 1.5 - robot.data.default_joint_pos[0, 5]
    # 0.14s ~ 0.30s (8 steps): ★ 空中タック（足を折りたたんで離地） ★
    elif t < 0.30:
        actions[0, 7] = 0.8 - robot.data.default_joint_pos[0, 7]
        actions[0, 11] = -2.4 - robot.data.default_joint_pos[0, 11]
    else:
        actions[0, 7] = 0.5 - robot.data.default_joint_pos[0, 7]
        actions[0, 11] = -1.5 - robot.data.default_joint_pos[0, 11]

    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    is_airborne = rr_foot[2] > 0.035
    
    log_data.append({
        'step': step, 't': t, 'rr_z': rr_foot[2], 'body_z': root_pos[2],
        'vz': root_vel[2], 'air': is_airborne
    })
    
    status = "AIRBORNE!" if is_airborne else "Grounded"
    print(f"Step {step:02d} (t={t:.2f}s) | RR_Z={rr_foot[2]*100:+5.1f}cm | Body_Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | {status}")

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/true_vertical_jump-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/true_vertical_jump.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

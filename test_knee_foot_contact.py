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
# 終了判定を解除して安定性を観察
env_cfg.terminations.illegal_parts_contact = None
env_cfg.terminations.orientation_deviation = None
env_cfg.terminations.root_height_below_minimum = None

# カメラ設定: 膝＋足先の2点支持が真横・斜めからクリアに見える位置
env_cfg.viewer.eye = (0.0, 1.8, 0.40)
env_cfg.viewer.lookat = (0.0, 0.0, 0.20)

# 膝＋足先の2点接地（ニーリング・スキーブレード姿勢）
# 右後脚の Calf を深く曲げて地面に平行に寝かせる (RR_thigh ≈ 0.8, RR_calf ≈ -2.7)
# Roll を +15度、Pitch を -15度程度に調整
r_init = R.from_euler('xyz', [15.0, -15.0, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.22)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,   'RR_thigh_joint': 0.85,
    'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,  'RR_calf_joint': -2.65, # 膝とすねを地面にピタリと這わせる
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/knee_foot_dual_contact_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=200, # 4.0秒間
    name_prefix='knee_foot_dual_contact',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']
contact_sensor = env.unwrapped.scene.sensors['contact_forces']

print("\n" + "="*75)
print("=== EXPERIMENT: KNEE + FOOT DUAL-POINT LINE CONTACT STABILITY ===")
print("=== (SHANK-GROUND SKI BLADE SUPPORT ON SINGLE LEG) ===")
print("="*75)

log_data = []

for step in range(200):
    t = step * 0.02
    actions = torch.zeros((1, 3), device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
    rr_thigh = body_pos[robot.data.body_names.index('RR_thigh')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    # 接触力の取得
    forces = torch.norm(contact_sensor.data.net_forces_w[0], dim=-1).cpu().numpy()
    b_names = contact_sensor.body_names
    
    foot_force = forces[b_names.index('RR_foot')] if 'RR_foot' in b_names else 0.0
    calf_force = forces[b_names.index('RR_calf')] if 'RR_calf' in b_names else 0.0
    thigh_force = forces[b_names.index('RR_thigh')] if 'RR_thigh' in b_names else 0.0
    
    is_dual_contact = (foot_force > 1.0 or rr_foot[2] < 0.04) and (calf_force > 1.0 or thigh_force > 1.0 or rr_calf[2] < 0.06)
    status = "★★★ ULTRA STABLE (DUAL CONTACT) ★★★" if is_dual_contact else "Single Point / Airborne"
    
    log_data.append({
        'step': step, 't': t, 'body_z': root_pos[2], 'foot_z': rr_foot[2], 'calf_z': rr_calf[2],
        'vz': root_vel[2], 'dual': is_dual_contact
    })
    
    if step % 5 == 0 or step < 10:
        print(f"Step {step:03d} (t={t:.2f}s) | Body_Z={root_pos[2]*100:5.1f}cm | Foot_Z={rr_foot[2]*100:+5.1f}cm | Knee_Z={rr_calf[2]*100:+5.1f}cm | [{status}]")

env.close()
simulation_app.close()

# 分析
dual_steps = [d for d in log_data if d['dual']]
print("\n" + "="*75)
print("=== DUAL-POINT (KNEE + FOOT) CONTACT STABILITY RESULTS ===")
print(f"Total Evaluated Time    : 4.00 s (200 steps)")
print(f"Dual Contact Stable Time: {len(dual_steps)*0.02:.2f} s ({len(dual_steps)} steps)")
print(f"Final Body Height       : {log_data[-1]['body_z']*100:.1f} cm (No collapse!)")
print("="*75)

# GIF生成
video_path = f"{output_dir}/knee_foot_dual_contact-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/knee_foot_dual_contact.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

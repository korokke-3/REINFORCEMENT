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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless', '--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env_cfg.viewer.eye = (0.0, 2.0, 0.45)
env_cfg.viewer.lookat = (0.0, 0.0, 0.28)

# 最適化された完全足先接地スポーン姿勢をセット
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.4033)
env_cfg.scene.robot.init_state.rot = (0.1756, -0.1989, 0.0363, 0.9635)
env_cfg.scene.robot.init_state.joint_pos = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.2,
    "RR_hip_joint": 0.0545,
    "FL_thigh_joint": -1.4,
    "FR_thigh_joint": -1.4,
    "RL_thigh_joint": 1.8,
    "RR_thigh_joint": 0.5673,
    "FL_calf_joint": -0.9,
    "FR_calf_joint": -0.9,
    "RL_calf_joint": -2.6,
    "RR_calf_joint": -1.3895,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/perfect_toe_jump_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=150, # 3.0秒
    name_prefix='perfect_toe_jump',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*80)
print("=== TESTING PURE TOE EXACT GROUND SPAWN & ACTIVE JUMP KICK ===")
print("="*80)

max_foot_height = 0.0
max_vz = 0.0
min_knee = 999.0

# 最初の 5ステップ静止 -> 6ステップ目から足先ゴム球で一気に床をフルキック！
for step in range(150):
    t = step * 0.02
    actions = torch.zeros((1, 3), device='cuda:0')
    
    if step >= 5 and step <= 25:
        # 膝を一気に伸展させて真上に爆発キック！
        actions[0, 1] = -0.60 # RR_thigh
        actions[0, 2] = +1.00 # RR_calf (最大伸展)
    elif step > 25:
        # 空中で足を抱え込む (膝クリアランス確保)
        actions[0, 1] = +0.20
        actions[0, 2] = -0.50
        
    obs, rewards, dones, truncated, infos = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    if rr_foot[2] > max_foot_height: max_foot_height = rr_foot[2]
    if root_vel[2] > max_vz: max_vz = root_vel[2]
    if rr_calf[2] < min_knee: min_knee = rr_calf[2]
    
    is_airborne = rr_foot[2] > 0.035
    status = "★★★ PURE TOE JUMP IN FLIGHT ★★★" if is_airborne else "Toe Push Ground"
    
    print(f"Step {step:03d} (t={t:.2f}s) | Body_Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | Foot_Z={rr_foot[2]*100:+5.1f}cm | Knee_Z={rr_calf[2]*100:+5.1f}cm | [{status}]")

print("\n" + "="*80)
print(f"=== PURE TOE EXACT GROUND LAUNCH RESULTS ===")
print(f"Max Vertical Launch Vz : {max_vz:+.2f} m/s")
print(f"Max Foot Flight Height : {max_foot_height*100:.1f} cm")
print(f"Min Knee Clearance     : {min_knee*100:.1f} cm (Knee Floor Clearance)")
print("="*80)

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/perfect_toe_jump-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/perfect_toe_jump.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

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
# 明るい標準スタジオ環境＆最適なカメラ位置
env_cfg.viewer.eye = (0.0, 1.8, 0.38)
env_cfg.viewer.lookat = (0.0, 0.0, 0.25)

# 求まった最適受動自立姿勢をセット
opt_roll = math.radians(19.0)
opt_pitch = math.radians(-13.0)
cr = math.cos(opt_roll * 0.5); sr = math.sin(opt_roll * 0.5)
cp = math.cos(opt_pitch * 0.5); sp = math.sin(opt_pitch * 0.5)
qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.420)
env_cfg.scene.robot.init_state.rot = (qx, qy, qz, qw)
env_cfg.scene.robot.init_state.joint_pos = {
    "FL_hip_joint": 0.30,
    "FR_hip_joint": -0.30,
    "RL_hip_joint": 0.30,
    "RR_hip_joint": 0.05,
    "FL_thigh_joint": -1.3,
    "FR_thigh_joint": -1.3,
    "RL_thigh_joint": 1.7,
    "RR_thigh_joint": 0.00, # 垂直真下
    "FL_calf_joint": -1.0,
    "FR_calf_joint": -1.0,
    "RL_calf_joint": -2.5,
    "RR_calf_joint": -0.838, # 最大伸展
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/passive_stand_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=150, # 3.0秒
    name_prefix='passive_stand_perfect_lighting',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*80)
print("=== RECORDING PASSIVE STANDING (PERFECT LIGHTING, ZERO ACTIVE MOVEMENT) ===")
print("="*80)

actions = torch.zeros((1, 3), device='cuda:0')
stand_steps = 0
fall_step = None

for step in range(150):
    obs, rewards, dones, truncated, infos = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    
    min_other = min(fl_foot[2], fr_foot[2], rl_foot[2])
    is_standing = rr_calf[2] > 0.05 and min_other > 0.05 and rr_foot[2] < 0.05
    
    if is_standing:
        stand_steps += 1
        status = "★ PERFECT PASSIVE STANDING ★"
    else:
        status = "Fallen"
        if fall_step is None:
            fall_step = step
            
    print(f"Step {step:03d} (t={step*0.02:.2f}s) | Body_Z={root_pos[2]*100:5.1f}cm | Foot_Z={rr_foot[2]*100:+5.1f}cm | Knee_Z={rr_calf[2]*100:+5.1f}cm | Other_Min_Z={min_other*100:+5.1f}cm | [{status}]")

print("\n" + "="*80)
print(f"=== PASSIVE STAND RESULT ===")
print(f"Passive Standing Duration: {stand_steps} steps ({stand_steps*0.02:.2f} seconds) without ANY movement!")
print("="*80)

env.close()
simulation_app.close()

# GIF & MP4 生成
src_video = f"{output_dir}/passive_stand_perfect_lighting-step-0.mp4"
if os.path.exists(src_video):
    cap = cv2.VideoCapture(src_video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/passive_stand_balance.gif"
        mp4_path = f"{output_dir}/passive_stand_balance.mp4"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        import shutil
        shutil.copy(src_video, mp4_path)
        print(f"Saved clear bright video to {mp4_path}")

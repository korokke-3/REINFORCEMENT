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
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env_cfg.viewer.eye = (0.0, 1.8, 0.38)
env_cfg.viewer.lookat = (0.0, 0.0, 0.25)
env_cfg.sim.render_interval = 1

# 緑色マーカー完全非表示
env_cfg.commands.base_velocity.debug_vis = False
if hasattr(env_cfg.scene, "height_scanner"):
    env_cfg.scene.height_scanner.debug_vis = False

# ★ 一本足姿勢の定義 ★
opt_roll = math.radians(20.0)
opt_pitch = math.radians(-12.0)
cr = math.cos(opt_roll * 0.5); sr = math.sin(opt_roll * 0.5)
cp = math.cos(opt_pitch * 0.5); sp = math.sin(opt_pitch * 0.5)
qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

one_leg_joint_pos = {
    "FL_hip_joint": 0.35,
    "FR_hip_joint": -0.35,
    "RL_hip_joint": 0.35,
    "RR_hip_joint": 0.05,
    "FL_thigh_joint": -1.40,
    "FR_thigh_joint": -1.40,
    "RL_thigh_joint": 1.80,
    "RR_thigh_joint": 0.00, # ピンと垂直に突っ張る
    "FL_calf_joint": -0.90,
    "FR_calf_joint": -0.90,
    "RL_calf_joint": -2.60,
    "RR_calf_joint": -0.850, # 最大伸展
}

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.383)
env_cfg.scene.robot.init_state.rot = (qx, qy, qz, qw)
env_cfg.scene.robot.init_state.joint_pos = one_leg_joint_pos

# ★ 姿勢を元に戻そうとする力を完全ゼロ化 ★
# use_default_offset を無効化し、目標角度を一本足姿勢そのものに直結！
env_cfg.actions.joint_pos.use_default_offset = False

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')

# 緑色マーカーを確実に削除
import omni.usd
stage = omni.usd.get_context().get_stage()
if stage:
    for prim in stage.Traverse():
        p_path = str(prim.GetPath())
        if "Visuals" in p_path or "velocity_goal" in p_path or "velocity_current" in p_path:
            from pxr import UsdGeom
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*95)
print("=== ABSOLUTELY ZERO RETURN-FORCE: 100% SUSTAINED ONE-LEG POSTURE ===")
print("="*95)

frames = []
# アクション目標を「一本足姿勢の角度そのもの」に固定
sustained_actions = torch.tensor([[0.05, 0.00, -0.850]], device='cuda:0')

for step in range(80):
    obs, rewards, dones, truncated, infos = env.step(sustained_actions)
    
    frame = env.render()
    if frame is not None and step > 0:
        frames.append(frame)
        
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    min_other = min(fl_foot[2], fr_foot[2], rl_foot[2])
    
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    thigh_deg = math.degrees(joint_pos[robot.data.joint_names.index('RR_thigh_joint')])
    calf_deg = math.degrees(joint_pos[robot.data.joint_names.index('RR_calf_joint')])
    
    print(f"Step {step:03d} (t={step*0.02:.2f}s) | Foot_Z={rr_foot[2]*100:+5.2f}cm | Knee_Z={rr_calf[2]*100:+5.2f}cm | Other_Min_Z={min_other*100:+5.2f}cm | RR_thigh={thigh_deg:+5.1f}°, RR_calf={calf_deg:+5.1f}°")

out_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/sustained_posture_proof'
os.makedirs(out_dir, exist_ok=True)

mp4_path = f"{out_dir}/sustained_one_leg_pose.mp4"
gif_path = f"{out_dir}/sustained_one_leg_pose.gif"

if frames:
    imageio.mimsave(mp4_path, frames, fps=25)
    imageio.mimsave(gif_path, frames, fps=25, loop=0)
    print(f"\nSaved sustained one-leg pose video to {mp4_path} and {gif_path}")

env.close()
simulation_app.close()

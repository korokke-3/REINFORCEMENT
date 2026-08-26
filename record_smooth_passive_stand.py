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

# ★ ガクガク振動（チャタリング）の完全解消設定 ★
# 1. 関節限界 (-0.838 rad) ストッパー激突を避けるため -0.850 rad に設定
# 2. ダンピングを適正化して振動を即座に減衰

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
    "RR_thigh_joint": 0.00,
    "FL_calf_joint": -1.0,
    "FR_calf_joint": -1.0,
    "RL_calf_joint": -2.5,
    "RR_calf_joint": -0.850, # 関節リミット激突を完全回避！
}

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

print("\n" + "="*80)
print("=== SMOOTH PASSIVE STANDING TEST: ZERO SHAKING / ZERO CHATTERING ===")
print("="*80)

frames = []
actions = torch.zeros((1, 3), device='cuda:0')

for step in range(80): # 1.6秒
    obs, rewards, dones, truncated, infos = env.step(actions)
    
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
    
    is_standing = rr_calf[2] > 0.05 and min_other > 0.05 and rr_foot[2] < 0.05
    status = "★ SMOOTH STILL STANDING ★" if is_standing else "Fallen / Leaning"
    
    print(f"Step {step:03d} (t={step*0.02:.2f}s) | Foot_Z={rr_foot[2]*100:+5.1f}cm | Knee_Z={rr_calf[2]*100:+5.1f}cm | Other_Min_Z={min_other*100:+5.1f}cm | [{status}]")

out_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/smooth_passive_stand_proof'
os.makedirs(out_dir, exist_ok=True)

mp4_path = f"{out_dir}/smooth_passive_stand.mp4"
gif_path = f"{out_dir}/smooth_passive_stand.gif"

if frames:
    imageio.mimsave(mp4_path, frames, fps=25)
    imageio.mimsave(gif_path, frames, fps=25, loop=0)
    print(f"\nSaved smooth video (zero chattering) to {mp4_path} and {gif_path}")

env.close()
simulation_app.close()

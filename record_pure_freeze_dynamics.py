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

# ★ 転倒しても一切リセットせず、倒れる一部始終をノーカット記録 ★
env_cfg.terminations.illegal_parts_contact = None
env_cfg.terminations.orientation_deviation = None
env_cfg.terminations.root_height_below_minimum = None
env_cfg.terminations.time_out = None
env_cfg.episode_length_s = 10.0

# ★ アクションによる復元トルクをゼロ化し、スポーン姿勢そのものを維持 ★
env_cfg.actions.joint_pos.use_default_offset = False

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
    "RR_calf_joint": -0.850,
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

print("\n" + "="*85)
print("=== PURE PASSIVE TOPPLING DYNAMICS: COMPLETE POSE FREEZE (NO RESET, 100% VISIBLE) ===")
print("="*85)

frames = []
# アクション目標を「スポーン姿勢」そのものに固定
spawn_actions = torch.tensor([[0.05, 0.00, -0.850]], device='cuda:0')

for step in range(120): # 2.4秒間ノーカット
    obs, rewards, dones, truncated, infos = env.step(spawn_actions)
    
    frame = env.render()
    if frame is not None and step > 0:
        frames.append(frame)
        
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    root_p = robot.data.root_pos_w[0].cpu().numpy()
    proj_g = robot.data.projected_gravity_b[0].cpu().numpy()
    
    print(f"Step {step:03d} (t={step*0.02:.2f}s) | Base_Z={root_p[2]*100:5.1f}cm | Foot_Z={rr_foot[2]*100:+5.1f}cm | Knee_Z={rr_calf[2]*100:+5.1f}cm | RL_Z={rl_foot[2]*100:+5.1f}cm | Roll/Pitch=({proj_g[0]:.2f}, {proj_g[1]:.2f})")

out_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/freeze_fall_proof'
os.makedirs(out_dir, exist_ok=True)

mp4_path = f"{out_dir}/freeze_pose_fall.mp4"
gif_path = f"{out_dir}/freeze_pose_fall.gif"

if frames:
    imageio.mimsave(mp4_path, frames, fps=25)
    imageio.mimsave(gif_path, frames, fps=25, loop=0)
    print(f"\nSaved 100% crystal clear freeze-fall video to {mp4_path} and {gif_path}")

env.close()
simulation_app.close()

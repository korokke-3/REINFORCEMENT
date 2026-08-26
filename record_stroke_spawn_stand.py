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
env_cfg.viewer.eye = (0.0, 1.8, 0.35)
env_cfg.viewer.lookat = (0.0, 0.0, 0.22)
env_cfg.sim.render_interval = 1

# 緑色マーカー完全非表示
env_cfg.commands.base_velocity.debug_vis = False
if hasattr(env_cfg.scene, "height_scanner"):
    env_cfg.scene.height_scanner.debug_vis = False

# ★ タメを持つスポーン姿勢 ★
opt_roll = math.radians(20.0)
opt_pitch = math.radians(-12.0)
cr = math.cos(opt_roll * 0.5); sr = math.sin(opt_roll * 0.5)
cp = math.cos(opt_pitch * 0.5); sp = math.sin(opt_pitch * 0.5)
qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

spawn_joints = {
    "FL_hip_joint": 0.35,
    "FR_hip_joint": -0.35,
    "RL_hip_joint": 0.35,
    "RR_hip_joint": 0.05,
    "FL_thigh_joint": -1.40,
    "FR_thigh_joint": -1.40,
    "RL_thigh_joint": 1.80,
    "RR_thigh_joint": -0.20,
    "FL_calf_joint": -0.90,
    "FR_calf_joint": -0.90,
    "RL_calf_joint": -2.60,
    "RR_calf_joint": -1.200, # タメ角 (-1.20 rad)
}

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.355)
env_cfg.scene.robot.init_state.rot = (qx, qy, qz, qw)
env_cfg.scene.robot.init_state.joint_pos = spawn_joints

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
contact_sensor = env.unwrapped.scene.sensors['contact_forces']

print("\n" + "="*95)
print("=== MEASURING PASSIVE STANDING DURATION OF PRE-CROUCH STROKE SPAWN POSE ===")
print("="*95)

frames = []
# アクション目標をスポーン姿勢そのものに完全固定（受動静止）
actions = torch.zeros((1, 3), device='cuda:0')
stand_steps = 0
fall_step = None

for step in range(80): # 1.6秒間
    obs, rewards, dones, truncated, infos = env.step(actions)
    
    frame = env.render()
    if frame is not None and step > 0:
        frames.append(frame)
        
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_knee = body_pos[robot.data.body_names.index('RR_calf')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    min_other = min(fl_foot[2], fr_foot[2], rl_foot[2])
    root_p = robot.data.root_pos_w[0].cpu().numpy()
    
    forces = contact_sensor.data.net_forces_w[0].cpu().numpy()
    f_knee = np.linalg.norm(forces[robot.data.body_names.index('RR_calf')])
    
    # 膝やすね・他脚が床に触れていないか判定
    is_standing = rr_knee[2] > 0.04 and min_other > 0.05 and f_knee < 1.0
    if is_standing:
        stand_steps += 1
        status = "★ BALANCED STILL STANDING ★"
    else:
        status = "Fallen / Leaning"
        if fall_step is None:
            fall_step = step
            
    print(f"Step {step:03d} (t={step*0.02:.2f}s) | Base_Z={root_p[2]*100:5.1f}cm | Foot_Z={rr_foot[2]*100:+5.2f}cm | Knee_Z={rr_knee[2]*100:+5.2f}cm (F={f_knee:4.1f}N) | Other_Min_Z={min_other*100:+5.1f}cm | [{status}]")

print("\n" + "="*95)
print(f"=== PRE-CROUCH STROKE SPAWN POSE RESULTS ===")
print(f"Total Passive Standing Duration: {stand_steps} steps ({stand_steps*0.02:.2f} seconds) without ANY active movement!")
print("="*95)

out_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/stroke_spawn_proof'
os.makedirs(out_dir, exist_ok=True)

mp4_path = f"{out_dir}/stroke_spawn_stand.mp4"
gif_path = f"{out_dir}/stroke_spawn_stand.gif"

if frames:
    imageio.mimsave(mp4_path, frames, fps=25)
    imageio.mimsave(gif_path, frames, fps=25, loop=0)
    print(f"\nSaved stroke spawn video to {mp4_path} and {gif_path}")

env.close()
simulation_app.close()

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

opt_roll = math.radians(20.0)
opt_pitch = math.radians(-12.0)
cr = math.cos(opt_roll * 0.5); sr = math.sin(opt_roll * 0.5)
cp = math.cos(opt_pitch * 0.5); sp = math.sin(opt_pitch * 0.5)
qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

one_leg_joint_dict = {
    "FL_hip_joint": 0.35,
    "FR_hip_joint": -0.35,
    "RL_hip_joint": 0.35,
    "RR_hip_joint": 0.05,
    "FL_thigh_joint": -1.40,
    "FR_thigh_joint": -1.40,
    "RL_thigh_joint": 1.80,
    "RR_thigh_joint": 0.00,
    "FL_calf_joint": -0.90,
    "FR_calf_joint": -0.90,
    "RL_calf_joint": -2.60,
    "RR_calf_joint": -0.850,
}

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.383)
env_cfg.scene.robot.init_state.rot = (qx, qy, qz, qw)
env_cfg.scene.robot.init_state.joint_pos = one_leg_joint_dict

# ★ アクション目標空間を全12関節に拡張し、全関節の PD 目標を一本足姿勢そのものに完全ロック！ ★
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ActionsCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass

@configclass
class All12JointsLockActionsCfg(ActionsCfg):
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"], # 全12関節すべて
        scale=1.0,
        use_default_offset=False, # デフォルト姿勢への復元力を完全無効化！
    )

env_cfg.actions = All12JointsLockActionsCfg()

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

# 全12関節の目標角度テンソルを構築
joint_names = robot.data.joint_names
target_12_joints = torch.zeros((1, 12), device='cuda:0')
for idx, name in enumerate(joint_names):
    target_12_joints[0, idx] = one_leg_joint_dict[name]

# ロボット内部の default_joint_pos も一本足姿勢に完全上書き
robot.data.default_joint_pos[:] = target_12_joints

print("\n" + "="*115)
print("=== ABSOLUTE 12-JOINT LOCK: ZERO RESTORING TORQUE TO 4-LEG STANDING ===")
print("="*115)

frames = []
for step in range(80):
    # 全12関節に対して一本足姿勢を完全ロック指令
    obs, rewards, dones, truncated, infos = env.step(target_12_joints)
    
    frame = env.render()
    if frame is not None and step > 0:
        frames.append(frame)
        
    current_joints = robot.data.joint_pos[0].cpu().numpy()
    fl_th = math.degrees(current_joints[joint_names.index('FL_thigh_joint')])
    fr_th = math.degrees(current_joints[joint_names.index('FR_thigh_joint')])
    rl_th = math.degrees(current_joints[joint_names.index('RL_thigh_joint')])
    rr_th = math.degrees(current_joints[joint_names.index('RR_thigh_joint')])
    rr_calf = math.degrees(current_joints[joint_names.index('RR_calf_joint')])
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_knee = body_pos[robot.data.body_names.index('RR_calf')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    min_other = min(fl_foot[2], fr_foot[2], rl_foot[2])
    
    print(f"Step {step:03d} (t={step*0.02:.2f}s) | RR_thigh={rr_th:+5.1f}°, RR_calf={rr_calf:+5.1f}° | FL_thigh={fl_th:+5.1f}°, FR_thigh={fr_th:+5.1f}°, RL_thigh={rl_th:+5.1f}° | Foot_Z={rr_foot[2]*100:+5.2f}cm | Other_Min_Z={min_other*100:+5.2f}cm")

out_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/absolute_lock_proof'
os.makedirs(out_dir, exist_ok=True)

mp4_path = f"{out_dir}/absolute_lock_one_leg.mp4"
gif_path = f"{out_dir}/absolute_lock_one_leg.gif"

if frames:
    imageio.mimsave(mp4_path, frames, fps=25)
    imageio.mimsave(gif_path, frames, fps=25, loop=0)
    print(f"\nSaved absolute 12-joint locked video to {mp4_path} and {gif_path}")

env.close()
simulation_app.close()

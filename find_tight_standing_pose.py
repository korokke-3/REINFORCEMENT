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

# 1. 姿勢探索グリッド (右後ろ脚は完全伸展・突っ張り固定)
print("\n" + "="*85)
print("=== OPTIMIZING PERFECT STILL STANDING POSE (TIGHT RIGHT-REAR LEG, HIGH OTHER LEGS) ===")
print("="*85)

# ロール角 (+15° ~ +25°), ピッチ角 (-18° ~ -8°), ベース高さ (41.0cm ~ 43.0cm), 他脚バランス角
roll_grid = np.linspace(math.radians(16.0), math.radians(24.0), 9)
pitch_grid = np.linspace(math.radians(-16.0), math.radians(-8.0), 9)
z_grid = np.linspace(0.412, 0.428, 5)

best_stand_steps = 0
best_pose = None

env_cfg = Go2SingleLegEnvCfg_PLAY()
env_cfg.commands.base_velocity.debug_vis = False
if hasattr(env_cfg.scene, "height_scanner"):
    env_cfg.scene.height_scanner.debug_vis = False

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
robot = env.unwrapped.scene['robot']

for r in roll_grid:
    for p in pitch_grid:
        for z in z_grid:
            cr = math.cos(r * 0.5); sr = math.sin(r * 0.5)
            cp = math.cos(p * 0.5); sp = math.sin(p * 0.5)
            qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

            root_pos = torch.tensor([[0.0, 0.0, z]], dtype=torch.float32, device='cuda:0')
            root_rot = torch.tensor([[qx, qy, qz, qw]], dtype=torch.float32, device='cuda:0')
            joint_pos = torch.tensor([[
                0.35, -0.35, 0.35, 0.05,
                -1.40, -1.40, 1.80, 0.00, # RR_thigh=0.0 (垂直突っ張り)
                -0.90, -0.90, -2.60, -0.850 # RR_calf=-0.850 (最大伸展ピン張り)
            ]], dtype=torch.float32, device='cuda:0')

            robot.write_root_pose_to_sim(torch.cat([root_pos, root_rot], dim=-1))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device='cuda:0'))
            robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
            env.unwrapped.scene.reset()

            steps = 0
            for step in range(100):
                # 関節角度を完全維持
                robot.set_joint_position_target(joint_pos)
                robot.write_data_to_sim()
                env.unwrapped.sim.step()
                env.unwrapped.scene.update(0.005)

                body_pos = robot.data.body_pos_w[0].cpu().numpy()
                rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
                rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
                fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
                fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
                rl_foot = body_pos[robot.data.body_names.index('RL_foot')]

                min_other = min(fl_foot[2], fr_foot[2], rl_foot[2])
                if rr_calf[2] > 0.05 and min_other > 0.08 and rr_foot[2] < 0.04:
                    steps += 1
                else:
                    break

            if steps > best_stand_steps:
                best_stand_steps = steps
                best_pose = (r, p, z)
                print(f"★ Improved Stand Time: {best_stand_steps} steps ({best_stand_steps*0.005:.3f}s) | Roll={math.degrees(r):.1f}°, Pitch={math.degrees(p):.1f}°, Z={z*100:.1f}cm")

opt_r, opt_p, opt_z = best_pose
print("\n" + "="*85)
print(f"=== BEST STANDING POSE FOUND ===")
print(f"Roll: {math.degrees(opt_r):.2f}°, Pitch: {math.degrees(opt_p):.2f}°, Base Z: {opt_z*100:.2f}cm")
print(f"Duration: {best_stand_steps} sub-steps ({best_stand_steps*0.005:.3f}s)")
print("="*85)

# 最適姿勢でのクリアな動画・GIF生成
out_dir = "/home/exhibition-spakona/Desktop/REINFORCEMENT/tight_stand_proof"
os.makedirs(out_dir, exist_ok=True)

cr = math.cos(opt_r * 0.5); sr = math.sin(opt_r * 0.5)
cp = math.cos(opt_p * 0.5); sp = math.sin(opt_p * 0.5)
qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, opt_z)
env_cfg.scene.robot.init_state.rot = (qx, qy, qz, qw)
env_cfg.scene.robot.init_state.joint_pos = {
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

env.close()

# 新しい環境で動画撮影
env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

frames = []
actions = torch.tensor([[0.05, 0.00, -0.850]], device='cuda:0')

for step in range(80):
    obs, rewards, dones, truncated, infos = env.step(actions)
    frame = env.render()
    if frame is not None and step > 0:
        frames.append(frame)

mp4_path = f"{out_dir}/tight_stand_still.mp4"
gif_path = f"{out_dir}/tight_stand_still.gif"

if frames:
    imageio.mimsave(mp4_path, frames, fps=25)
    imageio.mimsave(gif_path, frames, fps=25, loop=0)
    print(f"\nSaved tight stand still video to {mp4_path} and {gif_path}")

env.close()
simulation_app.close()

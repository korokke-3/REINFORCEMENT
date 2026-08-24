import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import envs
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env_cfg.terminations.illegal_parts_contact = None
env_cfg.terminations.forward_fall = None
env_cfg.terminations.root_height_below_minimum = None
env_cfg.terminations.base_contact = None

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)
obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*50)
print("=== GO2 ROBOT DYNAMICS & PARAMETER ANALYSIS ===")
print("="*50)

# 質量
masses = robot.data.default_mass[0].cpu().numpy()
body_names = robot.data.body_names
joint_names = robot.data.joint_names
total_mass = np.sum(masses)
print(f"Total Robot Mass: {total_mass:.2f} kg")

for name, mass in zip(body_names, masses):
    print(f"  - Body: {name:<15} Mass: {mass:.3f} kg")

print("\n--- Joint Names & Default Positions ---")
for idx, name in enumerate(joint_names):
    default_pos = robot.data.default_joint_pos[0, idx].item()
    print(f"  - Joint [{idx:02d}]: {name:<18} Default Pos: {default_pos:+.3f} rad ({np.degrees(default_pos):+.1f} deg)")

print("\n--- Robot CoM & Foot Positions in Base Frame ---")
root_pos = robot.data.root_pos_w[0].cpu().numpy()
body_pos = robot.data.body_pos_w[0].cpu().numpy()
for name in ["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
    idx = body_names.index(name)
    pos = body_pos[idx] - root_pos
    print(f"  - {name:<10} Relative Pos (X, Y, Z): ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m")

env.close()
simulation_app.close()

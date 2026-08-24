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
# 終了判定を一旦オフにして足先の位置を計測
env_cfg.terminations.illegal_parts_contact = None
env_cfg.terminations.orientation_deviation = None
env_cfg.terminations.root_height_below_minimum = None

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)
obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

body_pos = robot.data.body_pos_w[0].cpu().numpy()
root_pos = robot.data.root_pos_w[0].cpu().numpy()
rr_foot = body_pos[robot.data.body_names.index('RR_foot')]

print(f"Current Root Pos Z   : {root_pos[2]:.4f} m")
print(f"Current RR Foot Pos Z: {rr_foot[2]:.4f} m")

# 地面接地目標 Z = 0.0235 m (足先球半径)
foot_ground_target = 0.0235
required_root_z = root_pos[2] + (foot_ground_target - rr_foot[2])
print(f"Required Root Pos Z for perfect ground contact: {required_root_z:.4f} m")

env.close()
simulation_app.close()

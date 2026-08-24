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
env_cfg.terminations.orientation_deviation = None
env_cfg.terminations.root_height_below_minimum = None

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)
obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

body_names = robot.data.body_names
body_pos = robot.data.body_pos_w[0].cpu().numpy()
root_pos = robot.data.root_pos_w[0].cpu().numpy()

print(f"Root Pos: {root_pos}")
for name, pos in zip(body_names, body_pos):
    print(f"  Link '{name:<15}': Pos = [{pos[0]:+6.3f}, {pos[1]:+6.3f}, {pos[2]:+6.3f}]")

env.close()
simulation_app.close()

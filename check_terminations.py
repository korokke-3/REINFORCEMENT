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
import envs.go2_single_leg_rewards as custom_rewards
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg

env_cfg = Go2SingleLegEnvCfg_PLAY()
env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*60)
print("=== CHECKING EXACT TERMINATIONS & CONTACT FORCES ===")
print("="*60)

# 1. illegal_parts_contact
sensor_cfg = SceneEntityCfg("contact_forces", body_names=["Head_.*", "FL_.*", "FR_.*", "RL_.*", "RR_thigh", "base"])
c_sensor = env.unwrapped.scene.sensors['contact_forces']
body_ids, body_names = c_sensor.find_bodies(sensor_cfg.body_names)
forces = torch.norm(c_sensor.data.net_forces_w[:, body_ids, :], dim=-1)
print(f"Illegal Parts Forces: {forces}")
for b_name, f in zip(body_names, forces[0]):
    if f > 0.1:
        print(f"  - WARNING! Force on '{b_name}': {f.item():.2f} N")

# 2. orientation
is_rot_done = custom_rewards.orientation_deviation_termination(env.unwrapped)
print(f"Orientation deviation termination: {is_rot_done}")

# 3. height
is_h_done = mdp.root_height_below_minimum(env.unwrapped, minimum_height=0.14)
print(f"Height below minimum termination: {is_h_done}")

env.close()
simulation_app.close()

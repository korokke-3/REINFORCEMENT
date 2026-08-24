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
env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']
term_mgr = env.unwrapped.termination_manager

print("\n" + "="*60)
print("=== CHECKING TERMINATION TERM VALUES AT STEP 0 ===")
print("="*60)

for name, term in term_mgr._terms.items():
    res = term.func(env.unwrapped, **term.params)
    print(f"Term '{name:<25}': Result = {res}")

contact_sensor = env.unwrapped.scene.sensors['contact_forces']
forces = torch.norm(contact_sensor.data.net_forces_w[0], dim=-1)
for b_name, f in zip(contact_sensor.data.body_names, forces):
    print(f"  - Contact Force on '{b_name:<15}': {f.item():.2f} N")

env.close()
simulation_app.close()

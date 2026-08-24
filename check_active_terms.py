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

print("\n" + "="*60)
print("=== CHECKING ALL ACTIVE TERMINATIONS ===")
print("="*60)

term_mgr = env.unwrapped.termination_manager
for term_name in term_mgr._term_names:
    val = term_mgr.get_term(term_name)
    print(f"Termination Term '{term_name:<25}': {val}")

env.close()
simulation_app.close()

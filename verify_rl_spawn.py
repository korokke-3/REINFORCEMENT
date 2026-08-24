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

print("\n" + "="*60)
print("=== VERIFYING SPAWN AND STRICT TERMINATION CONDITIONS ===")
print("="*60)

terminated_steps = []
for step in range(50):
    # アクションなし（初期姿勢維持）: 支持脚3関節
    actions = torch.zeros((1, 3), device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    
    print(f"Step {step:02d} | Body_Z={root_pos[2]*100:.1f}cm, Foot_Z={rr_foot[2]*100:.1f}cm | Reward={reward.item():+.3f} | Done={terminated}")
    if terminated:
        print(f">>> Terminated at step {step} <<<")
        break

env.close()
simulation_app.close()
print("Verification complete.")

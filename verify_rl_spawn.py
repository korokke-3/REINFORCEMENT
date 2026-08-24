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
import envs
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)
obs, _ = env.reset()
robot = env.unwrapped.scene['robot']
contact_sensor = env.unwrapped.scene.sensors['contact_forces']

print("\n" + "="*70)
print("=== VERIFYING STRICT SINGLE-LEG BALANCE ENVIRONMENT ===")
print("="*70)

for step in range(30):
    actions = torch.zeros((1, 3), device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    base_pos = body_pos[robot.data.body_names.index('base')]
    other_min_z = min(fl_foot[2], fr_foot[2], rl_foot[2], base_pos[2])
    
    print(f"Step {step:02d} | Body_Z={root_pos[2]*100:4.1f}cm | RR_Foot={rr_foot[2]*100:+4.1f}cm | Other_Min_Z={other_min_z*100:+4.1f}cm | Rew={reward.item():+5.2f} | Done={terminated.item()}")
    if terminated.item():
        print(f">>> Reset at step {step} <<<")
        break

env.close()
simulation_app.close()

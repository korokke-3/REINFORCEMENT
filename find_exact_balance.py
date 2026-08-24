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
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

import envs
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)
obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

body_names = robot.data.body_names
masses = robot.data.default_mass[0].cpu().numpy()
total_mass = np.sum(masses)

# 各ボディの base フレームでの相対位置を取得
root_pos = robot.data.root_pos_w[0].cpu().numpy()
body_pos_w = robot.data.body_pos_w[0].cpu().numpy()
rel_body_pos = body_pos_w - root_pos

# 固定する他3脚の関節角
fixed_joints = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,
    'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,
}

print(f"Total mass: {total_mass:.3f} kg")

# ロボット全体のCoMを計算する関数 (base基準)
def compute_com_base():
    com = np.zeros(3)
    for idx, (name, m) in enumerate(zip(body_names, masses)):
        pos = robot.data.body_pos_w[0, idx].cpu().numpy()
        com += m * pos
    com /= total_mass
    return com

com_w = compute_com_base()
rr_foot_w = robot.data.body_pos_w[0, body_names.index('RR_foot')].cpu().numpy()
print(f"Current CoM World Pos : {com_w}")
print(f"Current RR Foot Pos   : {rr_foot_w}")
print(f"Offset (CoM - Foot)   : {com_w - rr_foot_w}")

env.close()
simulation_app.close()

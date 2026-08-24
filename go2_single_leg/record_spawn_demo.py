import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record Spawn Pose Demo")
parser.add_argument("--video_length", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless', '--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import cv2
import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg

import envs
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
# 重心直下直立スポーン
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.35)
env_cfg.scene.robot.init_state.rot = (-0.1527, -0.2138, -0.0339, 0.9643)
env_cfg.scene.robot.init_state.joint_pos = {
    'FL_hip_joint': 0.2,
    'FR_hip_joint': -0.1,
    'RL_hip_joint': 0.2,
    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,
    'FR_thigh_joint': -1.2,
    'RL_thigh_joint': 0.0,
    'RR_thigh_joint': 0.8,
    'FL_calf_joint': -1.0,
    'FR_calf_joint': -1.0,
    'RL_calf_joint': -1.0,
    'RR_calf_joint': -1.5,
}

env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0", cfg=env_cfg, render_mode="rgb_array")
video_folder = "/home/exhibition-spakona/Desktop/REINFORCEMENT/spawn_evidence"
os.makedirs(video_folder, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=video_folder,
    step_trigger=lambda step: step == 0,
    video_length=100,
    name_prefix="spawn_pose_evidence",
)

obs, _ = env.reset()

for step in range(100):
    # アクションなし（初期直立姿勢で自然着地とバランスの推移を記録）
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
simulation_app.close()
print("Evidence recorded successfully in /home/exhibition-spakona/Desktop/REINFORCEMENT/spawn_evidence")

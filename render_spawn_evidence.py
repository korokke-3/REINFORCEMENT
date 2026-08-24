import sys
import os
import torch
import numpy as np
import cv2

WORKSPACE = "/home/exhibition-spakona/Desktop/REINFORCEMENT"
sys.path.append(os.path.join(WORKSPACE, "go2_single_leg"))

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless', '--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import envs
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY
from isaaclab.sensors import TiledCameraCfg, TiledCamera

cfg = Go2SingleLegEnvCfg_PLAY()
# 重心が右後足の真上に乗った直立ポーズ
cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.35)
cfg.scene.robot.init_state.rot = (-0.1527, -0.2138, -0.0339, 0.9643)
cfg.scene.robot.init_state.joint_pos = {
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

# ロボットの全体像がはっきり見えるカメラアングル
cfg.scene.tiled_camera = TiledCameraCfg(
    prim_path='{ENV_REGEX_NS}/Camera',
    offset=TiledCameraCfg.OffsetCfg(pos=(-1.8, -1.8, 0.8), rot=(0.9238, 0.0, 0.0, 0.3826), convention='world'),
    data_types=['rgb'],
    spawn=None,
    width=1280,
    height=720,
)

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=cfg)
env.reset()

camera: TiledCamera = env.unwrapped.scene['tiled_camera']

# 1. スポーン直後 (Step 0) の静止画像をキャプチャ
for _ in range(5):
    env.sim.render()
    camera.update(0.02)

rgb_step0 = camera.data.output['rgb'][0].cpu().numpy()
cv2.imwrite('/home/exhibition-spakona/Desktop/REINFORCEMENT/spawn_pose_step0.png', cv2.cvtColor(rgb_step0, cv2.COLOR_RGB2BGR))
print('Saved spawn_pose_step0.png')

# 2. スポーンから自然着地・初期挙動 (60フレーム = 1.2秒) の動画を保存
frames = []
for step in range(60):
    # アクションなし（初期姿勢のまま自然落下・着地を見る）
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    obs, r, term, trunc, info = env.step(actions)
    env.sim.render()
    camera.update(0.02)
    rgb = camera.data.output['rgb'][0].cpu().numpy()
    frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# MP4動画として書き出し
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/exhibition-spakona/Desktop/REINFORCEMENT/spawn_pose_evidence.mp4', fourcc, 30.0, (1280, 720))
for f in frames:
    out.write(f)
out.release()
print('Saved spawn_pose_evidence.mp4')

env.close()
simulation_app.close()

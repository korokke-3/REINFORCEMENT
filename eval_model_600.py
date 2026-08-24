import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless', '--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import cv2
import imageio
import gymnasium as gym
import torch
import numpy as np

from rsl_rl.modules import ActorCritic
import envs
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

checkpoint_path = "/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg/logs/rsl_rl/2026-08-24_00-00-57/model_600.pt"

env_cfg = Go2SingleLegEnvCfg_PLAY()
# テスト時は終了判定を少し緩めて挙動を見る
env_cfg.viewer.eye = (0.0, 1.8, 0.45)
env_cfg.viewer.lookat = (0.0, 0.0, 0.25)

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/eval_model_600'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=100,
    name_prefix='eval_model_600',
)

obs, _ = env.reset()

# モデルロード
actor_critic = ActorCritic(
    num_actor_obs=235,
    num_critic_obs=235,
    num_actions=12,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation='elu',
).to(env.unwrapped.device)

loaded_dict = torch.load(checkpoint_path, map_location=env.unwrapped.device, weights_only=True)
actor_critic.load_state_dict(loaded_dict['model_state_dict'])
actor_critic.eval()

robot = env.unwrapped.scene['robot']

print("\n" + "="*60)
print(f"=== EVALUATING MODEL_600.PT POLICY ===")
print("="*60)

ep_lengths = []
cur_len = 0

for step in range(100):
    with torch.no_grad():
        policy_obs = obs['policy']
        actions = actor_critic.act_inference(policy_obs)
    
    obs, reward, terminated, truncated, info = env.step(actions)
    cur_len += 1
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    
    print(f"Step {step:02d} | Body_Z={root_pos[2]*100:5.1f}cm | Foot_Z={rr_foot[2]*100:+5.1f}cm | Reward={reward.item():+.3f} | Done={terminated.item()}")
    
    if terminated or truncated:
        ep_lengths.append(cur_len)
        print(f"  >>> Episode Ended at length {cur_len} <<<")
        cur_len = 0

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/eval_model_600-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/eval_model_600.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved eval animation GIF to {gif_path}")

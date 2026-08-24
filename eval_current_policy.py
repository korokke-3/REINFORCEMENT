from __future__ import annotations
import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
import os
import cv2
import imageio
import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless', '--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata

import envs
from agents.rsl_rl_ppo_cfg import UnitreeGo2SingleLegPPORunnerCfg
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env_cfg.viewer.eye = (0.0, 1.8, 0.45)
env_cfg.viewer.lookat = (0.0, 0.0, 0.25)

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/eval_current_policy'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=150,
    name_prefix='eval_current_policy',
)

env_wrapped = RslRlVecEnvWrapper(env)
agent_cfg = UnitreeGo2SingleLegPPORunnerCfg()
try:
    rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
except Exception:
    rsl_rl_version = importlib.metadata.version("rsl_rl")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
ckpt_path = "/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg/logs/rsl_rl/2026-08-24_00-00-57/model_600.pt"
runner.load(ckpt_path)
policy = runner.get_inference_policy(device="cuda:0")

obs, _ = env_wrapped.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*65)
print("=== EVALUATION OF CURRENT SINGLE-LEG RL POLICY (ITER 600) ===")
print("="*65)

survived_steps = 0
resets = 0

for step in range(150):
    with torch.no_grad():
        actions = policy(obs)
    obs, rewards, dones, infos = env_wrapped.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    
    is_done = dones[0].item()
    print(f"Step {step:03d} | Body_Z={root_pos[2]*100:5.1f}cm | Foot_Z={rr_foot[2]*100:+5.1f}cm | Reward={rewards[0].item():+.3f} | Done={is_done}")
    
    if is_done:
        resets += 1
        print(f"  >>> Reset #{resets} (Episode Survived: {survived_steps} steps = {survived_steps*0.02:.2f}s) <<<")
        survived_steps = 0
    else:
        survived_steps += 1

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/eval_current_policy-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/eval_current_policy.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved eval animation GIF to {gif_path}")

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
env_cfg.viewer.eye = (0.0, 2.2, 0.50)
env_cfg.viewer.lookat = (0.0, 0.0, 0.28)

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/true_single_leg_hopping_eval'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=200, # 4.0秒
    name_prefix='true_single_leg_hopping_policy',
)

env_wrapped = RslRlVecEnvWrapper(env)
agent_cfg = UnitreeGo2SingleLegPPORunnerCfg()
try:
    rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
except Exception:
    rsl_rl_version = importlib.metadata.version("rsl_rl")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
ckpt_path = "/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg/logs/rsl_rl/2026-08-24_17-33-29/model_999.pt"
runner.load(ckpt_path)
policy = runner.get_inference_policy(device="cuda:0")

obs, _ = env_wrapped.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*80)
print("=== EVALUATION OF TRUE SINGLE-LEG FORWARD HOPPING RL POLICY ===")
print("="*80)

survived_steps = 0
resets = 0
max_survival = 0
total_air_steps = 0
forward_dist = 0.0
init_x = robot.data.root_pos_w[0, 0].item()

for step in range(200):
    with torch.no_grad():
        actions = policy(obs)
    obs, rewards, dones, infos = env_wrapped.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    other_min_z = min(fl_foot[2], fr_foot[2], rl_foot[2])
    is_airborne = rr_foot[2] > 0.035
    if is_airborne:
        total_air_steps += 1
    
    is_done = dones[0].item()
    status = "★ HOPPING (In Air) ★" if is_airborne else "Pushed Ground"
    if is_done:
        status = "RESET"
    
    curr_dist = root_pos[0] - init_x
    if curr_dist > forward_dist:
        forward_dist = curr_dist
        
    print(f"Step {step:03d} (t={step*0.02:.2f}s) | Pos_X={root_pos[0]:+5.2f}m | Vx={root_vel[0]:+5.2f}m/s | Foot_Z={rr_foot[2]*100:+5.1f}cm | Other_Min_Z={other_min_z*100:+5.1f}cm | [{status}]")
    
    if is_done:
        resets += 1
        if survived_steps > max_survival:
            max_survival = survived_steps
        print(f"  >>> Reset #{resets} (Episode Survived: {survived_steps} steps = {survived_steps*0.02:.2f}s, Travel: {curr_dist:.2f}m) <<<")
        survived_steps = 0
        init_x = robot.data.root_pos_w[0, 0].item()
    else:
        survived_steps += 1

if survived_steps > max_survival:
    max_survival = survived_steps

print("\n" + "="*80)
print(f"=== TRUE SINGLE-LEG HOPPING EVALUATION RESULTS ===")
print(f"Total Steps Evaluated  : 200 (4.00 s)")
print(f"Max Continuous Survival: {max_survival} steps ({max_survival*0.02:.2f} s)")
print(f"Max Forward Travel Dist: {forward_dist:.2f} m")
print(f"Total Airborne Steps   : {total_air_steps} steps ({total_air_steps*0.02:.2f} s in air)")
print("="*80)

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/true_single_leg_hopping_policy-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/true_single_leg_hopping.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved eval animation GIF to {gif_path}")

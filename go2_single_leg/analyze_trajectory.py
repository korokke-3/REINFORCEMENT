import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import os

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Analyze True Single-Leg Hopping Trajectory")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import envs
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY
from rsl_rl.runners import OnPolicyRunner
import glob

# 最新モデルのロード
log_dirs = sorted(glob.glob("/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg/logs/rsl_rl/*"))
latest_dir = log_dirs[-1]
latest_model = sorted(glob.glob(os.path.join(latest_dir, "model_*.pt")))[-1]
print(f"Loading: {latest_model}")

env_cfg = Go2SingleLegEnvCfg_PLAY()
env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0", cfg=env_cfg)

agent_cfg = {
    "class_name": "ActorCritic",
    "init_noise_std": 1.0,
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu"
}
policy = torch.jit.load(latest_model.replace("model_", "exported/policy_")) if os.path.exists(latest_model.replace("model_", "exported/policy_")) else None

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

steps = 300
rr_foot_traj = []
root_z_traj = []
air_time_traj = []

for step in range(steps):
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_z = robot.data.root_pos_w[0, 2].item()
    
    rr_foot_traj.append(rr_foot.copy())
    root_z_traj.append(root_z)
    
    if terminated or truncated:
        obs, _ = env.reset()

rr_foot_traj = np.array(rr_foot_traj)
root_z_traj = np.array(root_z_traj)

# 右後足の真の移動距離 (Foot Displacement)
foot_disp = np.linalg.norm(rr_foot_traj[-1, :2] - rr_foot_traj[0, :2])

print(f"=== TRUE FOOT-BASED HOPPING ANALYSIS ===")
print(f"Right Foot Start Pos : ({rr_foot_traj[0,0]:.4f}, {rr_foot_traj[0,1]:.4f}, {rr_foot_traj[0,2]:.4f})")
print(f"Right Foot End Pos   : ({rr_foot_traj[-1,0]:.4f}, {rr_foot_traj[-1,1]:.4f}, {rr_foot_traj[-1,2]:.4f})")
print(f"TRUE FOOT DISPLACEMENT : {foot_disp:.4f} m")
print(f"Base Height Mean Z   : {np.mean(root_z_traj):.4f} m")

# プロット作成
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(np.arange(steps)*0.02, root_z_traj, 'b-', label='Base Height Z (m)')
plt.axhline(0.35, color='g', linestyle='--', label='Target Upright (0.35m)')
plt.ylabel('Height (m)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(steps)*0.02, rr_foot_traj[:, 2], 'r-', label='RR_foot Height (Air Time Indicator)')
plt.axhline(0.02, color='k', linestyle='--', label='Ground Threshold')
plt.ylabel('Foot Z (m)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg/logs/trajectory_analysis.png")
print("Saved true foot hopping analysis plot.")

env.close()
simulation_app.close()

from __future__ import annotations
import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata

import envs
from agents.rsl_rl_ppo_cfg import UnitreeGo2SingleLegPPORunnerCfg
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)
env_wrapped = RslRlVecEnvWrapper(env)

agent_cfg = UnitreeGo2SingleLegPPORunnerCfg()
try:
    rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
except Exception:
    rsl_rl_version = importlib.metadata.version("rsl_rl")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
ckpt_path = "/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg/logs/rsl_rl/2026-08-26_18-22-46/model_2499.pt"
runner.load(ckpt_path)
policy = runner.get_inference_policy(device="cuda:0")

obs, _ = env_wrapped.reset()
robot = env.unwrapped.scene['robot']
contact_sensor = env.unwrapped.scene.sensors['contact_forces']

print("\n" + "="*85)
print("=== EXACT RESET CAUSE DIAGNOSIS FOR PURE TOE POLICY ===")
print("="*85)

for step in range(30):
    with torch.no_grad():
        actions = policy(obs)
    obs, rewards, dones, infos = env_wrapped.step(actions)
    
    # 接触力の詳細
    forces = contact_sensor.data.net_forces_w[0].cpu().numpy() # (num_bodies, 3)
    body_names = robot.data.body_names
    
    high_forces = []
    for idx, name in enumerate(body_names):
        f_norm = torch.norm(torch.tensor(forces[idx])).item()
        if f_norm > 1.0:
            high_forces.append(f"{name}={f_norm:.1f}N")
            
    proj_g = robot.data.projected_gravity_b[0].cpu().numpy()
    is_done = dones[0].item()
    
    print(f"Step {step:02d} | Done={is_done} | proj_g=({proj_g[0]:.2f}, {proj_g[1]:.2f}, {proj_g[2]:.2f}) | Contacts: {high_forces}")

env.close()
simulation_app.close()

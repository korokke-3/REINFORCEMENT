"""
Unitree Go2 後足2本立ち歩行 (Bipedal Locomotion) 強化学習トレーニングスクリプト
Isaac Lab + RSL-RL (PPO)
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Go2 Bipedal Policy with RSL-RL")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments to simulate")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-Bipedal-v0", help="Task name")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=None, help="Max RL iterations")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.headless:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata

import envs
from agents.rsl_rl_ppo_cfg import UnitreeGo2BipedalPPORunnerCfg
from envs.go2_bipedal_env_cfg import Go2BipedalEnvCfg


def main():
    env_cfg = Go2BipedalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = UnitreeGo2BipedalPPORunnerCfg()
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    try:
        rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    except Exception:
        rsl_rl_version = importlib.metadata.version("rsl_rl")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    log_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs", "rsl_rl"))
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Training directory: {log_dir}")
    print(f"[INFO] Simulating {args_cli.num_envs} Go2 robots concurrently on GPU (Bipedal)...")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    print("[INFO] Bipedal training finished successfully!")


if __name__ == "__main__":
    main()
    simulation_app.close()

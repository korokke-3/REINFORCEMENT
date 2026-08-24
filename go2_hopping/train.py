"""
Unitree Go2 片足けんけん 強化学習トレーニングスクリプト
Isaac Lab + RSL-RL (PPO)
"""

from __future__ import annotations

import argparse
import os
import sys

# Isaac Sim / AppLauncher の初期化
from isaaclab.app import AppLauncher

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="Train Go2 Hopping Policy with RSL-RL")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments to simulate")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-Hopping-v0", help="Task name")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=None, help="Max RL iterations")

# AppLauncher の引数を追加
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# デフォルトで headless (GUI非表示) で高速に学習
if not args_cli.headless:
    args_cli.headless = True

# シミュレータの起動
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""シミュレータ起動後にモジュールをインポート"""
import gymnasium as gym
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata

# カスタム環境と設定の読み込み
import envs
from agents.rsl_rl_ppo_cfg import UnitreeGo2HoppingPPORunnerCfg
from envs.go2_hopping_env_cfg import Go2HoppingEnvCfg


def main():
    """メイン学習ループ"""
    # 1. 環境設定の初期化
    env_cfg = Go2HoppingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # 2. Gym 環境の作成
    env = gym.make(args_cli.task, cfg=env_cfg)
    # RSL-RL 用のラッパーでラップ
    env = RslRlVecEnvWrapper(env)

    # 3. PPO Runner 設定
    agent_cfg = UnitreeGo2HoppingPPORunnerCfg()
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    # RSL-RL バージョン互換性処理
    try:
        rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    except Exception:
        rsl_rl_version = importlib.metadata.version("rsl_rl")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    # ログ出力先ディレクトリ
    log_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs", "rsl_rl"))
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Training directory: {log_dir}")
    print(f"[INFO] Simulating {args_cli.num_envs} Go2 robots concurrently on GPU...")

    # 4. RSL-RL Runner の初期化と学習実行
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 5. 学習スタート
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 6. 環境クローズ
    env.close()
    print("[INFO] Training finished successfully!")


if __name__ == "__main__":
    main()
    simulation_app.close()

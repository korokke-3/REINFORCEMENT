"""
Unitree Go2 片足けんけん 学習結果可視化・シミュレーション再生スクリプト
Isaac Sim GUIでロボットの動作を確認します
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="Play/Visualize Go2 Hopping Policy")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-Hopping-Play-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robots to display in GUI")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")

# AppLauncher引数の追加
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# デフォルトで GUI (Kit ビューアー) を有効化
if not args_cli.visualizer and not args_cli.headless:
    args_cli.visualizer = ["kit"]

# 可視化のためGUIを有効化
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""シミュレータ起動後にモジュールをインポート"""
import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata

import envs
from agents.rsl_rl_ppo_cfg import UnitreeGo2HoppingPPORunnerCfg
from envs.go2_hopping_env_cfg import Go2HoppingEnvCfg_PLAY


def get_latest_checkpoint(log_dir: str) -> str:
    """最新のモデル重みファイルを取得"""
    import glob
    checkpoints = glob.glob(os.path.join(log_dir, "**", "model_*.pt"), recursive=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint model_*.pt found in {log_dir}")
    # 更新日時が最新のものを選択
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    return latest_ckpt


def main():
    """可視化ループ"""
    env_cfg = Go2HoppingEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = UnitreeGo2HoppingPPORunnerCfg()
    try:
        rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    except Exception:
        rsl_rl_version = importlib.metadata.version("rsl_rl")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    # チェックポイントパスの解決
    ckpt_path = args_cli.checkpoint
    if ckpt_path is None:
        log_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs", "rsl_rl"))
        ckpt_path = get_latest_checkpoint(log_root)
    
    print(f"[INFO] Loading trained checkpoint: {ckpt_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 評価実行ループ
    obs = env.get_observations()
    print("[INFO] Simulation running. Close window to exit.")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

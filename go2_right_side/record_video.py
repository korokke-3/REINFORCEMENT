from __future__ import annotations
import argparse
import glob
import os
import shutil
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record Go2 Right-Side Video")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-RightSide-Play-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robots")
parser.add_argument("--video_length", type=int, default=300, help="Number of steps")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata

import envs
from agents.rsl_rl_ppo_cfg import UnitreeGo2RightSidePPORunnerCfg
from envs.go2_right_side_env_cfg import Go2RightSideEnvCfg_PLAY


def get_latest_checkpoint(log_dir: str) -> str:
    checkpoints = glob.glob(os.path.join(log_dir, "**", "model_*.pt"), recursive=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint model_*.pt found in {log_dir}")
    return max(checkpoints, key=os.path.getmtime)


def main():
    env_cfg = Go2RightSideEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs

    video_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs", "videos"))
    os.makedirs(video_folder, exist_ok=True)
    for f in glob.glob(os.path.join(video_folder, "*.mp4")):
        try: os.remove(f)
        except: pass

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    video_kwargs = {
        "video_folder": video_folder,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = UnitreeGo2RightSidePPORunnerCfg()
    try:
        rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    except Exception:
        rsl_rl_version = importlib.metadata.version("rsl_rl")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    ckpt_path = args_cli.checkpoint
    if ckpt_path is None:
        log_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs", "rsl_rl"))
        ckpt_path = get_latest_checkpoint(log_root)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    print(f"[INFO] Recording {args_cli.video_length} steps to MP4...")

    for step in range(args_cli.video_length):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

    env.close()

    videos = glob.glob(os.path.join(video_folder, "*.mp4"))
    if videos:
        latest_video = max(videos, key=os.path.getmtime)
        final_mp4 = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs", "go2_right_side.mp4"))
        shutil.copyfile(latest_video, final_mp4)
        print(f"[SUCCESS] Video saved: {final_mp4}")
    else:
        print("[ERROR] No MP4 file found in output folder!")


if __name__ == "__main__":
    main()
    simulation_app.close()

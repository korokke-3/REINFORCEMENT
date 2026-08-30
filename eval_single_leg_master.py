"""
Unitree Go2 1本足連続跳躍 (Single-Leg Explosive Continuous Jump) マスター評価スクリプト
学習済みベストモデル (7400 iterations) を読み込み、5秒間の連続跳躍動作と物理メトリクスを評価・録画します。
"""

from __future__ import annotations

import argparse
import os
import sys

# パス設定
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "go2_single_leg"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate 7400-Iteration Single-Leg Continuous Jump Policy")
parser.add_argument("--checkpoint", type=str, default=os.path.join(CURRENT_DIR, "models", "single_leg_jump_7400iter.pt"), help="Path to model checkpoint (.pt)")
parser.add_argument("--output_dir", type=str, default=os.path.join(CURRENT_DIR, "assets", "eval_output"), help="Output directory for recorded video")
parser.add_argument("--duration_steps", type=int, default=250, help="Number of steps (250 steps = 5.0 seconds)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless", "--enable_cameras"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import imageio.v3 as iio
from PIL import Image

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata

import envs
from agents.rsl_rl_ppo_cfg import UnitreeGo2SingleLegPPORunnerCfg
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY


def main():
    env_cfg = Go2SingleLegEnvCfg_PLAY()
    env_cfg.viewer.eye = (0.0, 2.0, 0.45)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.28)

    os.makedirs(args_cli.output_dir, exist_ok=True)

    raw_env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0", cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=args_cli.output_dir,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.duration_steps,
        name_prefix="eval_single_leg_jump",
    )

    env_wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = UnitreeGo2SingleLegPPORunnerCfg()
    try:
        rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    except Exception:
        rsl_rl_version = importlib.metadata.version("rsl_rl")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env_wrapped.reset()
    robot = raw_env.unwrapped.scene["robot"]

    print("\n" + "=" * 80)
    print("=== EVALUATION OF 7,400-ITERATION SINGLE-LEG CONTINUOUS JUMP POLICY ===")
    print("=" * 80)

    survived_steps = 0
    resets = 0
    max_survival = 0
    total_air_steps = 0
    max_foot_height = 0.0
    max_body_vz = 0.0

    for step in range(args_cli.duration_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, infos = env_wrapped.step(actions)

        body_pos = robot.data.body_pos_w[0].cpu().numpy()
        rr_foot = body_pos[robot.data.body_names.index("RR_foot")]
        fl_foot = body_pos[robot.data.body_names.index("FL_foot")]
        fr_foot = body_pos[robot.data.body_names.index("FR_foot")]
        rl_foot = body_pos[robot.data.body_names.index("RL_foot")]
        root_pos = robot.data.root_pos_w[0].cpu().numpy()
        root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()

        other_min_z = min(fl_foot[2], fr_foot[2], rl_foot[2])
        is_airborne = rr_foot[2] > 0.035
        if is_airborne:
            total_air_steps += 1
            if rr_foot[2] > max_foot_height:
                max_foot_height = rr_foot[2]

        if root_vel[2] > max_body_vz:
            max_body_vz = root_vel[2]

        is_done = dones[0].item()
        status = "★ AIRBORNE JUMP ★" if is_airborne else "Push & Touchdown"
        if is_done:
            status = "RESET"

        print(
            f"Step {step:03d} (t={step*0.02:.2f}s) | "
            f"Body_Z={root_pos[2]*100:5.1f}cm | "
            f"Vz={root_vel[2]:+5.2f}m/s | "
            f"Foot_Z={rr_foot[2]*100:+5.1f}cm | "
            f"Other_Legs_Min_Z={other_min_z*100:+5.1f}cm | [{status}]"
        )

        if is_done:
            resets += 1
            if survived_steps > max_survival:
                max_survival = survived_steps
            print(f"  >>> Reset #{resets} (Episode Survived: {survived_steps} steps = {survived_steps*0.02:.2f}s) <<<")
            survived_steps = 0
        else:
            survived_steps += 1

    if survived_steps > max_survival:
        max_survival = survived_steps

    print("\n" + "=" * 80)
    print("=== 7,400-ITERATION CONTINUOUS JUMP FINAL RESULTS ===")
    print(f"Total Evaluated Time   : {args_cli.duration_steps*0.02:.2f} s ({args_cli.duration_steps} steps)")
    print(f"Max Continuous Survival: {max_survival*0.02:.2f} s ({max_survival} steps)")
    print(f"Max Foot Clearance     : {max_foot_height*100:.1f} cm")
    print(f"Max Vertical Launch Vz : {max_body_vz:+.2f} m/s")
    print(f"Total Airborne Ratio   : {total_air_steps/args_cli.duration_steps*100:.1f}% ({total_air_steps*0.02:.2f} s in flight)")
    print("=" * 80)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

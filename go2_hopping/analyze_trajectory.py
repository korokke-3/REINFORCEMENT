import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

import envs
from envs.go2_hopping_env_cfg import Go2HoppingEnvCfg_PLAY
from agents.rsl_rl_ppo_cfg import UnitreeGo2HoppingPPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata

def main():
    env_cfg = Go2HoppingEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1

    env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-Hopping-Play-v0', cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = UnitreeGo2HoppingPPORunnerCfg()
    try:
        rsl_rl_version = importlib.metadata.version('rsl-rl-lib')
    except Exception:
        rsl_rl_version = importlib.metadata.version('rsl_rl')
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    ckpt = max(glob.glob('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_hopping/logs/rsl_rl/**/model_*.pt', recursive=True), key=os.path.getmtime)
    print(f"Loading: {ckpt}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()

    positions = []
    velocities = []
    commands = []
    joint_positions = []

    num_steps = 300
    for step in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            
            pos = env.unwrapped.scene['robot'].data.root_pos_w[0].cpu().numpy().copy()
            vel = env.unwrapped.scene['robot'].data.root_lin_vel_w[0].cpu().numpy().copy()
            cmd = env.unwrapped.command_manager.get_command('base_velocity')[0].cpu().numpy().copy()
            jp = env.unwrapped.scene['robot'].data.joint_pos[0].cpu().numpy().copy()
            
            positions.append(pos)
            velocities.append(vel)
            commands.append(cmd)
            joint_positions.append(jp)

    env.close()

    positions = np.array(positions)
    velocities = np.array(velocities)
    commands = np.array(commands)
    joint_positions = np.array(joint_positions)

    print('=== TRAJECTORY ANALYSIS (300 steps, dt=0.02s => 6.0 seconds) ===')
    print(f'Start Position (X, Y, Z): ({positions[0,0]:.4f}, {positions[0,1]:.4f}, {positions[0,2]:.4f})')
    print(f'End Position   (X, Y, Z): ({positions[-1,0]:.4f}, {positions[-1,1]:.4f}, {positions[-1,2]:.4f})')
    
    delta_x = positions[-1, 0] - positions[0, 0]
    delta_y = positions[-1, 1] - positions[0, 1]
    total_displacement = np.linalg.norm(positions[-1, :2] - positions[0, :2])
    print(f'Delta X (Forward)       : {delta_x:.4f} m')
    print(f'Delta Y (Lateral)       : {delta_y:.4f} m')
    print(f'Total XY Displacement   : {total_displacement:.4f} m')
    print(f'Mean Actual Velocity    : Vx={np.mean(velocities[:,0]):.4f} m/s, Vy={np.mean(velocities[:,1]):.4f} m/s')
    print(f'Mean Command Velocity   : Cmd_Vx={np.mean(commands[:,0]):.4f} m/s, Cmd_Vy={np.mean(commands[:,1]):.4f} m/s')
    print(f'Base Height (Z)         : Min={np.min(positions[:,2]):.4f}m, Max={np.max(positions[:,2]):.4f}m, Mean={np.mean(positions[:,2]):.4f}m')

    # グラフの作成
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    time_axis = np.arange(num_steps) * 0.02

    # 1. 位置の推移 (X, Y, Z)
    axes[0, 0].plot(time_axis, positions[:, 0], label='X (Forward / 前進)', color='blue', lw=2)
    axes[0, 0].plot(time_axis, positions[:, 1], label='Y (Lateral / 横)', color='orange', lw=2)
    axes[0, 0].plot(time_axis, positions[:, 2], label='Z (Height / 高さ)', color='green', lw=2)
    axes[0, 0].set_title('Robot Position (X, Y, Z) vs Time (sec)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.5)

    # 2. 平面軌跡 (X vs Y)
    axes[0, 1].plot(positions[:, 0], positions[:, 1], 'b-', lw=2, label='Trajectory')
    axes[0, 1].scatter([positions[0, 0]], [positions[0, 1]], color='green', s=80, zorder=5, label='Start')
    axes[0, 1].scatter([positions[-1, 0]], [positions[-1, 1]], color='red', s=80, zorder=5, label='End')
    axes[0, 1].set_title('2D Trajectory on Ground (XY Plane)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('X (Forward) [m]')
    axes[0, 1].set_ylabel('Y (Lateral) [m]')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.5)

    # 3. 速度と目標速度の比較
    axes[1, 0].plot(time_axis, velocities[:, 0], label='Actual Vx', color='blue', lw=1.5)
    axes[1, 0].plot(time_axis, commands[:, 0], 'r--', label='Command Vx', lw=2)
    axes[1, 0].set_title('Forward Velocity (Vx) vs Command (Cmd_Vx)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.5)

    # 4. 関節角度の動き (右前脚 vs 左前脚)
    axes[1, 1].plot(time_axis, joint_positions[:, 4], label='FL_thigh (Left Front)', color='teal', lw=1.5)
    axes[1, 1].plot(time_axis, joint_positions[:, 5], label='FR_thigh (Right Front - Raised)', color='purple', lw=1.5)
    axes[1, 1].set_title('Joint Position (Thigh Joint Comparison)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Joint Angle (rad)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.5)

    plt.tight_layout()
    plot_path1 = '/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_hopping/logs/trajectory_analysis.png'
    plot_path2 = '/home/exhibition-spakona/.gemini/antigravity-cli/brain/0d9f1357-37bf-4b0e-acf1-ef7ab31a2def/trajectory_analysis.png'
    plt.savefig(plot_path1, dpi=150)
    plt.savefig(plot_path2, dpi=150)
    print(f"Saved trajectory plots to {plot_path1}")

if __name__ == '__main__':
    main()
    simulation_app.close()

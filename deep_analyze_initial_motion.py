from __future__ import annotations
import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
import gymnasium as gym
import torch
import math
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
# 最適静止姿勢
opt_roll = math.radians(20.0)
opt_pitch = math.radians(-12.0)
cr = math.cos(opt_roll * 0.5); sr = math.sin(opt_roll * 0.5)
cp = math.cos(opt_pitch * 0.5); sp = math.sin(opt_pitch * 0.5)
qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.420)
env_cfg.scene.robot.init_state.rot = (qx, qy, qz, qw)
env_cfg.scene.robot.init_state.joint_pos = {
    "FL_hip_joint": 0.35,
    "FR_hip_joint": -0.35,
    "RL_hip_joint": 0.35,
    "RR_hip_joint": 0.05,
    "FL_thigh_joint": -1.40,
    "FR_thigh_joint": -1.40,
    "RL_thigh_joint": 1.80,
    "RR_thigh_joint": 0.00,
    "FL_calf_joint": -0.90,
    "FR_calf_joint": -0.90,
    "RL_calf_joint": -2.60,
    "RR_calf_joint": -0.850,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)
obs, _ = env.reset()
robot = env.unwrapped.scene['robot']
contact_sensor = env.unwrapped.scene.sensors['contact_forces']

print("\n" + "="*110)
print("=== MILLIMETER-PRECISION STEP-BY-STEP KINEMATICS & CONTACT ANALYSIS ===")
print("="*110)

actions = torch.tensor([[0.05, 0.00, -0.850]], device='cuda:0')

for step in range(25):
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    body_vel = robot.data.body_lin_vel_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    root_p = robot.data.root_pos_w[0].cpu().numpy()
    root_v = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    forces = contact_sensor.data.net_forces_w[0].cpu().numpy()
    body_names = robot.data.body_names
    contact_str = []
    for idx, name in enumerate(body_names):
        f = np.linalg.norm(forces[idx])
        if f > 0.1:
            contact_str.append(f"{name}:{f:.1f}N")
            
    proj_g = robot.data.projected_gravity_b[0].cpu().numpy()
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    rr_thigh_deg = math.degrees(joint_pos[robot.data.joint_names.index('RR_thigh_joint')])
    rr_calf_deg = math.degrees(joint_pos[robot.data.joint_names.index('RR_calf_joint')])
    
    print(f"Step {step:02d} (t={step*0.02:.2f}s) | Base_Z={root_p[2]*100:5.2f}cm (Vz={root_v[2]:+5.2f}m/s) | Foot_Z={rr_foot[2]*100:+5.2f}cm | Knee_Z={rr_calf[2]*100:+5.2f}cm | RL_Z={rl_foot[2]*100:+5.2f}cm | RR_thigh={rr_thigh_deg:+5.1f}°, RR_calf={rr_calf_deg:+5.1f}° | Contacts: {contact_str}")
    
    obs, rewards, dones, truncated, infos = env.step(actions)
    if dones[0].item():
        print(f"  >>> [EPISODE TERMINATED & RESET at Step {step} (t={step*0.02:.2f}s)] <<<")

env.close()
simulation_app.close()

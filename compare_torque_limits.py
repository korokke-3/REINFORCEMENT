import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

import envs
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

def run_jump_test(test_name, torque_limit, stiffness_val):
    print(f"\n{'='*60}")
    print(f"=== RUNNING {test_name}: EffortLimit={torque_limit}Nm, Stiffness={stiffness_val} ===")
    print(f"{'='*60}")
    
    env_cfg = Go2SingleLegEnvCfg_PLAY()
    env_cfg.terminations.illegal_parts_contact = None
    env_cfg.terminations.forward_fall = None
    env_cfg.terminations.root_height_below_minimum = None
    env_cfg.terminations.base_contact = None
    env_cfg.actions.joint_pos.scale = 1.0

    # 最適姿勢
    r_init = R.from_euler('xyz', [28.5, -30.5, 0], degrees=True)
    quat_init = tuple(r_init.as_quat().tolist())

    fixed_joints = {
        'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
        'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,
        'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,
    }

    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.215)
    env_cfg.scene.robot.init_state.rot = quat_init
    env_cfg.scene.robot.init_state.joint_pos = {
        **fixed_joints,
        'RR_thigh_joint': 0.75,
        'RR_calf_joint': -2.15,
    }
    
    # アクチュエータ特性を変更
    env_cfg.scene.robot.actuators["base_legs"].effort_limit = torque_limit
    env_cfg.scene.robot.actuators["base_legs"].stiffness = stiffness_val
    env_cfg.scene.robot.actuators["base_legs"].saturation_effort = torque_limit

    env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg)
    obs, _ = env.reset()
    robot = env.unwrapped.scene['robot']
    joint_names = robot.data.joint_names

    max_foot_z = 0.0
    max_body_z = 0.0
    max_vz = -100.0
    air_frames = 0

    for step in range(50):
        t = step * 0.02
        actions = torch.zeros((1, 12), device=env.unwrapped.device)
        for jname, val in fixed_joints.items():
            j_idx = joint_names.index(jname)
            actions[0, j_idx] = val - robot.data.default_joint_pos[0, j_idx]

        # キックシーケンス
        if t < 0.08:
            target_thigh = 0.75
            target_calf = -2.15
        elif t < 0.16:
            # 爆発的キック
            target_thigh = -0.30
            target_calf = -0.60
        elif t < 0.28:
            # 空中タック
            target_thigh = 0.75
            target_calf = -2.30
        else:
            target_thigh = 0.60
            target_calf = -1.60

        rr_thigh_idx = joint_names.index('RR_thigh_joint')
        rr_calf_idx = joint_names.index('RR_calf_joint')
        actions[0, rr_thigh_idx] = target_thigh - robot.data.default_joint_pos[0, rr_thigh_idx]
        actions[0, rr_calf_idx] = target_calf - robot.data.default_joint_pos[0, rr_calf_idx]

        obs, reward, terminated, truncated, info = env.step(actions)
        
        body_pos = robot.data.body_pos_w[0].cpu().numpy()
        rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
        root_pos = robot.data.root_pos_w[0].cpu().numpy()
        root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()

        if rr_foot[2] > max_foot_z: max_foot_z = rr_foot[2]
        if root_pos[2] > max_body_z: max_body_z = root_pos[2]
        if root_vel[2] > max_vz: max_vz = root_vel[2]
        if rr_foot[2] > 0.035: air_frames += 1

    print(f"[{test_name} RESULTS]")
    print(f"  - Max Vertical Velocity (Vz) : {max_vz:+.3f} m/s")
    print(f"  - Max Foot Lift Height       : {max_foot_z*100:.1f} cm (Clearance: +{(max_foot_z-0.024)*100:.1f} cm)")
    print(f"  - Max Body Height            : {max_body_z*100:.1f} cm")
    print(f"  - Airborne Steps             : {air_frames} steps ({air_frames*0.02:.2f} s)")
    
    env.close()

# テスト1: Go2 実機定格 (23.5Nm, Kp=25)
run_jump_test("Go2 Real Spec", torque_limit=23.5, stiffness_val=25.0)

# テスト2: 高剛性・高トルク (45.0Nm, Kp=50)
run_jump_test("High-Power Spec (45Nm)", torque_limit=45.0, stiffness_val=50.0)

# テスト3: 超高出力 (80.0Nm, Kp=100)
run_jump_test("Super-Power Spec (80Nm)", torque_limit=80.0, stiffness_val=100.0)

simulation_app.close()

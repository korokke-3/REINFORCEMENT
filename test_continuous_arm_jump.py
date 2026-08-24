import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless', '--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import cv2
import imageio
import gymnasium as gym
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

import envs
from envs.go2_single_leg_env_cfg import Go2SingleLegEnvCfg_PLAY

env_cfg = Go2SingleLegEnvCfg_PLAY()
env_cfg.terminations.illegal_parts_contact = None
env_cfg.terminations.forward_fall = None
env_cfg.terminations.root_height_below_minimum = None
env_cfg.terminations.base_contact = None
env_cfg.actions.joint_pos.scale = 1.0

# カメラ設定: 連続ジャンプのダイナミックな打ち上がりが全身見える位置
env_cfg.viewer.eye = (0.0, 1.8, 0.45)
env_cfg.viewer.lookat = (0.0, 0.0, 0.28)

# 重心直下アライメント姿勢 (Roll=+18.0°, Pitch=-35.0°)
TARGET_ROLL_DEG = 18.0
TARGET_PITCH_DEG = -35.0
r_init = R.from_euler('xyz', [TARGET_ROLL_DEG, TARGET_PITCH_DEG, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

# 初期状態: 接地・タメ
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.22)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,   'RR_thigh_joint': 0.80,
    'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,  'RR_calf_joint': -2.20,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/continuous_arm_swing_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=150, # 3.0秒 (複数回の連続ジャンプ)
    name_prefix='continuous_arm_jump',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']
joint_names = robot.data.joint_names

fl_thigh_idx = joint_names.index('FL_thigh_joint')
fr_thigh_idx = joint_names.index('FR_thigh_joint')
rl_thigh_idx = joint_names.index('RL_thigh_joint')
rr_thigh_idx = joint_names.index('RR_thigh_joint')
rr_calf_idx = joint_names.index('RR_calf_joint')

print("\n" + "="*75)
print("=== EXPERIMENT: CONTINUOUS REPEATED INERTIAL-ASSISTED SINGLE-LEG JUMPS ===")
print("=== (CYCLIC ARM-SWING + EXPLOSIVE GROUND PUSH-OFF IN EVERY HOP) ===")
print("="*75)

# 連続ジャンプ周期パラメータ
# 1周期 = 0.36秒 (18 steps)
# - Phase A (t_in_cycle: 0.00 ~ 0.08s, 4 steps): SQUAT & ARM DOWN (接地タメ・腕を下げてタメ)
# - Phase B (t_in_cycle: 0.08 ~ 0.18s, 5 steps): ★ FULL THRUST & RAPID ARM UP (一斉キック＆急振り上げ) ★
# - Phase C (t_in_cycle: 0.18 ~ 0.30s, 6 steps): ★ BALLISTIC FLIGHT (体全体が上空へ打ち上がる) ★
# - Phase D (t_in_cycle: 0.30 ~ 0.36s, 3 steps): TOUCHDOWN PREPARATION (着地構え)

CYCLE_STEPS = 18
log_data = []
jump_count = 0

for step in range(150):
    t = step * 0.02
    cycle_step = step % CYCLE_STEPS
    
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 基本固定関節
    actions[0, joint_names.index('FL_hip_joint')] = 0.1 - robot.data.default_joint_pos[0, 0]
    actions[0, joint_names.index('FR_hip_joint')] = -0.1 - robot.data.default_joint_pos[0, 1]
    actions[0, joint_names.index('RL_hip_joint')] = 0.2 - robot.data.default_joint_pos[0, 2]
    actions[0, joint_names.index('RR_hip_joint')] = -0.1 - robot.data.default_joint_pos[0, 3]
    actions[0, joint_names.index('FL_calf_joint')] = -1.0 - robot.data.default_joint_pos[0, 8]
    actions[0, joint_names.index('FR_calf_joint')] = -1.0 - robot.data.default_joint_pos[0, 9]
    actions[0, joint_names.index('RL_calf_joint')] = -2.5 - robot.data.default_joint_pos[0, 10]

    if cycle_step < 4:
        # Phase A: 接地タメ ＆ 腕下げ（反動の準備）
        target_rr_thigh = 0.80
        target_rr_calf = -2.25
        target_arm_front = -1.2 # 前脚を下げる
        target_arm_rear = 1.6   # 左後脚を下げる
        phase_name = "SQUAT_PREP"
        if cycle_step == 0:
            jump_count += 1
            print(f"\n>>> [STARTING JUMP #{jump_count}] at t={t:.2f}s <<<")

    elif cycle_step < 9:
        # Phase B: ★ 爆発的キック ＋ 3脚一斉急激振り上げ（慣性反動発射！） ★
        target_rr_thigh = -0.40 # 支持脚急伸展
        target_rr_calf = -0.60  # 膝フルストロークキック
        target_arm_front = 1.8  # 前脚2本を一気に上方へ急スイング！
        target_arm_rear = -0.8  # 左後脚を上方へ急スイング！
        phase_name = "★ THRUST_LAUNCH ★"

    elif cycle_step < 15:
        # Phase C: ★ 弾道飛行（体全体が宙に打ち上がる） ★
        # 空中で脚を自然な高さに維持
        target_rr_thigh = 0.60
        target_rr_calf = -1.80
        target_arm_front = 0.5
        target_arm_rear = 0.8
        phase_name = "★ BALLISTIC APEX ★"

    else:
        # Phase D: 着地準備
        target_rr_thigh = 0.75
        target_rr_calf = -2.10
        target_arm_front = -0.8
        target_arm_rear = 1.3
        phase_name = "TOUCHDOWN"

    actions[0, rr_thigh_idx] = target_rr_thigh - robot.data.default_joint_pos[0, rr_thigh_idx]
    actions[0, rr_calf_idx] = target_rr_calf - robot.data.default_joint_pos[0, rr_calf_idx]
    actions[0, fl_thigh_idx] = target_arm_front - robot.data.default_joint_pos[0, fl_thigh_idx]
    actions[0, fr_thigh_idx] = target_arm_front - robot.data.default_joint_pos[0, fr_thigh_idx]
    actions[0, rl_thigh_idx] = target_arm_rear - robot.data.default_joint_pos[0, rl_thigh_idx]

    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    is_airborne = rr_foot[2] > 0.035
    status = "★ FULL BODY FLIGHT ★" if (is_airborne and root_pos[2] > 0.24) else ("Airborne" if is_airborne else "Grounded")
    
    log_data.append({
        'step': step, 't': t, 'foot_z': rr_foot[2], 'body_z': root_pos[2],
        'vz': root_vel[2], 'air': is_airborne
    })
    
    print(f"Step {step:03d} (t={t:.2f}s | {phase_name:<19}) | Foot_Z={rr_foot[2]*100:+5.1f}cm | Body_Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | [{status}]")

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/continuous_arm_jump-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/continuous_arm_jump.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

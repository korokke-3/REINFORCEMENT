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

# カメラ設定
env_cfg.viewer.eye = (0.0, 1.4, 0.35)
env_cfg.viewer.lookat = (0.0, 0.0, 0.22)

# 最適直立バランス角 (Roll=-35.0°, Pitch=-10.0°)
TARGET_ROLL_DEG = -35.0
TARGET_PITCH_DEG = -10.0
r_init = R.from_euler('xyz', [TARGET_ROLL_DEG, TARGET_PITCH_DEG, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

# 固定する他3脚の関節角
FIXED_JOINTS = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,
    'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,
}

# 初期設定
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.215)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    **FIXED_JOINTS,
    'RR_thigh_joint': 0.70,
    'RR_calf_joint': -2.00,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/statemachine_hop_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=150, # 3.0秒
    name_prefix='statemachine_hop',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*75)
print("=== EXPERIMENT 2: STATE-MACHINE DRIVEN CONTINUOUS SINGLE-LEG HOPPING ===")
print("=== (CONTACT & HEIGHT SENSING, SQUAT -> THRUST -> TUCK -> PREPARE) ===")
print("="*75)

joint_names = robot.data.joint_names
rr_hip_idx = joint_names.index('RR_hip_joint')
rr_thigh_idx = joint_names.index('RR_thigh_joint')
rr_calf_idx = joint_names.index('RR_calf_joint')

# ステートマシンの定義
# 0: SQUAT (接地タメ・反発エネルギー蓄積)
# 1: THRUST (最大出力キック急伸展)
# 2: TUCK (空中滞空・脚引き戻し)
# 3: EXTEND (着地前接地準備)
state = 0
state_timer = 0
hop_count = 0

log_data = []

Kp_roll = 0.5
Kd_roll = 0.05
Kp_pitch = 0.8
Kd_pitch = 0.08

for step in range(150):
    t = step * 0.02
    state_timer += 1
    
    # 計測
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    root_ang_vel = robot.data.root_ang_vel_w[0].cpu().numpy()
    rot = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
    euler_deg = rot.as_euler('xyz', degrees=True)
    roll_cur, pitch_cur, yaw_cur = euler_deg
    
    is_grounded = rr_foot[2] <= 0.030
    is_airborne = rr_foot[2] > 0.035
    
    # ステートマシン遷移ロジック
    if state == 0: # SQUAT
        target_thigh = 0.75
        target_calf = -2.15
        # 4ステップ以上タメて接地しているか、一定時間経過でキックへ
        if state_timer >= 4 and is_grounded:
            state = 1
            state_timer = 0
            hop_count += 1
            print(f">>> [HOP #{hop_count} TRIGGERED] at step {step:03d} (t={t:.2f}s) <<<")
            
    elif state == 1: # THRUST (キック)
        target_thigh = -0.30
        target_calf = -0.60
        # 4ステップキックしたら空中タックへ
        if state_timer >= 4:
            state = 2
            state_timer = 0
            
    elif state == 2: # TUCK (空中引き込み)
        target_thigh = 0.70
        target_calf = -2.20
        # 6ステップ滞空したら着地準備へ
        if state_timer >= 6:
            state = 3
            state_timer = 0
            
    elif state == 3: # EXTEND (接地準備)
        target_thigh = 0.65
        target_calf = -1.80
        # 接地した瞬間、または3ステップ経過でSQUAT（タメ）へ
        if is_grounded or state_timer >= 4:
            state = 0
            state_timer = 0

    # 姿勢フィードバック補正
    roll_err = np.radians(TARGET_ROLL_DEG - roll_cur)
    pitch_err = np.radians(TARGET_PITCH_DEG - pitch_cur)
    delta_hip = -(Kp_roll * roll_err - Kd_roll * root_ang_vel[0])
    delta_thigh = -(Kp_pitch * pitch_err - Kd_pitch * root_ang_vel[1])

    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    for jname, val in FIXED_JOINTS.items():
        j_idx = joint_names.index(jname)
        actions[0, j_idx] = val - robot.data.default_joint_pos[0, j_idx]

    actions[0, rr_hip_idx] = FIXED_JOINTS['RR_hip_joint'] + np.clip(delta_hip, -0.25, 0.25) - robot.data.default_joint_pos[0, rr_hip_idx]
    actions[0, rr_thigh_idx] = target_thigh + np.clip(delta_thigh, -0.35, 0.35) - robot.data.default_joint_pos[0, rr_thigh_idx]
    actions[0, rr_calf_idx] = target_calf - robot.data.default_joint_pos[0, rr_calf_idx]

    obs, reward, terminated, truncated, info = env.step(actions)
    
    state_names = ["SQUAT", "THRUST", "TUCK", "EXTEND"]
    status = "★ AIRBORNE ★" if is_airborne else "Grounded"
    
    log_data.append({
        'step': step, 't': t, 'state': state_names[state], 'rr_z': rr_foot[2],
        'body_z': root_pos[2], 'vz': root_vel[2], 'roll': roll_cur, 'pitch': pitch_cur, 'air': is_airborne
    })
    
    if step % 2 == 0 or is_airborne or state == 1:
        print(f"Step {step:03d} (t={t:.2f}s | {state_names[state]:<6}) | RR_Z={rr_foot[2]*100:+5.1f}cm | Body_Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | Roll={roll_cur:+5.1f}° Pitch={pitch_cur:+5.1f}° | {status}")

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/statemachine_hop-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/statemachine_hop.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

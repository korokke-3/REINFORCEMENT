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
# 終了条件を無効化
env_cfg.terminations.illegal_parts_contact = None
env_cfg.terminations.forward_fall = None
env_cfg.terminations.root_height_below_minimum = None
env_cfg.terminations.base_contact = None

# アクションスケールを1.0に設定
env_cfg.actions.joint_pos.scale = 1.0

# カメラ設定: ロボットの姿勢全体と足元が美しく見える位置
env_cfg.viewer.eye = (0.0, 1.5, 0.35)
env_cfg.viewer.lookat = (0.0, 0.0, 0.22)

# 一定姿勢の定義:
# 胴体姿勢: Roll=+16.0度, Pitch=-38.0度（合成重心がRR足先の鉛直真上）
# 他3脚: 綺麗に空中に折りたたんで完全固定
r_init = R.from_euler('xyz', [16.0, -38.0, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

# 他3脚の固定関節角度
FIXED_JOINTS = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,
    'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,
}

# 初期状態（支持脚は屈曲タメ状態）
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.23)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    **FIXED_JOINTS,
    'RR_thigh_joint': 0.75,
    'RR_calf_joint': -2.10,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/same_posture_jump_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=80,
    name_prefix='same_posture_jump',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']

print("\n" + "="*65)
print("=== EXPERIMENT: SINGLE-LEG JUMP MAINTAINING FIXED POSTURE ===")
print("=== (NO ARM-SWING, RIGID 3-LEG TUCK, PURE LEG PISTON THRUST) ===")
print("="*65)

log_data = []

# 支持脚（RR）のIK直列キック軌道
# 膝（Calf）を伸ばすのと同時に大腿（Thigh）の角度を調整し、足先を「重心直下鉛直線」上で伸縮させる
for step in range(80):
    t = step * 0.02
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 1. 他3脚は常に完全に同じ角度に固定（一切動かさない）
    actions[0, 0] = FIXED_JOINTS['FL_hip_joint'] - robot.data.default_joint_pos[0, 0]
    actions[0, 1] = FIXED_JOINTS['FR_hip_joint'] - robot.data.default_joint_pos[0, 1]
    actions[0, 2] = FIXED_JOINTS['RL_hip_joint'] - robot.data.default_joint_pos[0, 2]
    actions[0, 3] = FIXED_JOINTS['RR_hip_joint'] - robot.data.default_joint_pos[0, 3]
    actions[0, 4] = FIXED_JOINTS['FL_thigh_joint'] - robot.data.default_joint_pos[0, 4]
    actions[0, 5] = FIXED_JOINTS['FR_thigh_joint'] - robot.data.default_joint_pos[0, 5]
    actions[0, 6] = FIXED_JOINTS['RL_thigh_joint'] - robot.data.default_joint_pos[0, 6]
    actions[0, 8] = FIXED_JOINTS['FL_calf_joint'] - robot.data.default_joint_pos[0, 8]
    actions[0, 9] = FIXED_JOINTS['FR_calf_joint'] - robot.data.default_joint_pos[0, 9]
    actions[0, 10] = FIXED_JOINTS['RL_calf_joint'] - robot.data.default_joint_pos[0, 10]

    # 2. 支持脚（RR）の純粋ピストン伸展（同じ姿勢を保ったままジャンプ）
    # Phase 1 (0.00s ~ 0.08s): タメ（接地・屈曲保持）
    if t < 0.08:
        target_rr_thigh = 0.75
        target_rr_calf = -2.10
    # Phase 2 (0.08s ~ 0.16s): ★ 鉛直ストレートキック（急伸展） ★
    elif t < 0.16:
        target_rr_thigh = -0.20 # 膝の伸展に合わせて股関節を前送り（足先が真下に動くようにIK連動）
        target_rr_calf = -0.70  # 膝を最大伸展
    # Phase 3 (0.16s ~ 0.28s): ★ 空中滞空・足引き戻し（離地クリアランス確保） ★
    elif t < 0.28:
        target_rr_thigh = 0.75
        target_rr_calf = -2.10 # 初期タメ姿勢に瞬時に戻す
    # Phase 4 (0.28s 以降): 着地・初期姿勢保持
    else:
        target_rr_thigh = 0.60
        target_rr_calf = -1.60

    actions[0, 7] = target_rr_thigh - robot.data.default_joint_pos[0, 7]
    actions[0, 11] = target_rr_calf - robot.data.default_joint_pos[0, 11]

    obs, reward, terminated, truncated, info = env.step(actions)
    
    # 計測
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy() # (w, x, y, z)
    # オイラー角 (Roll, Pitch, Yaw)
    rot = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
    euler_deg = rot.as_euler('xyz', degrees=True)
    
    is_airborne = rr_foot[2] > 0.035
    
    log_data.append({
        'step': step, 't': t, 'rr_z': rr_foot[2], 'body_z': root_pos[2],
        'vz': root_vel[2], 'roll': euler_deg[0], 'pitch': euler_deg[1], 'air': is_airborne
    })
    
    status = "★ AIRBORNE ★" if is_airborne else "Grounded"
    print(f"Step {step:02d} (t={t:.2f}s) | RR_Z={rr_foot[2]*100:+5.1f}cm | Body_Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | Roll={euler_deg[0]:+5.1f}° Pitch={euler_deg[1]:+5.1f}° | {status}")

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/same_posture_jump-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/same_posture_jump.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

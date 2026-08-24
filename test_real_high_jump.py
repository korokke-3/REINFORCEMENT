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

# カメラ設定: ロボットが立ち姿勢から高く空中にジャンプする全体を捉える
env_cfg.viewer.eye = (0.0, 2.0, 0.45)
env_cfg.viewer.lookat = (0.0, 0.0, 0.35)

# 直立・一本足立ち構え (Roll=+10.0°, Pitch=-15.0°)
# 倒れ込まず、堂々と立った姿勢
r_init = R.from_euler('xyz', [10.0, -15.0, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

# 他3脚: 美しく空中に引き上げたフラミンゴ姿勢
FIXED_JOINTS = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -0.8,  'FR_thigh_joint': -0.8,  'RL_thigh_joint': 1.2,
    'FL_calf_joint': -1.2,   'FR_calf_joint': -1.2,   'RL_calf_joint': -2.0,
}

# 初期状態: 直立高さ (Body Z = 0.33m) でしっかり接地
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.33)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    **FIXED_JOINTS,
    'RR_thigh_joint': 0.60,
    'RR_calf_joint': -1.60, # 軽く曲げて構えた直立状態
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/real_high_jump_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=80,
    name_prefix='real_high_jump',
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
print("=== EXPERIMENT: TRUE HIGH-ALTITUDE SINGLE-LEG JUMP ===")
print("=== (STAND TALL -> DEEP SQUAT -> EXPLOSIVE HIGH LAUNCH -> APEX) ===")
print("="*75)

log_data = []

# タイムライン設計:
# 0.00s ~ 0.10s (5 steps): 直立姿勢で静止安定化 (Stand Tall, Body Z ≈ 33cm)
# 0.10s ~ 0.20s (5 steps): ★ 深いしゃがみ込み（タメ） (Deep Squat, Body Z: 33cm -> 23cm)
#                          前脚を下げて振り上げ準備
# 0.20s ~ 0.30s (5 steps): ★ 爆発的キック ＋ 前脚・他脚の強烈な上方突き上げ ★
#                          支持脚を最大伸展 (23cm -> 35cm -> 45cm+ へ発射)
# 0.30s ~ 0.50s (10 steps): ★ 弾道空中飛行（Apex: 胴体が45cm以上の上空へ打ち上がる！） ★
# 0.50s 以降: 着地

for step in range(80):
    t = step * 0.02
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 基本関節
    for jname, val in FIXED_JOINTS.items():
        j_idx = joint_names.index(jname)
        actions[0, j_idx] = val - robot.data.default_joint_pos[0, j_idx]

    if t < 0.10:
        # 1. 直立静止 (Stand)
        target_rr_thigh = 0.60
        target_rr_calf = -1.60
        target_fl_thigh = -0.8
        target_fr_thigh = -0.8
        target_rl_thigh = 1.2
        phase_str = "STAND_TALL"
    elif t < 0.20:
        # 2. 深いタメ（Squat）
        target_rr_thigh = 0.95
        target_rr_calf = -2.45
        target_fl_thigh = -1.5 # 腕を下げてタメる
        target_fr_thigh = -1.5
        target_rl_thigh = 1.8
        phase_str = "DEEP_SQUAT"
    elif t < 0.30:
        # 3. ★ 爆発的大跳躍キック ＆ 全開振り上げ ★
        target_rr_thigh = -0.50 # 最大伸展
        target_rr_calf = -0.50  # 膝を最大急伸展（地面を突き抜く）
        target_fl_thigh = 1.8   # 前脚を一気に天空へフルスイング！
        target_fr_thigh = 1.8
        target_rl_thigh = -0.8
        phase_str = "★ EXPLOSIVE_LAUNCH ★"
    elif t < 0.50:
        # 4. ★ 空中頂点滞空 (Apex Flight) ★
        target_rr_thigh = 0.50
        target_rr_calf = -1.50 # 空中で足を自然に戻す
        target_fl_thigh = 0.0
        target_fr_thigh = 0.0
        target_rl_thigh = 0.8
        phase_str = "★ HIGH APEX FLIGHT ★"
    else:
        # 5. 着地
        target_rr_thigh = 0.70
        target_rr_calf = -1.80
        target_fl_thigh = -0.8
        target_fr_thigh = -0.8
        target_rl_thigh = 1.2
        phase_str = "LANDING"

    actions[0, rr_thigh_idx] = target_rr_thigh - robot.data.default_joint_pos[0, rr_thigh_idx]
    actions[0, rr_calf_idx] = target_rr_calf - robot.data.default_joint_pos[0, rr_calf_idx]
    actions[0, fl_thigh_idx] = target_fl_thigh - robot.data.default_joint_pos[0, fl_thigh_idx]
    actions[0, fr_thigh_idx] = target_fr_thigh - robot.data.default_joint_pos[0, fr_thigh_idx]
    actions[0, rl_thigh_idx] = target_rl_thigh - robot.data.default_joint_pos[0, rl_thigh_idx]

    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    is_airborne = rr_foot[2] > 0.035
    status = "★ HIGH JUMP AIRBORNE ★" if (is_airborne and root_pos[2] > 0.35) else ("Airborne" if is_airborne else "Grounded")
    
    log_data.append({
        'step': step, 't': t, 'foot_z': rr_foot[2], 'body_z': root_pos[2],
        'vz': root_vel[2], 'air': is_airborne
    })
    
    print(f"Step {step:02d} (t={t:.2f}s | {phase_str:<22}) | Foot_Z={rr_foot[2]*100:+5.1f}cm | Body_Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | [{status}]")

env.close()
simulation_app.close()

# 分析
stand_body_z = log_data[4]['body_z'] # 直立時高度
squat_body_z = min(d['body_z'] for d in log_data[:15]) # タメ時高度
max_body_z = max(d['body_z'] for d in log_data[15:]) # 最高跳躍高度
max_foot_z = max(d['foot_z'] for d in log_data[15:])
max_vz = max(d['vz'] for d in log_data[10:25])

print("\n" + "="*75)
print("=== TRUE HIGH JUMP MEASUREMENT RESULTS ===")
print(f"1. Standing Body Height (立位高度)    : {stand_body_z*100:.1f} cm")
print(f"2. Deep Squat Height (タメ沈み込み)   : {squat_body_z*100:.1f} cm")
print(f"3. Maximum Launch Velocity (跳躍初速) : {max_vz:+.3f} m/s")
print(f"4. PEAK BODY HEIGHT (最高到達高度)    : {max_body_z*100:.1f} cm (Jump Height over Standing: +{(max_body_z - stand_body_z)*100:.1f} cm!)")
print(f"5. PEAK FOOT HEIGHT (足先最高地上高)  : {max_foot_z*100:.1f} cm (Ground Clearance: +{(max_foot_z - 0.024)*100:.1f} cm!)")
print("="*75)

# GIF生成
video_path = f"{output_dir}/real_high_jump-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/real_high_jump.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

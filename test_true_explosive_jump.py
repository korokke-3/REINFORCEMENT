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

# カメラ設定: ロボットの真横から跳躍全体をクリアに撮影
env_cfg.viewer.eye = (0.0, 1.6, 0.35)
env_cfg.viewer.lookat = (0.0, 0.0, 0.25)

# 重心直下アライメント姿勢 (Roll=+18.0°, Pitch=-25.0°)
r_init = R.from_euler('xyz', [18.0, -25.0, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

# 他3脚: 上方に折りたたみ
FIXED_JOINTS = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,
    'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,
}

# 初期状態: RR足先が地面(Z=0.023m)にピタリと着地したタメ状態 (Body Z = 0.22m)
# 落下衝撃ゼロの完全接地状態からスタート
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.22)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    **FIXED_JOINTS,
    'RR_thigh_joint': 0.70,
    'RR_calf_joint': -2.00, # タメ状態
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/true_explosive_jump_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=60,
    name_prefix='true_explosive_jump',
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
print("=== EXPERIMENT: DEFINITIVE EXPLOSIVE SINGLE-LEG JUMP TEST ===")
print("=== (GROUND-LOCKED SQUAT -> MAXIMUM IMPULSE LAUNCH -> TRUE FLIGHT) ===")
print("="*75)

log_data = []

# タイムライン:
# 0.00s ~ 0.06s (3 steps): 地面をしっかり踏みしめて安定 (Settle)
# 0.06s ~ 0.16s (5 steps): ★ 全力地面押し切りキック ＋ 前脚上方急加速スイング ★
# 0.16s ~ 0.40s (12 steps): ★ 弾道飛行（体全体が宙に浮き、最高点へ打ち上がる） ★

for step in range(60):
    t = step * 0.02
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 固定関節
    for jname, val in FIXED_JOINTS.items():
        j_idx = joint_names.index(jname)
        actions[0, j_idx] = val - robot.data.default_joint_pos[0, j_idx]

    if t < 0.06:
        # 地面をしっかり押さえてタメ
        target_rr_thigh = 0.70
        target_rr_calf = -2.00
        target_fl = -1.2
        target_fr = -1.2
        target_rl = 1.5
        phase = "GROUND_SETTLE"
    elif t < 0.16:
        # ★ 爆発的発射（支持脚を最大伸展 ＆ 腕を一気に天空へ突き上げる） ★
        target_rr_thigh = -0.50 # 股関節を急伸展
        target_rr_calf = -0.50  # 膝を地面に突き刺すように最大急伸展
        target_fl = 2.0         # 前脚を天空へフルスイング！
        target_fr = 2.0
        target_rl = -1.0
        phase = "★ EXPLOSIVE_LAUNCH ★"
    else:
        # 空中滞空
        target_rr_thigh = 0.50
        target_rr_calf = -1.50
        target_fl = 0.0
        target_fr = 0.0
        target_rl = 0.8
        phase = "★ BALLISTIC_FLIGHT ★"

    actions[0, rr_thigh_idx] = target_rr_thigh - robot.data.default_joint_pos[0, rr_thigh_idx]
    actions[0, rr_calf_idx] = target_rr_calf - robot.data.default_joint_pos[0, rr_calf_idx]
    actions[0, fl_thigh_idx] = target_fl - robot.data.default_joint_pos[0, fl_thigh_idx]
    actions[0, fr_thigh_idx] = target_fr - robot.data.default_joint_pos[0, fr_thigh_idx]
    actions[0, rl_thigh_idx] = target_rl - robot.data.default_joint_pos[0, rl_thigh_idx]

    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    # 4脚すべての足先高さを取得
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    all_feet_min_z = min(rr_foot[2], fl_foot[2], fr_foot[2], rl_foot[2])
    
    # ロボット全体が完全に地面から離れているか
    is_true_flight = (all_feet_min_z > 0.035) and (root_pos[2] > 0.25)
    status = "★★★ TRUE JUMP (ALL IN AIR) ★★★" if is_true_flight else ("Airborne" if rr_foot[2] > 0.035 else "Grounded")
    
    log_data.append({
        'step': step, 't': t, 'rr_foot_z': rr_foot[2], 'body_z': root_pos[2],
        'vz': root_vel[2], 'min_foot_z': all_feet_min_z, 'flight': is_true_flight
    })
    
    print(f"Step {step:02d} (t={t:.2f}s | {phase:<20}) | Foot_Z={rr_foot[2]*100:+5.1f}cm | Body_Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | [{status}]")

env.close()
simulation_app.close()

# 集計
initial_z = log_data[0]['body_z']
max_body_z = max(d['body_z'] for d in log_data)
max_foot_z = max(d['rr_foot_z'] for d in log_data)
max_vz = max(d['vz'] for d in log_data)
flight_steps = [d for d in log_data if d['flight']]

print("\n" + "="*75)
print("=== DEFINITIVE MEASUREMENT RESULTS ===")
print(f"1. Initial Grounded Body Height  : {initial_z*100:.1f} cm")
print(f"2. PEAK BODY LAUNCH HEIGHT       : {max_body_z*100:.1f} cm (Actual Lift: +{(max_body_z - initial_z)*100:.1f} cm!)")
print(f"3. Maximum Vertical Velocity     : {max_vz:+.3f} m/s")
print(f"4. Maximum RR Foot Height        : {max_foot_z*100:.1f} cm (Clearance: +{(max_foot_z - 0.024)*100:.1f} cm!)")
print(f"5. Total TRUE FLIGHT Frames      : {len(flight_steps)} steps ({len(flight_steps)*0.02:.2f} s)")
print("="*75)

# GIF生成
video_path = f"{output_dir}/true_explosive_jump-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/true_explosive_jump.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

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

# カメラ設定: ロボット全体が飛び上がる様子がはっきりわかる画角
env_cfg.viewer.eye = (0.0, 1.8, 0.4)
env_cfg.viewer.lookat = (0.0, 0.0, 0.25)

# 重心直下アライメント姿勢 (Roll=+28.5°, Pitch=-30.5°)
TARGET_ROLL_DEG = 28.5
TARGET_PITCH_DEG = -30.5
r_init = R.from_euler('xyz', [TARGET_ROLL_DEG, TARGET_PITCH_DEG, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

FIXED_JOINTS = {
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': -1.2,  'FR_thigh_joint': -1.2,  'RL_thigh_joint': 1.5,
    'FL_calf_joint': -1.0,   'FR_calf_joint': -1.0,   'RL_calf_joint': -2.5,
}

# 初期状態: 地面にピタリと足先をつけてしっかりしゃがみ込んだ状態
# 足先が地面(Z=0.023m)に接地
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.20)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    **FIXED_JOINTS,
    'RR_thigh_joint': 0.85,
    'RR_calf_joint': -2.35, # 深いしゃがみ込み（タメ）
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/true_body_launch_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=60,
    name_prefix='true_body_launch',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']
joint_names = robot.data.joint_names
rr_thigh_idx = joint_names.index('RR_thigh_joint')
rr_calf_idx = joint_names.index('RR_calf_joint')

print("\n" + "="*75)
print("=== EXPERIMENT: TRUE BODY LAUNCH SINGLE-LEG JUMP ===")
print("=== (FULL GROUND PUSH-OFF -> BALLISTIC BODY TRAJECTORY -> APEX) ===")
print("="*75)

log_data = []

# 制御シーケンス:
# 0.00s ~ 0.06s (3 steps): 地面をしっかり踏みしめてタメ (Squat settle)
# 0.06s ~ 0.16s (5 steps): ★ 地面を押し切るまでフルストローク最大急伸展（Push-Off） ★
#                          足を途中で引っ込めず、完全に伸ばし切って胴体を上方に打ち出す！
# 0.16s ~ 0.35s (10 steps): ★ 弾道飛行（Ballistic Flight: 胴体が上空へ打ち上がり最高点へ） ★
# 0.35s 以降: 着地

for step in range(60):
    t = step * 0.02
    actions = torch.zeros((1, 12), device=env.unwrapped.device)
    
    # 他3脚は完全固定
    for jname, val in FIXED_JOINTS.items():
        j_idx = joint_names.index(jname)
        actions[0, j_idx] = val - robot.data.default_joint_pos[0, j_idx]

    if t < 0.06:
        # タメ
        target_thigh = 0.85
        target_calf = -2.35
    elif t < 0.16:
        # ★ 地面を真下に全力で押し切るフルストロークキック ★
        target_thigh = -0.40
        target_calf = -0.50 # 膝を限界まで伸ばして地面を押し切る
    else:
        # 打ち上がった後は、脚を自然な位置（初期立ち位置）に保持
        # ※ 足を持ち上げるためではなく、着地のために伸ばしたままにしない
        target_thigh = 0.60
        target_calf = -1.60

    actions[0, rr_thigh_idx] = target_thigh - robot.data.default_joint_pos[0, rr_thigh_idx]
    actions[0, rr_calf_idx] = target_calf - robot.data.default_joint_pos[0, rr_calf_idx]

    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    
    # 全身が浮いているか（足先が接地高 0.025m より上か）
    is_airborne = rr_foot[2] > 0.035
    
    log_data.append({
        'step': step, 't': t, 'foot_z': rr_foot[2], 'body_z': root_pos[2],
        'vz': root_vel[2], 'air': is_airborne
    })
    
    status = "★ BALLISTIC AIRBORNE ★" if is_airborne else "Grounded"
    print(f"Step {step:02d} (t={t:.2f}s) | Foot_Z={rr_foot[2]*100:+5.1f}cm | Body_Z={root_pos[2]*100:5.1f}cm | Vz={root_vel[2]:+5.2f}m/s | {status}")

env.close()
simulation_app.close()

# 分析
initial_body_z = log_data[0]['body_z']
max_body_z = max(d['body_z'] for d in log_data)
max_vz = max(d['vz'] for d in log_data)
apex_step = np.argmax([d['body_z'] for d in log_data])

print("\n" + "="*70)
print("=== MEASUREMENT OF TRUE BODY LAUNCH ===")
print(f"Initial Grounded Body Height : {initial_body_z*100:.1f} cm")
print(f"Peak Body Height (Apex)      : {max_body_z*100:.1f} cm (Actual Lift: +{(max_body_z-initial_body_z)*100:.1f} cm)")
print(f"Max Vertical Velocity (Vz)   : {max_vz:+.3f} m/s")
print(f"Apex reached at step {apex_step} (t={apex_step*0.02:.2f}s)")
print("="*70)

# GIF生成
video_path = f"{output_dir}/true_body_launch-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/true_body_launch.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

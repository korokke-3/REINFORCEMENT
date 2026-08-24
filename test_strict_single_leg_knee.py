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
import envs.go2_single_leg_rewards as custom_rewards
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm

env_cfg = Go2SingleLegEnvCfg_PLAY()

# ★ 厳格な即死判定: FL, FR, RL の3脚、胴体、頭部が 1.0N でも触れたら即終了！
env_cfg.terminations.illegal_parts_contact = DoneTerm(
    func=custom_rewards.strict_illegal_contact_termination,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", "FL_.*", "FR_.*", "RL_.*", "base"]),
        "threshold": 1.0,
    },
)

# カメラ設定: 右後脚（RR）1本のみで浮いていることが全方位から見える位置
env_cfg.viewer.eye = (0.0, 1.8, 0.40)
env_cfg.viewer.lookat = (0.0, 0.0, 0.22)

# ★ オプティマイザで算出した「重心がRR膝-足先直線の真上に乗る完全アライメント姿勢」
# Roll = +25.3°, Pitch = -25.2°
r_init = R.from_euler('xyz', [25.3, -25.2, 0], degrees=True)
quat_init = tuple(r_init.as_quat().tolist())

# スポーン高度（RRの膝と足先が地上+2.3cmにピタリとソフト接地し、他3脚は地上+9cmに完全浮上）
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.330)
env_cfg.scene.robot.init_state.rot = quat_init
env_cfg.scene.robot.init_state.joint_pos = {
    # 他3脚: 上空へ完全に引き上げて固定（絶対接地禁止）
    'FL_hip_joint': 0.1,    'FR_hip_joint': -0.1,   'RL_hip_joint': 0.2,
    'FL_thigh_joint': -1.4,  'FR_thigh_joint': -1.4,  'RL_thigh_joint': 1.8,
    'FL_calf_joint': -0.9,   'FR_calf_joint': -0.9,   'RL_calf_joint': -2.6,
    # 支持脚(RR)1本のみ: 膝＋足先完全アライメント
    'RR_hip_joint': 0.05,
    'RR_thigh_joint': 0.42,
    'RR_calf_joint': -1.50,
}

env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0', cfg=env_cfg, render_mode='rgb_array')
output_dir = '/home/exhibition-spakona/Desktop/REINFORCEMENT/strict_single_leg_knee_proof'
os.makedirs(output_dir, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=200, # 4.0秒間
    name_prefix='strict_single_leg_aligned_knee',
)

obs, _ = env.reset()
robot = env.unwrapped.scene['robot']
contact_sensor = env.unwrapped.scene.sensors['contact_forces']

print("\n" + "="*80)
print("=== STRICT EXPERIMENT: 1-LEG ONLY (RR KNEE + FOOT ALIGNED) ===")
print("=== 3-LEGS (FL, FR, RL) ABSOLUTELY AIRBORNE | ZERO ILLEGAL CONTACT ===")
print("="*80)

survived_steps = 0
for step in range(200):
    t = step * 0.02
    actions = torch.zeros((1, 3), device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    base_pos = body_pos[robot.data.body_names.index('base')]
    
    # 他パーツの地上クリアランス（すべて空中に浮いていること）
    other_min_z = min(fl_foot[2], fr_foot[2], rl_foot[2], base_pos[2])
    
    is_strict_1leg = (other_min_z > 0.04) and (not terminated.item())
    status = "★★★ STRICT 1-LEG ONLY ★★★" if is_strict_1leg else "TERMINATED"
    
    if step % 5 == 0 or step < 10 or terminated.item():
        print(f"Step {step:03d} (t={t:.2f}s) | Body_Z={root_pos[2]*100:5.1f}cm | RR_Foot={rr_foot[2]*100:+5.1f}cm | RR_Knee={rr_calf[2]*100:+5.1f}cm | Other_3Legs_Min_Z={other_min_z*100:+5.1f}cm | [{status}]")
    
    if terminated.item():
        print(f"\n>>> TERMINATED by strict violation at step {step} (Survived: {survived_steps} steps = {survived_steps*0.02:.2f}s) <<<")
        break
    survived_steps += 1

env.close()
simulation_app.close()

# GIF生成
video_path = f"{output_dir}/strict_single_leg_aligned_knee-step-0.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        gif_path = f"{output_dir}/strict_single_leg_aligned_knee.gif"
        imageio.mimsave(gif_path, frames, fps=25, loop=0)
        print(f"Saved animation GIF to {gif_path}")

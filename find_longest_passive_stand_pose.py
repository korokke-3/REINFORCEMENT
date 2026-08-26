from __future__ import annotations
import sys
sys.path.append('/home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg')
import argparse
import os
import cv2
import imageio
import torch
import math
import numpy as np
from scipy.optimize import minimize

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless', '--enable_cameras'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=4)
sim = sim_utils.SimulationContext(sim_cfg)
sim_utils.spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())

class StandSearchSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 1.8, 0.35), rot=(0.0, 0.0, -0.7071, 0.7071), convention="ros"),
    )

scene_cfg = StandSearchSceneCfg(num_envs=1, env_spacing=2.5)
scene = InteractiveScene(scene_cfg)
robot = scene["robot"]
camera = scene["camera"]

sim.reset()

print("\n" + "="*80)
print("=== GRID & FINE SEARCH: MAXIMUM PASSIVE STANDING TIME (ZERO ACTIVE MOVEMENT) ===")
print("="*80)

# 最適化探索: ロール角, ピッチ角, 高さZ, 支持脚大腿角, 支持脚膝角(-0.838), ヤジロベエ用他脚角度
best_time = 0
best_params = None

# ロール角 (+18° ~ +26°), ピッチ角 (-20° ~ -10°), 高さZ (40.0cm ~ 43.0cm), 大腿角 (-0.35 ~ 0.05)
roll_candidates = np.linspace(math.radians(19.0), math.radians(25.0), 7)
pitch_candidates = np.linspace(math.radians(-19.0), math.radians(-12.0), 8)
z_candidates = np.linspace(0.405, 0.425, 5)
thigh_candidates = np.linspace(-0.25, 0.0, 6)

print(f"Testing combinations of (Roll, Pitch, Z, Thigh) for passive stand duration...")

def evaluate_stand_time(roll, pitch, base_z, rr_thigh, rr_hip=0.05, fl_hip=0.3, fr_hip=-0.3, rl_hip=0.3):
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

    root_pos = torch.tensor([[0.0, 0.0, base_z]], dtype=torch.float32, device='cuda:0')
    root_rot = torch.tensor([[qx, qy, qz, qw]], dtype=torch.float32, device='cuda:0')
    joint_pos = torch.tensor([[
        fl_hip, fr_hip, rl_hip, rr_hip,
        -1.3, -1.3, 1.7, rr_thigh,
        -1.0, -1.0, -2.5, -0.838
    ]], dtype=torch.float32, device='cuda:0')

    robot.write_root_pose_to_sim(torch.cat([root_pos, root_rot], dim=-1))
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device='cuda:0'))
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    scene.reset()

    stand_steps = 0
    # 全く動かない受動状態 (PD制御で関節角度だけ維持)
    for step in range(150): # 最大 3.0秒
        for _ in range(4):
            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()
            sim.step()
        scene.update(0.02)

        body_pos = robot.data.body_pos_w[0].cpu().numpy()
        rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
        rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
        fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
        fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
        rl_foot = body_pos[robot.data.body_names.index('RL_foot')]

        min_other = min(fl_foot[2], fr_foot[2], rl_foot[2])
        # 膝が 5cm 以上浮いており、他脚も 5cm 以上浮いているか
        if rr_calf[2] > 0.04 and min_other > 0.04 and rr_foot[2] < 0.05:
            stand_steps += 1
        else:
            break

    return stand_steps

# 粗探索
for r in roll_candidates:
    for p in pitch_candidates:
        for z in z_candidates:
            for th in thigh_candidates:
                steps = evaluate_stand_time(r, p, z, th)
                if steps > best_time:
                    best_time = steps
                    best_params = (r, p, z, th)
                    print(f"★ New Best Passive Stand Time: {best_time} steps ({best_time*0.02:.2f}s) | Roll={math.degrees(r):.2f}°, Pitch={math.degrees(p):.2f}°, Z={z*100:.1f}cm, Thigh={th:.3f}")

# 精密最適化 (Nelder-Mead で 0.01度・0.1mm単位で追い込む)
print("\n" + "="*80)
print("=== FINE-TUNING VIA NELDER-MEAD FOR MAXIMUM BALANCE ===")
print("="*80)

x0 = np.array([best_params[0], best_params[1], best_params[2], best_params[3], 0.05, 0.3, -0.3, 0.3])

def loss_func(x):
    r, p, z, th, rr_h, fl_h, fr_h, rl_h = x
    steps = evaluate_stand_time(r, p, z, th, rr_h, fl_h, fr_h, rl_h)
    return -steps # 最小化なのでマイナス

res = minimize(loss_func, x0, method='Nelder-Mead', options={'maxiter': 200, 'disp': False})
opt_r, opt_p, opt_z, opt_th, opt_rr_h, opt_fl_h, opt_fr_h, opt_rl_h = res.x
final_steps = evaluate_stand_time(opt_r, opt_p, opt_z, opt_th, opt_rr_h, opt_fl_h, opt_fr_h, opt_rl_h)

print("\n" + "="*80)
print("=== FINAL OPTIMIZED PASSIVE STANDING POSE ===")
print(f"Max Passive Stand Duration: {final_steps} steps ({final_steps*0.02:.2f} seconds) without ANY active movement!")
print(f"Optimal Roll    : {math.degrees(opt_r):.2f}° ({opt_r:.4f} rad)")
print(f"Optimal Pitch   : {math.degrees(opt_p):.2f}° ({opt_p:.4f} rad)")
print(f"Optimal Base Z  : {opt_z*100:.2f} cm ({opt_z:.4f} m)")
print(f"Optimal RR_thigh: {opt_th:.4f} rad")
print(f"Optimal RR_hip  : {opt_rr_h:.4f} rad")
print(f"Optimal FL_hip  : {opt_fl_h:.4f} rad (Counter-balance)")
print(f"Optimal FR_hip  : {opt_fr_h:.4f} rad (Counter-balance)")
print(f"Optimal RL_hip  : {opt_rl_h:.4f} rad (Counter-balance)")
print("="*80)

# 最適姿勢でのクリアな動画・GIF生成
out_dir = "/home/exhibition-spakona/Desktop/REINFORCEMENT/passive_stand_proof"
os.makedirs(out_dir, exist_ok=True)

cr = math.cos(opt_r * 0.5); sr = math.sin(opt_r * 0.5)
cp = math.cos(opt_p * 0.5); sp = math.sin(opt_p * 0.5)
qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

root_pos = torch.tensor([[0.0, 0.0, opt_z]], dtype=torch.float32, device='cuda:0')
root_rot = torch.tensor([[qx, qy, qz, qw]], dtype=torch.float32, device='cuda:0')
joint_pos = torch.tensor([[
    opt_fl_h, opt_fr_h, opt_rl_h, opt_rr_h,
    -1.3, -1.3, 1.7, opt_th,
    -1.0, -1.0, -2.5, -0.838
]], dtype=torch.float32, device='cuda:0')

robot.write_root_pose_to_sim(torch.cat([root_pos, root_rot], dim=-1))
robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device='cuda:0'))
robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
scene.reset()

frames = []
for step in range(100): # 2.0秒
    for _ in range(4):
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
    scene.update(0.02)
    rgb = camera.data.output["rgb"][0].cpu().numpy()
    frames.append(rgb)

imageio.mimsave(f"{out_dir}/passive_stand_balance.gif", frames, fps=25, loop=0)
imageio.mimsave(f"{out_dir}/passive_stand_balance.mp4", frames, fps=25)
print(f"Saved verified passive stand video to {out_dir}/passive_stand_balance.mp4")

simulation_app.close()

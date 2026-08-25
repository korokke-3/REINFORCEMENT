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

class StraightLegSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 2.2, 0.45), rot=(0.0, 0.0, -0.7071, 0.7071), convention="ros"),
    )

scene_cfg = StraightLegSceneCfg(num_envs=1, env_spacing=2.5)
scene = InteractiveScene(scene_cfg)
robot = scene["robot"]
camera = scene["camera"]

sim.reset()

print("\n" + "="*80)
print("=== NUMERICAL OPTIMIZATION: STRAIGHT-LEG PURE TOE COM ALIGNMENT ===")
print("="*80)

# 足を伸ばした状態での最適化: [Roll, Pitch, Base_Z, RR_thigh, RR_hip]
# 膝 (RR_calf) は伸ばした状態 (-0.84 rad) に固定
thigh_fixed = 0.05
calf_fixed = -0.84 # 足をほぼ真っ直ぐ伸ばした状態 (膝高さ ~25cm)

x0 = np.array([math.radians(24.0), math.radians(-16.0), 0.430, 0.05, 0.05])

def loss_func(x):
    roll, pitch, base_z, thigh, hip = x
    
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp
    
    root_pos = torch.tensor([[0.0, 0.0, base_z]], dtype=torch.float32, device='cuda:0')
    root_rot = torch.tensor([[qx, qy, qz, qw]], dtype=torch.float32, device='cuda:0')
    joint_pos = torch.tensor([[
        0.1, -0.1, 0.2, hip,
        -1.4, -1.4, 1.8, thigh,
        -0.9, -0.9, -2.6, calf_fixed
    ]], dtype=torch.float32, device='cuda:0')
    
    robot.write_root_pose_to_sim(torch.cat([root_pos, root_rot], dim=-1))
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device='cuda:0'))
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    scene.reset()
    scene.update(0.0)
    
    body_pos = robot.data.body_pos_w[0].cpu().numpy()
    rr_foot = body_pos[robot.data.body_names.index('RR_foot')]
    rr_calf = body_pos[robot.data.body_names.index('RR_calf')]
    fl_foot = body_pos[robot.data.body_names.index('FL_foot')]
    fr_foot = body_pos[robot.data.body_names.index('FR_foot')]
    rl_foot = body_pos[robot.data.body_names.index('RL_foot')]
    com = robot.data.root_pos_w[0].cpu().numpy()
    
    # 1. 足先が床 Z=0.020m にピタリと接地
    foot_z_loss = (rr_foot[2] - 0.020)**2 * 3000.0
    
    # 2. 重心が足先ゴム球の真上
    com_x_loss = (com[0] - rr_foot[0])**2 * 1000.0
    com_y_loss = (com[1] - rr_foot[1])**2 * 1000.0
    
    # 3. 膝が 20cm 以上の超高空
    knee_loss = max(0.0, 0.20 - rr_calf[2])**2 * 1000.0
    
    return foot_z_loss + com_x_loss + com_y_loss + knee_loss

res = minimize(loss_func, x0, method='Nelder-Mead', options={'maxiter': 400, 'disp': False})
opt_roll, opt_pitch, opt_z, opt_thigh, opt_hip = res.x

print(f"Optimal Roll    : {math.degrees(opt_roll):.2f}° ({opt_roll:.4f} rad)")
print(f"Optimal Pitch   : {math.degrees(opt_pitch):.2f}° ({opt_pitch:.4f} rad)")
print(f"Optimal Base Z  : {opt_z*100:.2f} cm ({opt_z:.4f} m)")
print(f"Optimal RR_thigh: {opt_thigh:.4f} rad")
print(f"Optimal RR_calf : {calf_fixed:.4f} rad (Straight leg)")
print(f"Optimal RR_hip  : {opt_hip:.4f} rad")

cr = math.cos(opt_roll * 0.5); sr = math.sin(opt_roll * 0.5)
cp = math.cos(opt_pitch * 0.5); sp = math.sin(opt_pitch * 0.5)
qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

root_pos = torch.tensor([[0.0, 0.0, opt_z]], dtype=torch.float32, device='cuda:0')
root_rot = torch.tensor([[qx, qy, qz, qw]], dtype=torch.float32, device='cuda:0')
joint_pos = torch.tensor([[
    0.1, -0.1, 0.2, opt_hip,
    -1.4, -1.4, 1.8, opt_thigh,
    -0.9, -0.9, -2.6, calf_fixed
]], dtype=torch.float32, device='cuda:0')

robot.write_root_pose_to_sim(torch.cat([root_pos, root_rot], dim=-1))
robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device='cuda:0'))
robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
scene.reset()

print("\n" + "="*80)
print("=== TESTING PASSIVE STRAIGHT-LEG TOE BALANCE (NO KICKING, JUST PURE STANDING) ===")
print("="*80)

frames = []
survived_steps = 0
fall_step = None

for step in range(150): # 3.0秒間
    # 関節剛性を保持して足を伸ばしたままにする (PD制御で姿勢維持)
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
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    
    min_other = min(fl_foot[2], fr_foot[2], rl_foot[2])
    is_knee_safe = rr_calf[2] > 0.05
    is_other_safe = min_other > 0.05
    
    status = "★ PERFECT TOE STANDING ★" if (is_knee_safe and is_other_safe) else "Knee / Other Grounded"
    if not (is_knee_safe and is_other_safe) and fall_step is None:
        fall_step = step
        
    print(f"Step {step:03d} (t={step*0.02:.2f}s) | Body_Z={root_pos[2]*100:5.1f}cm | Foot_Z={rr_foot[2]*100:+5.1f}cm | Knee_Z={rr_calf[2]*100:+5.1f}cm | Other_Min_Z={min_other*100:+5.1f}cm | [{status}]")
    
    rgb = camera.data.output["rgb"][0].cpu().numpy()
    frames.append(rgb)

print("\n" + "="*80)
print(f"=== STRAIGHT LEG PASSIVE BALANCE RESULTS ===")
if fall_step is not None:
    print(f"Maintained Pure Toe Stance with Zero Knee Contact for: {fall_step} steps ({fall_step*0.02:.2f} s)")
else:
    print(f"Maintained Pure Toe Stance with Zero Knee Contact for: FULL 150 steps (3.00 s)!")
print("="*80)

out_dir = "/home/exhibition-spakona/Desktop/REINFORCEMENT/straight_leg_toe_proof"
os.makedirs(out_dir, exist_ok=True)
imageio.mimsave(f"{out_dir}/straight_leg_toe_balance.gif", frames, fps=25, loop=0)
imageio.mimsave(f"{out_dir}/straight_leg_toe_balance.mp4", frames, fps=25)
print(f"Saved verified video to {out_dir}/straight_leg_toe_balance.mp4")

simulation_app.close()

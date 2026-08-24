# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 一本足アクティブ・ホッピング (Active Continuous Hopping) 報酬＆終了条件設計

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ----------------------------------------------------------------------
# 1. 報酬関数 (Rewards)
# ----------------------------------------------------------------------

def single_leg_air_time_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.04
) -> torch.Tensor:
    """
    【アクティブ・ホッピング滞空報酬】
    支持脚(RR_foot)が地面を蹴って空中に浮き(Air time)、再び接地した瞬間に滞空時間に応じて特大加点
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    
    # 滞空時間 0.04秒(約2ステップ)〜0.20秒(約10ステップ)のマイクロホップを高く評価
    air_reward = torch.sum(torch.clamp(last_air_time - threshold, min=0.0, max=0.25) * first_contact.float(), dim=1)
    return air_reward * 10.0


def push_off_thrust_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    【接地キック推力報酬】
    接地中に膝(RR_calf)を素早く伸展させて上向きの打ち上げ速度を生み出す動作を評価
    """
    robot: RigidObject = env.scene["robot"]
    calf_vel = robot.data.joint_vel[:, 11] # RR_calf
    root_vz = robot.data.root_lin_vel_w[:, 2] # 鉛直上昇速度
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    is_grounded = (torch.sum(forces, dim=1) > 2.0).float()
    
    # 接地中に膝を伸ばし(+calf_vel) 上向き速度(+root_vz)を出す
    thrust = torch.clamp(calf_vel, min=0.0) * 0.1 + torch.clamp(root_vz, min=0.0) * 2.0
    return thrust * is_grounded


def alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """【生存報酬】倒れずに一本足で耐えている間、毎ステップ確実に加点"""
    return torch.ones(env.num_envs, device=env.device)


def precise_orientation_reward(
    env: ManagerBasedRLEnv,
    target_roll: float = math.radians(28.5),
    target_pitch: float = math.radians(-30.5),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """【重心直下姿勢維持報酬】目標の直立アライメント角をキープ"""
    robot = env.scene[asset_cfg.name]
    proj_g = robot.data.projected_gravity_b
    
    target_gx = math.sin(target_pitch)
    target_gy = -math.sin(target_roll) * math.cos(target_pitch)
    
    gx_err = torch.square(proj_g[:, 0] - target_gx)
    gy_err = torch.square(proj_g[:, 1] - target_gy)
    
    error = gx_err + gy_err
    return torch.exp(-error / 0.15)


def disabled_3legs_retraction_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_height: float = 0.16
) -> torch.Tensor:
    """【3脚の空中保持報酬】支持脚以外の3脚を高くキープ"""
    robot = env.scene[asset_cfg.name]
    foot_pos_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_ok = torch.all(foot_pos_z > min_height, dim=1).float()
    return height_ok


def illegal_contact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """【不正接触ペナルティ】右後足以外のあらゆる接地に対する減点"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    is_contact = torch.sum((forces > 2.0).float(), dim=1)
    return is_contact


def dof_torques_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.scene["robot"].data.applied_torque), dim=1)


# ----------------------------------------------------------------------
# 2. 終了条件 (Terminations)
# ----------------------------------------------------------------------

def strict_illegal_contact_termination(
    env: ManagerBasedRLEnv, threshold: float = 5.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """【転倒接触判定】右後足以外が地面に強く接触した瞬間にリセット"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return torch.any(current_forces > threshold, dim=1)


def orientation_deviation_termination(
    env: ManagerBasedRLEnv,
    max_angle_error: float = 0.60,
    target_roll: float = math.radians(28.5),
    target_pitch: float = math.radians(-30.5),
) -> torch.Tensor:
    """【姿勢崩れ終了判定】ホッピングの許容範囲（35度）を超えて倒れたらリセット"""
    robot = env.scene["robot"]
    proj_g = robot.data.projected_gravity_b
    target_gx = math.sin(target_pitch)
    target_gy = -math.sin(target_roll) * math.cos(target_pitch)
    
    error = torch.sqrt(torch.square(proj_g[:, 0] - target_gx) + torch.square(proj_g[:, 1] - target_gy))
    return error > max_angle_error

# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 "右側2脚歩行 (右前足 + 右後足)" 専用カスタム報酬関数群

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 1. 速度追従報酬 (前進・旋回)
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


# 2. 右側2脚 (FR, RR) の交互ステップ・滞空時間報酬
def right_side_feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.08
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact.float(), dim=1)
    
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return reward * (cmd_norm > 0.1).float()


# 3. 左側脚 (FL, RL) & 胴体の接地ペナルティ (引きずり根絶)
def illegal_left_contacts_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    is_contact = torch.sum((forces > 1.0).float(), dim=1)
    return is_contact


# 4. 左側脚の高さ維持報酬 (前進時にのみ付与)
def left_legs_height_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.20
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    foot_pos_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_error = torch.clamp(target_height - foot_pos_z, min=0.0)
    height_reward = torch.exp(-torch.mean(height_error, dim=1) / 0.05)
    
    robot_vel_x = env.scene["robot"].data.root_lin_vel_b[:, 0]
    is_moving = (torch.abs(robot_vel_x) > 0.15).float()
    return height_reward * is_moving


# 5. 静止ペナルティ
def stand_still_penalty(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", threshold: float = 0.15
) -> torch.Tensor:
    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    actual_speed = torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1)
    is_standing_still = (cmd_speed > 0.2) & (actual_speed < threshold)
    return is_standing_still.float()


# 6. 姿勢・エネルギー制御
def base_height_exp(env: ManagerBasedRLEnv, target_height: float = 0.32, std: float = 0.06) -> torch.Tensor:
    base_pos_z = env.scene["robot"].data.root_pos_w[:, 2]
    error = torch.square(base_pos_z - target_height)
    return torch.exp(-error / std**2)


def right_tilt_orientation_reward(
    env: ManagerBasedRLEnv, target_roll_deg: float = 15.0, std: float = 0.2
) -> torch.Tensor:
    """重心を右脚の真上に乗せるため、やや右に傾いた姿勢を評価"""
    projected_gravity = env.scene["robot"].data.projected_gravity_b
    target_gy = -math.sin(math.radians(target_roll_deg))  # 右傾き
    roll_err = torch.square(projected_gravity[:, 1] - target_gy)
    pitch_err = torch.square(projected_gravity[:, 0])
    return torch.exp(-(roll_err + pitch_err) / std**2)


def lin_vel_z_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.square(env.scene["robot"].data.root_lin_vel_b[:, 2])


def dof_torques_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.scene["robot"].data.applied_torque), dim=1)


def dof_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.scene["robot"].data.joint_acc), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

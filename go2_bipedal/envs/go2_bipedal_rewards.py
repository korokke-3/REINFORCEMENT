# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 "後足2本立ち歩行" 専用カスタム報酬関数群 (修正版: 前進連動ゲーティング)

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 1. 速度追従報酬
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


# 2. 前足浮かせ & 後足歩行 (前進速度連動)
def front_feet_contact_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    is_contact = torch.sum((forces > 1.0).float(), dim=1)
    return is_contact


def front_feet_height_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.30
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    foot_pos_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_error = torch.clamp(target_height - foot_pos_z, min=0.0)
    height_reward = torch.exp(-torch.mean(height_error, dim=1) / 0.05)
    
    # 前進速度ゲーティング: 動いている時のみ報酬
    robot_vel_x = env.scene["robot"].data.root_lin_vel_b[:, 0]
    is_moving = (torch.abs(robot_vel_x) > 0.15).float()
    return height_reward * is_moving


def biped_feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.08
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact.float(), dim=1)
    
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return reward * (cmd_norm > 0.1).float()


# 3. 直立姿勢制御 (前進速度連動)
def biped_base_height_exp(env: ManagerBasedRLEnv, target_height: float = 0.38, std: float = 0.08) -> torch.Tensor:
    base_pos_z = env.scene["robot"].data.root_pos_w[:, 2]
    error = torch.square(base_pos_z - target_height)
    height_reward = torch.exp(-error / std**2)
    
    robot_vel_x = env.scene["robot"].data.root_lin_vel_b[:, 0]
    is_moving = (torch.abs(robot_vel_x) > 0.15).float()
    return height_reward * is_moving


def biped_orientation_reward(
    env: ManagerBasedRLEnv, target_pitch_deg: float = 55.0, std: float = 0.2
) -> torch.Tensor:
    projected_gravity = env.scene["robot"].data.projected_gravity_b
    target_gx = math.sin(math.radians(target_pitch_deg))
    pitch_err = torch.square(projected_gravity[:, 0] - target_gx)
    roll_err = torch.square(projected_gravity[:, 1])
    orient_reward = torch.exp(-(pitch_err + 2.0 * roll_err) / std**2)
    
    # 前進速度ゲーティング: 静止立ちを排除
    robot_vel_x = env.scene["robot"].data.root_lin_vel_b[:, 0]
    is_moving = (torch.abs(robot_vel_x) > 0.15).float()
    return orient_reward * is_moving


# 4. 静止ペナルティ
def stand_still_penalty(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", threshold: float = 0.15
) -> torch.Tensor:
    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    actual_speed = torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1)
    is_standing_still = (cmd_speed > 0.2) & (actual_speed < threshold)
    return is_standing_still.float()


def lin_vel_z_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.square(env.scene["robot"].data.root_lin_vel_b[:, 2])


def dof_torques_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.scene["robot"].data.applied_torque), dim=1)


def dof_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.scene["robot"].data.joint_acc), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

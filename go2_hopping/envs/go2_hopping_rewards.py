# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 "片足けんけん" 専用カスタム報酬関数群

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ----------------------------------------------------------------------
# 1. 速度追従報酬 (Body Frame 基準)
# ----------------------------------------------------------------------

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """コマンドの目標水平速度 (vx, vy) への追従報酬 (ロボット座標系・指数関数)"""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """コマンドの目標旋回角速度 (yaw rate) への追従報酬 (ロボット座標系・指数関数)"""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


# ----------------------------------------------------------------------
# 2. ★ けんけん / 3脚歩行用 カスタム報酬・ペナルティ ★
# ----------------------------------------------------------------------

def disabled_leg_contact_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    【ペナルティ】浮かせるべき脚が地面に接触した場合のペナルティ。
    接触力 (contact force) が閾値を超えた場合に 1.0 (重みが負ならペナルティ) を返します。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    is_contact = torch.sum((forces > 1.0).float(), dim=1)
    return is_contact


def disabled_leg_height_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.15
) -> torch.Tensor:
    """
    【プラス報酬】浮かせるべき脚の地面からの高さを維持する報酬。
    目標の高さ (target_height 以上) に脚が上がっていれば高い報酬を与えます。
    """
    robot = env.scene[asset_cfg.name]
    foot_pos_z = robot.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    height_error = torch.clamp(target_height - foot_pos_z, min=0.0)
    return torch.exp(-height_error / 0.05)


def hopping_feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.08
) -> torch.Tensor:
    """
    【プラス報酬】接地脚の滞空時間報酬 (3本足でリズミカルにホッピング/ケンケンさせる)。
    足が離れている時間が適度に長いと接地時に報酬が加算されます。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact.float(), dim=1)
    
    # 静止コマンド時 (速度指令がほぼ0) は滞空報酬を出さない
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return reward * (cmd_norm > 0.1).float()


# ----------------------------------------------------------------------
# 3. 姿勢・バランス・安定化
# ----------------------------------------------------------------------

def base_height_exp(env: ManagerBasedRLEnv, target_height: float, std: float) -> torch.Tensor:
    """胴体 (base) の高さを適正値 (例: 0.28m) に保つ報酬"""
    base_pos_z = env.scene["robot"].data.root_pos_w[:, 2]
    error = torch.square(base_pos_z - target_height)
    return torch.exp(-error / std**2)


def flat_orientation_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """胴体の傾き (Roll, Pitch) に対するペナルティ (L2ノルム)"""
    projected_gravity = env.scene["robot"].data.projected_gravity_b
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


def lin_vel_z_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """胴体の上下振動 (vz) に対するペナルティ"""
    return torch.square(env.scene["robot"].data.root_lin_vel_b[:, 2])


# ----------------------------------------------------------------------
# 4. エネルギー・アクション抑制
# ----------------------------------------------------------------------

def dof_torques_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """モータトルクの大きさに対するペナルティ (省エネ化)"""
    return torch.sum(torch.square(env.scene["robot"].data.applied_torque), dim=1)


def dof_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """関節加速度に対するペナルティ (急激な動きを抑えて滑らかにする)"""
    return torch.sum(torch.square(env.scene["robot"].data.joint_acc), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """前回の行動との差分ペナルティ (高周波なジッター防止)"""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


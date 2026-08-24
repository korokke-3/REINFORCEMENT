# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 完全一本足アクティブ・バランシング (Strict Single-Leg Active Balancing) 報酬＆終了判定

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

def strict_alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """【一本足生存報酬】他パーツを一切床につけず生き残っている毎ステップに大きな報酬"""
    return torch.ones(env.num_envs, device=env.device)


def strict_posture_alignment_reward(
    env: ManagerBasedRLEnv,
    target_roll: float = math.radians(25.3),
    target_pitch: float = math.radians(-25.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """【精密重心アライメント姿勢報酬】左右(ロール)および前後(ピッチ)の傾きズレを最小化"""
    robot = env.scene[asset_cfg.name]
    proj_g = robot.data.projected_gravity_b
    
    target_gx = math.sin(target_pitch)
    target_gy = -math.sin(target_roll) * math.cos(target_pitch)
    
    gx_err = torch.square(proj_g[:, 0] - target_gx)
    gy_err = torch.square(proj_g[:, 1] - target_gy)
    
    error = gx_err + gy_err
    return torch.exp(-error / 0.08)


def roll_angular_velocity_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """【左右横倒れ角速度ペナルティ】左右への急激な傾き・ブレを抑制"""
    robot = env.scene[asset_cfg.name]
    ang_vel_x = robot.data.root_ang_vel_b[:, 0]
    return torch.square(ang_vel_x)


def disabled_3legs_high_airborne_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_height: float = 0.06
) -> torch.Tensor:
    """【他3脚の完全空中保持報酬】FL, FR, RLが床から十分に離れていることを評価"""
    robot = env.scene[asset_cfg.name]
    foot_pos_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_ok = torch.all(foot_pos_z > min_height, dim=1).float()
    return height_ok


def strict_illegal_contact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """【不正接触ペナルティ】他3脚や胴体が床に触れた瞬間への厳罰"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    is_contact = torch.sum((forces > 1.5).float(), dim=1)
    return is_contact


def dof_torques_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.scene["robot"].data.applied_torque), dim=1)


# ----------------------------------------------------------------------
# 2. 厳格な終了条件 (Terminations)
# ----------------------------------------------------------------------

def absolute_strict_illegal_contact_termination(
    env: ManagerBasedRLEnv, threshold: float = 2.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """【他パーツ接触即死判定】右後脚(RR)以外の部位が床に 2.0N でも触れたら即座にリセット！"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return torch.any(current_forces > threshold, dim=1)


def strict_orientation_deviation_termination(
    env: ManagerBasedRLEnv,
    max_angle_error: float = 0.40,
    target_roll: float = math.radians(25.3),
    target_pitch: float = math.radians(-25.2),
) -> torch.Tensor:
    """【姿勢崩れ終了判定】ロボットが横倒れ・転倒したら即座にリセット"""
    robot = env.scene["robot"]
    proj_g = robot.data.projected_gravity_b
    target_gx = math.sin(target_pitch)
    target_gy = -math.sin(target_roll) * math.cos(target_pitch)
    
    error = torch.sqrt(torch.square(proj_g[:, 0] - target_gx) + torch.square(proj_g[:, 1] - target_gy))
    return error > max_angle_error

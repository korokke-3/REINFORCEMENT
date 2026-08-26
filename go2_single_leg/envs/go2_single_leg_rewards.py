# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 真の足先ゴム球一本足ジャンプ報酬関数 (True Pure Toe Jump Rewards)

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

def explosive_toe_launch_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    【足先ゴム球 爆発的鉛直打ち上げ報酬】
    床を蹴って上向き速度 (Vz > +0.3 m/s) で力強く飛び上がった瞬間に巨大ボーナス
    """
    robot: RigidObject = env.scene["robot"]
    root_vz = robot.data.root_lin_vel_w[:, 2]
    
    # 鉛直上昇速度 (+0.25 m/s 超) を強力にブースト
    rebound_power = torch.clamp(root_vz - 0.25, min=0.0) * 20.0
    return rebound_power


def pure_toe_high_flight_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_clearance: float = 0.04
) -> torch.Tensor:
    """
    【完全足先滞空クリアランス報酬】
    支持脚 (RR_foot) が床から 4cm〜35cm 以上浮いて飛行している時間に高得点
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    rr_foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    
    air_clearance = torch.clamp(rr_foot_z - min_clearance, min=0.0, max=0.35)
    return air_clearance * 30.0


def ground_stagnation_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """【接地引きずり・寝そべり禁止ペナルティ】"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    is_grounded = (torch.sum(forces, dim=1) > 1.0).float()
    return is_grounded


def pure_toe_alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """【生存ボーナス】"""
    return torch.ones(env.num_envs, device=env.device) * 5.0


def disabled_3legs_high_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_height: float = 0.12
) -> torch.Tensor:
    """【他3脚の高空格納保持報酬】"""
    robot: RigidObject = env.scene[asset_cfg.name]
    foot_pos_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_ok = torch.all(foot_pos_z > min_height, dim=1).float()
    return height_ok * 8.0


def heavy_illegal_contact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """【膝・胴体・他脚接触の超巨大ペナルティ（足先のみを強制）】"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    # 接触力に応じた連続的かつ厳しいペナルティ
    is_contact = torch.sum(torch.clamp(forces - 1.0, min=0.0), dim=1)
    return is_contact


# ----------------------------------------------------------------------
# 2. 終了条件 (Terminations)
# ----------------------------------------------------------------------

def pure_toe_catastrophic_fall_termination(
    env: ManagerBasedRLEnv,
    max_angle_error: float = 1.10,
    target_roll: float = math.radians(20.0),
    target_pitch: float = math.radians(-12.0),
) -> torch.Tensor:
    """【完全転倒（姿勢崩壊）終了判定】"""
    robot = env.scene["robot"]
    proj_g = robot.data.projected_gravity_b
    target_gx = math.sin(target_pitch)
    target_gy = -math.sin(target_roll) * math.cos(target_pitch)
    
    error = torch.sqrt(torch.square(proj_g[:, 0] - target_gx) + torch.square(proj_g[:, 1] - target_gy))
    return error > max_angle_error

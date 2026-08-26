# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 純粋足先ゴム球一本足バウンディング (Pure Toe Continuous Dynamic Hopping)

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
# 1. Raibert Hopper 型 足先配置 & 弾性リバウンド報酬
# ----------------------------------------------------------------------

def pure_toe_explosive_rebound_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    【足先ゴム球 弾性リバウンド打ち上げ報酬】
    足先が接地した瞬間に、床を強く蹴り上げて上向き速度 (Vz > +0.3 m/s) で跳ね上がる動作に高得点
    """
    robot: RigidObject = env.scene["robot"]
    root_vz = robot.data.root_lin_vel_w[:, 2]
    
    # 鉛直上昇速度 (+0.2 m/s 超) に応じて報酬
    rebound_power = torch.clamp(root_vz - 0.20, min=0.0) * 15.0
    return rebound_power


def pure_toe_air_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_clearance: float = 0.04
) -> torch.Tensor:
    """
    【完全足先滞空クリアランス報酬】
    支持脚 (RR_foot) が床から 4cm〜25cm 以上浮いて飛行している時間を高く評価
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    rr_foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    
    air_clearance = torch.clamp(rr_foot_z - min_clearance, min=0.0, max=0.25)
    return air_clearance * 25.0


def raibert_foot_placement_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    【Raibert Hopper型 着地足先配置報酬】
    空中にいる間、足先ゴム球 (RR_foot) の (X, Y) を重心 (CoM) の直下にピタリと合わせる動作に加点
    (これにより、着地した瞬間の転倒モーメントをゼロにする！)
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    com_pos = robot.data.root_pos_w[:, :2] # (N, 2)
    rr_foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :2] # (N, 2)
    
    # 重心と足先の水平オフセット距離
    offset_dist = torch.norm(com_pos - rr_foot_pos, dim=-1)
    # 距離が 5cm 以内なら最大加点
    placement_score = torch.exp(-torch.square(offset_dist / 0.08))
    return placement_score * 10.0


def midair_arm_balance_orientation_reward(
    env: ManagerBasedRLEnv, target_roll: float = math.radians(22.0), target_pitch: float = math.radians(-16.0)
) -> torch.Tensor:
    """
    【空中姿勢アクティブ補正報酬】
    目標の安定ロール・ピッチ角度を維持していることにボーナス
    """
    robot: RigidObject = env.scene["robot"]
    proj_g = robot.data.projected_gravity_b
    target_gx = math.sin(target_pitch)
    target_gy = -math.sin(target_roll) * math.cos(target_pitch)
    
    error = torch.sqrt(torch.square(proj_g[:, 0] - target_gx) + torch.square(proj_g[:, 1] - target_gy))
    return torch.exp(-error * 3.0) * 8.0


def pure_toe_ground_stagnation_penalty(
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


def disabled_3legs_high_airborne_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_height: float = 0.12
) -> torch.Tensor:
    """【他3脚の超高空格納保持報酬】"""
    robot: RigidObject = env.scene[asset_cfg.name]
    foot_pos_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_ok = torch.all(foot_pos_z > min_height, dim=1).float()
    return height_ok * 5.0


def strict_illegal_contact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """【膝・胴体・他脚接触の超厳格ペナルティ】"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    is_contact = torch.sum((forces > 1.0).float(), dim=1)
    return is_contact


# ----------------------------------------------------------------------
# 2. 厳格な終了条件 (Terminations)
# ----------------------------------------------------------------------

def pure_toe_strict_illegal_contact_termination(
    env: ManagerBasedRLEnv, threshold: float = 3.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """【他部位接触即死判定】膝や他部位が床に 3.0N でも触れたら即リセット！"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return torch.any(current_forces > threshold, dim=1)


def pure_toe_orientation_deviation_termination(
    env: ManagerBasedRLEnv,
    max_angle_error: float = 0.70,
    target_roll: float = math.radians(22.0),
    target_pitch: float = math.radians(-16.0),
) -> torch.Tensor:
    """【姿勢大崩れ即死判定】"""
    robot = env.scene["robot"]
    proj_g = robot.data.projected_gravity_b
    target_gx = math.sin(target_pitch)
    target_gy = -math.sin(target_roll) * math.cos(target_pitch)
    
    error = torch.sqrt(torch.square(proj_g[:, 0] - target_gx) + torch.square(proj_g[:, 1] - target_gy))
    return error > max_angle_error

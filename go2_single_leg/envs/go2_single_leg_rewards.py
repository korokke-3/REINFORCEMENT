# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 真の一本足けんけん (Strict Single-Leg Forward Hopping) 報酬＆終了条件設計

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
# 1. けんけん報酬関数 (Forward Hopping Rewards)
# ----------------------------------------------------------------------

def single_leg_hop_air_time_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.04
) -> torch.Tensor:
    """
    【けんけん滞空時間報酬】
    支持脚(RR_foot)が床を蹴って体全体が空中に舞い上がり(Air time)、着地した瞬間に滞空時間を評価
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    
    air_reward = torch.sum(torch.clamp(last_air_time - threshold, min=0.0, max=0.25) * first_contact.float(), dim=1)
    return air_reward * 15.0


def single_leg_forward_thrust_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    【けんけん前進キック推力報酬】
    接地中に支持脚(RR)で床を後ろ・下向きに蹴り、前進速度(Vx > 0)と上向き速度(Vz > 0)を生み出す動作を評価
    """
    robot: RigidObject = env.scene["robot"]
    root_vx = robot.data.root_lin_vel_b[:, 0] # 機体前方速度
    root_vz = robot.data.root_lin_vel_w[:, 2] # 鉛直上昇速度
    calf_vel = robot.data.joint_vel[:, 11]   # 膝伸展速度
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    is_grounded = (torch.sum(forces, dim=1) > 2.0).float()
    
    thrust = torch.clamp(root_vx, min=0.0) * 2.0 + torch.clamp(root_vz, min=0.0) * 1.5 + torch.clamp(calf_vel, min=0.0) * 0.1
    return thrust * is_grounded


def track_forward_vel_exp(
    env: ManagerBasedRLEnv, std: float = 0.5, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """【前進速度追従報酬】目標の前進速度(Vx = +0.6 m/s)に向かってピョンピョン進むことを評価"""
    robot: RigidObject = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)
    lin_vel_err = torch.square(commands[:, 0] - robot.data.root_lin_vel_b[:, 0])
    return torch.exp(-lin_vel_err / (std**2))


def single_leg_alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """【けんけん生存報酬】一本足で跳び続けている毎ステップに報酬"""
    return torch.ones(env.num_envs, device=env.device)


def disabled_3legs_high_airborne_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_height: float = 0.08
) -> torch.Tensor:
    """【他3脚の完全空中保持報酬】FL, FR, RL が高く抱え込まれていることを評価"""
    robot: RigidObject = env.scene[asset_cfg.name]
    foot_pos_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_ok = torch.all(foot_pos_z > min_height, dim=1).float()
    return height_ok


def strict_illegal_contact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """【他パーツ接地ペナルティ】他3脚や胴体が床に触れた瞬間への厳罰"""
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
    env: ManagerBasedRLEnv, threshold: float = 3.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """【他パーツ接触即死判定】右後脚(RR)以外の部位が床に 3.0N でも触れたら即座にリセット！"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return torch.any(current_forces > threshold, dim=1)


def hopping_orientation_deviation_termination(
    env: ManagerBasedRLEnv,
    max_angle_error: float = 0.65,
    target_roll: float = math.radians(25.3),
    target_pitch: float = math.radians(-20.0),
) -> torch.Tensor:
    """【姿勢崩れ終了判定】けんけん中のピッチ・ロール許容角(約37度)を超えて転倒したらリセット"""
    robot = env.scene["robot"]
    proj_g = robot.data.projected_gravity_b
    target_gx = math.sin(target_pitch)
    target_gy = -math.sin(target_roll) * math.cos(target_pitch)
    
    error = torch.sqrt(torch.square(proj_g[:, 0] - target_gx) + torch.square(proj_g[:, 1] - target_gy))
    return error > max_angle_error

# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 真の一本足爆発ジャンプ (True Single-Leg Explosive Jump & Flight) 報酬＆終了判定

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
# 1. 爆発的ジャンプ報酬関数 (Explosive Jump Rewards)
# ----------------------------------------------------------------------

def explosive_vertical_launch_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    【爆発的鉛直打ち上げ報酬】
    膝を一気に伸展させて、体全体を上空(Vz > +0.4 m/s)へ打ち出す爆発的キック動作に特大ボーナス
    """
    robot: RigidObject = env.scene["robot"]
    root_vz = robot.data.root_lin_vel_w[:, 2] # 鉛直上昇速度
    calf_vel = robot.data.joint_vel[:, 11]   # 膝伸展角速度
    
    # 鉛直上昇速度が +0.3 m/s を超えた分に特大加点
    launch_power = torch.clamp(root_vz - 0.2, min=0.0) * 10.0 + torch.clamp(calf_vel - 1.0, min=0.0) * 0.2
    return launch_power


def flight_air_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_clearance: float = 0.05
) -> torch.Tensor:
    """
    【完全空中クリアランス報酬】
    支持脚(RR_foot)が床から 5cm〜20cm 以上宙に浮いて完全飛行している状態を高く評価
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    rr_foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    
    # 足先が床から 5cm 以上離れているほど得点
    air_clearance = torch.clamp(rr_foot_z - min_clearance, min=0.0, max=0.20)
    return air_clearance * 20.0


def ground_stagnation_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    【スライディング・接地引きずり禁止ペナルティ】
    足を床につけたままにしている間、毎ステップ強力な減点を科す (スライディング逃げ道を封鎖！)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    is_grounded = (torch.sum(forces, dim=1) > 1.0).float()
    return is_grounded


def single_leg_alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """【一本足生存報酬】"""
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
    """【他パーツ接地ペナルティ】"""
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
    max_angle_error: float = 0.70,
    target_roll: float = math.radians(25.3),
    target_pitch: float = math.radians(-20.0),
) -> torch.Tensor:
    """【姿勢崩れ終了判定】"""
    robot = env.scene["robot"]
    proj_g = robot.data.projected_gravity_b
    target_gx = math.sin(target_pitch)
    target_gy = -math.sin(target_roll) * math.cos(target_pitch)
    
    error = torch.sqrt(torch.square(proj_g[:, 0] - target_gx) + torch.square(proj_g[:, 1] - target_gy))
    return error > max_angle_error
